import { attention, matmul, softmax, transpose, type Matrix } from './attention';

/*
 * KV 架构对照：缓存字节账本、MHA / GQA / MQA 的 K/V 头共享，以及 MLA 的低秩压缩、
 * 解耦 RoPE 与矩阵吸收。所有向量都是行向量（X W），与本项目其他页面一致；
 * DeepSeek-V2 论文用列向量（W h），公式在页面上转写为行向量。
 */

// ---------------------------------------------------------------- 缓存字节

export interface KVCacheConfig {
  batch: number;
  layers: number;
  tokens: number;
  kvHeads: number;
  dK: number;
  dV: number;
  bytesPerElement: number;
}

function positiveInts(values: Record<string, number>) {
  for (const [key, value] of Object.entries(values))
    if (!Number.isInteger(value) || value <= 0) throw new Error(`${key} 必须是正整数。`);
}

/** batch × layers × tokens × kv_heads × (d_k + d_v)：只统计 KV 缓存的元素个数。 */
export function kvCacheElements(c: KVCacheConfig): number {
  positiveInts({ ...c });
  return c.batch * c.layers * c.tokens * c.kvHeads * (c.dK + c.dV);
}
export function kvCacheBytes(c: KVCacheConfig): number {
  return kvCacheElements(c) * c.bytesPerElement;
}

export interface MLACacheConfig {
  batch: number;
  layers: number;
  tokens: number;
  /** d_c：KV 联合压缩后的 latent 维度。 */
  latent: number;
  /** d_h^R：解耦 RoPE 的共享 Key 维度，同样要缓存。 */
  ropeDim: number;
  bytesPerElement: number;
}
/** MLA 每个 token 每层缓存 latent c 与共享的 RoPE Key k^R，两者都要算进去。 */
export function mlaCacheElements(c: MLACacheConfig): number {
  positiveInts({ ...c });
  return c.batch * c.layers * c.tokens * (c.latent + c.ropeDim);
}
export function mlaCacheBytes(c: MLACacheConfig): number {
  return mlaCacheElements(c) * c.bytesPerElement;
}

/** Binary units (KiB = 1024 B). Exact values keep no decimals, others keep up to 2. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) throw new Error('字节数必须是非负有限值。');
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const text = Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
  return `${text} ${units[unit]}`;
}

/** Attention projection parameters (no bias). Only W_K and W_V depend on the KV head count. */
export function projectionParams(p: {
  dModel: number;
  qHeads: number;
  kvHeads: number;
  dHead: number;
}) {
  positiveInts(p);
  if (p.qHeads % p.kvHeads) throw new Error('Q 头数必须能被 KV 头数整除。');
  const q = p.dModel * p.qHeads * p.dHead;
  const k = p.dModel * p.kvHeads * p.dHead;
  return { q, k, v: k, o: p.qHeads * p.dHead * p.dModel, total: 2 * q + 2 * k };
}

// ---------------------------------------------------------------- 教学数字

/** Deterministic half-step values in {−1, −0.5, 0, 0.5, 1}; teaching numbers, not trained weights. */
function seeded(rows: number, cols: number, seed: number): Matrix {
  let state = seed >>> 0;
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return ((state >>> 16) % 5) / 2 - 1;
    }),
  );
}

export const TOKENS = 4;
export const D_MODEL = 4;
export const Q_HEADS = 4;
export const D_HEAD = 2;
export const KV_OPTIONS = [4, 2, 1] as const;
export type KVHeads = (typeof KV_OPTIONS)[number];

/** 4 个位置的隐藏状态 X：[T, D]。 */
export const X: Matrix = [
  [1, 0.5, -0.5, 0],
  [0, 1, 0.5, -1],
  [-1, 0.5, 1, 0.5],
  [0.5, -0.5, 0, 1],
];

export interface MHAWeights {
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
  wo: Matrix;
}
/** MHA weights: W_Q, W_K, W_V are [D, H·d]; head h owns columns [h·d, (h+1)·d). W_O is [H·d, D]. */
export const MHA_WEIGHTS: MHAWeights = {
  wq: seeded(D_MODEL, Q_HEADS * D_HEAD, 11),
  wk: seeded(D_MODEL, Q_HEADS * D_HEAD, 23),
  wv: seeded(D_MODEL, Q_HEADS * D_HEAD, 37),
  wo: seeded(Q_HEADS * D_HEAD, D_MODEL, 41),
};

const block = (m: Matrix, start: number, width: number) =>
  m.map((row) => row.slice(start, start + width));

// ---------------------------------------------------------------- MHA / GQA / MQA

/** Query head `head` reads KV head ⌊head / (H / G)⌋: consecutive query heads share one group. */
export function groupOf(head: number, qHeads: number, kvHeads: number): number {
  if (qHeads % kvHeads) throw new Error('Q 头数必须能被 KV 头数整除。');
  if (!Number.isInteger(head) || head < 0 || head >= qHeads) throw new Error('Q 头编号超出范围。');
  return Math.floor(head / (qHeads / kvHeads));
}

/**
 * Convert per-head K or V projections into `kvHeads` shared ones by averaging the heads of each
 * group (the mean pooling used to convert MHA checkpoints in the GQA paper; the paper then keeps
 * training, which this page does not do).
 */
export function poolHeads(weights: Matrix, qHeads: number, kvHeads: number, dHead: number): Matrix {
  const size = qHeads / kvHeads;
  if (!Number.isInteger(size)) throw new Error('Q 头数必须能被 KV 头数整除。');
  return weights.map((row) =>
    Array.from({ length: kvHeads * dHead }, (_, c) => {
      const group = Math.floor(c / dHead);
      const d = c % dHead;
      let sum = 0;
      for (let h = group * size; h < (group + 1) * size; h++) sum += row[h * dHead + d];
      return sum / size;
    }),
  );
}

export interface GroupedResult {
  kvHeads: number;
  q: Matrix[];
  k: Matrix[];
  v: Matrix[];
  /** Per query head: [T, T] weights with zeros above the diagonal. */
  probs: Matrix[];
  heads: Matrix[];
  output: Matrix;
  /** Numbers held in the cache per token (one layer): K and V of every KV head. */
  cachePerToken: number;
  /** Query-key dot products over the causal triangle, all query heads. */
  scorePairs: number;
}

/** Grouped-query attention: each query head looks up the K/V of its group, nothing is copied. */
export function groupedAttention(
  x: Matrix,
  w: MHAWeights,
  kvHeads: number,
  causal = true,
): GroupedResult {
  const wk = poolHeads(w.wk, Q_HEADS, kvHeads, D_HEAD);
  const wv = poolHeads(w.wv, Q_HEADS, kvHeads, D_HEAD);
  const qAll = matmul(x, w.wq);
  const kAll = matmul(x, wk);
  const vAll = matmul(x, wv);
  const q = Array.from({ length: Q_HEADS }, (_, h) => block(qAll, h * D_HEAD, D_HEAD));
  const k = Array.from({ length: kvHeads }, (_, g) => block(kAll, g * D_HEAD, D_HEAD));
  const v = Array.from({ length: kvHeads }, (_, g) => block(vAll, g * D_HEAD, D_HEAD));
  let scorePairs = 0;
  const root = Math.sqrt(D_HEAD);
  const probs: Matrix[] = [];
  const heads: Matrix[] = [];
  q.forEach((qh, h) => {
    const g = groupOf(h, Q_HEADS, kvHeads);
    const p = qh.map((qi, i) => {
      const row = k[g].map((kj, j) => {
        if (causal && j > i) return -Infinity;
        scorePairs += 1;
        return qi.reduce((sum, value, d) => sum + value * kj[d], 0) / root;
      });
      return softmax(row);
    });
    probs.push(p);
    heads.push(matmul(p, v[g]));
  });
  const concat = x.map((_, t) => heads.flatMap((head) => head[t]));
  return {
    kvHeads,
    q,
    k,
    v,
    probs,
    heads,
    output: matmul(concat, w.wo),
    cachePerToken: kvHeads * 2 * D_HEAD,
    scorePairs,
  };
}

/**
 * Correctness baseline only: copy each shared K/V head once per query head, then run ordinary
 * per-head attention with the independent reference in attention.ts. Real implementations do not
 * store these copies.
 */
export function repeatedKVReference(x: Matrix, w: MHAWeights, kvHeads: number, causal = true) {
  const wk = poolHeads(w.wk, Q_HEADS, kvHeads, D_HEAD);
  const wv = poolHeads(w.wv, Q_HEADS, kvHeads, D_HEAD);
  const repeat = (m: Matrix) =>
    m.map((row) =>
      Array.from({ length: Q_HEADS }, (_, h) => {
        const g = groupOf(h, Q_HEADS, kvHeads);
        return row.slice(g * D_HEAD, (g + 1) * D_HEAD);
      }).flat(),
    );
  const q = matmul(x, w.wq);
  const k = matmul(x, repeat(wk));
  const v = matmul(x, repeat(wv));
  const heads = Array.from(
    { length: Q_HEADS },
    (_, h) =>
      attention(
        block(q, h * D_HEAD, D_HEAD),
        block(k, h * D_HEAD, D_HEAD),
        block(v, h * D_HEAD, D_HEAD),
        causal,
        true,
      ).output,
  );
  const concat = x.map((_, t) => heads.flatMap((head) => head[t]));
  return matmul(concat, w.wo);
}

// ---------------------------------------------------------------- MLA

export const D_LATENT = 3;
export const D_ROPE = 2;

/**
 * MLA weights in row-vector form. Per head i, the W_UK / W_UV / W_QC / W_QR columns
 * [i·d, (i+1)·d) belong to head i; W_O rows [i·d, (i+1)·d) multiply head i's output.
 * Query compression (W^DQ) is omitted: in the paper it reduces training activations, not the cache.
 */
export interface MLAWeights {
  /** [D, d_c]：联合压缩 K、V 的下投影。 */
  wdkv: Matrix;
  /** [d_c, H·d]：由 latent 还原各头内容 Key。 */
  wuk: Matrix;
  /** [d_c, H·d]：由 latent 还原各头 Value。 */
  wuv: Matrix;
  /** [D, d_R]：解耦 RoPE 的共享 Key，直接由 h 投影。 */
  wkr: Matrix;
  /** [D, H·d]：各头内容 Query。 */
  wqc: Matrix;
  /** [D, H·d_R]：各头 RoPE Query。 */
  wqr: Matrix;
  /** [H·d, D]。 */
  wo: Matrix;
}
export const MLA_WEIGHTS: MLAWeights = {
  wdkv: seeded(D_MODEL, D_LATENT, 53),
  wuk: seeded(D_LATENT, Q_HEADS * D_HEAD, 67),
  wuv: seeded(D_LATENT, Q_HEADS * D_HEAD, 71),
  wkr: seeded(D_MODEL, D_ROPE, 83),
  wqc: seeded(D_MODEL, Q_HEADS * D_HEAD, 97),
  wqr: seeded(D_MODEL, Q_HEADS * D_ROPE, 101),
  wo: seeded(Q_HEADS * D_HEAD, D_MODEL, 113),
};

/** RoPE for a 2-dim row vector at position `pos`, θ = 1 rad per position: [x cos − y sin, x sin + y cos]. */
export function rope2(vector: number[], pos: number): number[] {
  if (vector.length !== 2) throw new Error('本页的 RoPE 只作用于二维向量。');
  const [x, y] = vector;
  return [x * Math.cos(pos) - y * Math.sin(pos), x * Math.sin(pos) + y * Math.cos(pos)];
}

const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);

export interface MLACache {
  /** c_t = h_t W_DKV：[T, d_c]。 */
  latent: Matrix;
  /** k^R_t = RoPE_t(h_t W_KR)：[T, d_R]，所有头共享。 */
  ropeKey: Matrix;
}
/** What MLA keeps for every past token: the latent and the shared RoPE key. */
export function mlaCache(x: Matrix, w: MLAWeights): MLACache {
  return {
    latent: matmul(x, w.wdkv),
    ropeKey: matmul(x, w.wkr).map((row, t) => rope2(row, t)),
  };
}
export const mlaCachePerToken = () => D_LATENT + D_ROPE;

export interface MLAExpanded {
  cache: MLACache;
  /** Per head, restored from the latent: [T, d]. */
  kC: Matrix[];
  vC: Matrix[];
  qC: Matrix[];
  qR: Matrix[];
  /** Per head: scaled scores [T, T] (−∞ above the diagonal) and weights. */
  scores: Matrix[];
  probs: Matrix[];
  heads: Matrix[];
  output: Matrix;
}

const SCALE = Math.sqrt(D_HEAD + D_ROPE);

/** Textbook path: restore every head's K and V from the latent, then ordinary attention. */
export function mlaExpanded(x: Matrix, w: MLAWeights): MLAExpanded {
  const cache = mlaCache(x, w);
  const kAll = matmul(cache.latent, w.wuk);
  const vAll = matmul(cache.latent, w.wuv);
  const qAll = matmul(x, w.wqc);
  const qrAll = matmul(x, w.wqr);
  const kC = Array.from({ length: Q_HEADS }, (_, i) => block(kAll, i * D_HEAD, D_HEAD));
  const vC = Array.from({ length: Q_HEADS }, (_, i) => block(vAll, i * D_HEAD, D_HEAD));
  const qC = Array.from({ length: Q_HEADS }, (_, i) => block(qAll, i * D_HEAD, D_HEAD));
  const qR = Array.from({ length: Q_HEADS }, (_, i) =>
    block(qrAll, i * D_ROPE, D_ROPE).map((row, t) => rope2(row, t)),
  );
  const scores = qC.map((qh, i) =>
    qh.map((qt, t) =>
      x.map((_, j) =>
        j > t ? -Infinity : (dot(qt, kC[i][j]) + dot(qR[i][t], cache.ropeKey[j])) / SCALE,
      ),
    ),
  );
  const probs = scores.map((head) => head.map(softmax));
  const heads = probs.map((p, i) => matmul(p, vC[i]));
  const concat = x.map((_, t) => heads.flatMap((head) => head[t]));
  return { cache, kC, vC, qC, qR, scores, probs, heads, output: matmul(concat, w.wo) };
}

export interface MLAAbsorbed {
  /** W̃_Q^(i) = W_QC^(i) (W_UK^(i))ᵀ：[D, d_c]，把还原 Key 的矩阵并进 Query。 */
  absorbedQ: Matrix[];
  /** W̃_O^(i) = W_UV^(i) W_O^(i)：[d_c, D]，把还原 Value 的矩阵并进输出投影。 */
  absorbedO: Matrix[];
  /** q̃_t^(i) = h_t W̃_Q^(i)：直接与缓存里的 latent 做点积。 */
  qLatent: Matrix[];
  scores: Matrix[];
  probs: Matrix[];
  /** Σ_j p_j c_j：仍在 latent 空间里的加权和，[T, d_c]。 */
  contextLatent: Matrix[];
  output: Matrix;
}

/**
 * Inference path after absorption: keys and values are never restored. Scores use the cached
 * latent and the cached RoPE key; the output projection reads the weighted latent directly.
 */
export function mlaAbsorbed(
  x: Matrix,
  w: MLAWeights,
  cache: MLACache = mlaCache(x, w),
): MLAAbsorbed {
  const absorbedQ = Array.from({ length: Q_HEADS }, (_, i) =>
    matmul(block(w.wqc, i * D_HEAD, D_HEAD), transpose(block(w.wuk, i * D_HEAD, D_HEAD))),
  );
  const absorbedO = Array.from({ length: Q_HEADS }, (_, i) =>
    matmul(block(w.wuv, i * D_HEAD, D_HEAD), w.wo.slice(i * D_HEAD, (i + 1) * D_HEAD)),
  );
  const qrAll = matmul(x, w.wqr);
  const qLatent = absorbedQ.map((m) => matmul(x, m));
  const scores = qLatent.map((qh, i) =>
    qh.map((qt, t) => {
      const qr = rope2(block(qrAll, i * D_ROPE, D_ROPE)[t], t);
      return x.map((_, j) =>
        j > t ? -Infinity : (dot(qt, cache.latent[j]) + dot(qr, cache.ropeKey[j])) / SCALE,
      );
    }),
  );
  const probs = scores.map((head) => head.map(softmax));
  const contextLatent = probs.map((p) => matmul(p, cache.latent));
  const output = x.map((_, t) =>
    Array.from({ length: D_MODEL }, (_, d) =>
      contextLatent.reduce(
        (sum, latent, i) =>
          sum +
          dot(
            latent[t],
            absorbedO[i].map((row) => row[d]),
          ),
        0,
      ),
    ),
  );
  return { absorbedQ, absorbedO, qLatent, scores, probs, contextLatent, output };
}

export interface CoupledRow {
  key: number;
  distance: number;
  /** q·k when RoPE is applied to the restored content key itself. */
  exact: number;
  /** The same score with one position-free absorbed matrix (what absorption would need). */
  absorbed: number;
  error: number;
}

/**
 * Counter-example: put RoPE directly on the content Q and the restored content K (no separate
 * branch). Then q·k = q R_t R_jᵀ W_UKᵀ cᵀ, and the matrix between q and c depends on t − j.
 * A single absorbed matrix W_UKᵀ is exact only where t = j.
 */
export function coupledRopeRows(x: Matrix, w: MLAWeights, head: number, t: number): CoupledRow[] {
  const latent = matmul(x, w.wdkv);
  const wuk = block(w.wuk, head * D_HEAD, D_HEAD);
  const q = matmul(x, block(w.wqc, head * D_HEAD, D_HEAD))[t];
  const qRot = rope2(q, t);
  const absorbedQuery = matmul([q], transpose(wuk))[0];
  return latent.slice(0, t + 1).map((c, j) => {
    const k = rope2(matmul([c], wuk)[0], j);
    const exact = dot(qRot, k);
    const absorbed = dot(absorbedQuery, c);
    return { key: j, distance: t - j, exact, absorbed, error: Math.abs(exact - absorbed) };
  });
}

// ---------------------------------------------------------------- 真实规模对照（按公式）

/** DeepSeek-V2 attention hyper-parameters from the paper (§3.1.2, Table 1 caption). */
export const DEEPSEEK_V2 = {
  layers: 60,
  heads: 128,
  dHead: 128,
  latent: 512,
  ropeDim: 64,
} as const;
