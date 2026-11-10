import { attention, softmax, type Matrix } from './attention';
import { T } from '../i18n';

export const PE_BASE = 10000;

function checkVector(vector: number[], name: string) {
  if (!vector.length || vector.some((x) => !Number.isFinite(x)))
    throw new Error(`${name} 必须是只含有限数值的非空向量。`);
}
export function dot(a: number[], b: number[]): number {
  checkVector(a, T('向量'));
  checkVector(b, T('向量'));
  if (a.length !== b.length) throw new Error('点积的两个向量长度必须相同。');
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}
export function norm(vector: number[]): number {
  return Math.sqrt(dot(vector, vector));
}
export function addVectors(a: number[], b: number[]): number[] {
  if (a.length !== b.length) throw new Error('相加的两个向量长度必须相同。');
  return a.map((x, i) => x + b[i]);
}

// ---------------------------------------------------------------- Sinusoidal (第 5 章)

/** ω_i = base^(−2i/d_model): dimension pair i (dims 2i and 2i+1) turns ω_i radians per position. */
export function sinusoidalFrequency(i: number, dModel: number, base = PE_BASE): number {
  if (!Number.isInteger(dModel) || dModel < 2 || dModel % 2)
    throw new Error('d_model 必须是正偶数。');
  if (!Number.isInteger(i) || i < 0 || i >= dModel / 2) throw new Error('维度对编号超出范围。');
  return Math.pow(base, (-2 * i) / dModel);
}

/** PE(pos, 2i) = sin(pos·ω_i), PE(pos, 2i+1) = cos(pos·ω_i); positions start from 0. */
export function sinusoidalPE(pos: number, dModel: number, base = PE_BASE): number[] {
  if (!Number.isInteger(pos) || pos < 0) throw new Error('位置必须是非负整数。');
  return Array.from({ length: dModel }, (_, j) => {
    const angle = pos * sinusoidalFrequency(Math.floor(j / 2), dModel, base);
    return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
}

export function sinusoidalTable(positions: number, dModel: number, base = PE_BASE): Matrix {
  return Array.from({ length: positions }, (_, pos) => sinusoidalPE(pos, dModel, base));
}

/** The book prints the table cut (not rounded) to 2 decimals, e.g. sin 2 = 0.909… shows as 0.90. */
export function truncate2(value: number): number {
  return Math.trunc(value * 100) / 100;
}

/**
 * Four terms of an additive-PE dot product with W_Q = W_K = I (chapter 25.2.2):
 * (e_m + p_m)·(e_n + p_n) = e_m·e_n + e_m·p_n + p_m·e_n + p_m·p_n.
 */
export function crossTerms(em: number[], pm: number[], en: number[], pn: number[]) {
  const terms = {
    contentContent: dot(em, en),
    contentPosition: dot(em, pn),
    positionContent: dot(pm, en),
    positionPosition: dot(pm, pn),
  };
  return {
    ...terms,
    total: dot(addVectors(em, pm), addVectors(en, pn)),
  };
}

// ---------------------------------------------------------------- RoPE (第 25 章)

/** θ_i = base^(−2i/d) for pair i = 0 … d/2−1 (the paper writes i = 1 … d/2 with 2(i−1)). */
export function ropeTheta(i: number, d: number, base = PE_BASE): number {
  return sinusoidalFrequency(i, d, base);
}

/** Rotate (x, y) counter-clockwise by `angle`: [x cos − y sin, x sin + y cos]. */
export function rotatePair(pair: [number, number], angle: number): [number, number] {
  const [x, y] = pair;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

/** R_m v: every adjacent pair (2i, 2i+1) is rotated by m·θ_i. Only Q and K receive this. */
export function applyRope(vector: number[], position: number, base = PE_BASE): number[] {
  checkVector(vector, T('RoPE 输入'));
  if (vector.length % 2) throw new Error('RoPE 需要偶数维向量。');
  if (!Number.isFinite(position)) throw new Error('位置必须是有限数。');
  const out: number[] = [];
  for (let i = 0; i < vector.length / 2; i++) {
    const [x, y] = rotatePair(
      [vector[2 * i], vector[2 * i + 1]],
      position * ropeTheta(i, vector.length, base),
    );
    out.push(x, y);
  }
  return out;
}

/** Per-pair contributions of (R_m q)·(R_n k); their sum is the RoPE attention logit (before ÷√d). */
export function ropeScoreParts(q: number[], k: number[], m: number, n: number, base = PE_BASE) {
  const qr = applyRope(q, m, base);
  const kr = applyRope(k, n, base);
  return Array.from(
    { length: q.length / 2 },
    (_, i) => qr[2 * i] * kr[2 * i] + qr[2 * i + 1] * kr[2 * i + 1],
  );
}

export function ropeScore(q: number[], k: number[], m: number, n: number, base = PE_BASE): number {
  return ropeScoreParts(q, k, m, n, base).reduce((a, b) => a + b, 0);
}

// ---------------------------------------------------------------- ALiBi (第 25 章)

/** Head slopes 2^(−8h/n), h = 1 … n, as in the ALiBi paper for n a power of two. */
export function alibiSlopes(numHeads: number): number[] {
  if (!Number.isInteger(numHeads) || numHeads < 1 || (numHeads & (numHeads - 1)) !== 0)
    throw new Error('本实现只处理 2 的幂个头。');
  return Array.from({ length: numHeads }, (_, h) => Math.pow(2, (-8 * (h + 1)) / numHeads));
}

/** bias[i][j] = −m·(i − j) for visible keys j ≤ i; future keys are masked separately. */
export function alibiBias(tokens: number, slope: number): Matrix {
  return Array.from({ length: tokens }, (_, i) =>
    Array.from({ length: tokens }, (_, j) => (j < i ? -slope * (i - j) : 0)),
  );
}

/** Content scores (already ÷√d) + ALiBi bias, then the causal mask and a row softmax. */
export function alibiAttention(scores: Matrix, slope: number) {
  const tokens = scores.length;
  if (scores.some((row) => row.length !== tokens)) throw new Error('分数矩阵必须是方阵。');
  const bias = alibiBias(tokens, slope);
  const biased = scores.map((row, i) => row.map((s, j) => s + bias[i][j]));
  const masked = biased.map((row, i) => row.map((s, j) => (j > i ? -Infinity : s)));
  return { bias, biased, masked, probabilities: masked.map(softmax) };
}

// ---------------------------------------------------------------- 位置盲（第 5.1 节）

/**
 * Self-attention with W_Q = W_K = W_V = I, so the only thing that can carry order is what is
 * added to the embeddings. Returns the output row of every token.
 */
export function identityAttention(x: Matrix, causal: boolean) {
  return attention(x, x, x, causal, true);
}

// Fixed teaching values (not from a trained model).
/** First 4 embedding values of “L” printed in chapter 5 (the full vector has d_model entries). */
export const BOOK_L_EMBEDDING = [0.62, -0.51, 0.09, 0.85];
/** d_model = 4 toy embeddings for the “我爱你 / 你爱我” comparison. */
export const TOY_EMBEDDINGS: Record<'我' | '爱' | '你', number[]> = {
  我: [1, 0, 1, 0],
  爱: [0, 1, 0.5, 0.5],
  你: [0.5, 0, 0, 1],
};
/** d = 8 content vectors for RoPE: 4 pairs whose θ are exactly 1, 0.1, 0.01, 0.001. */
export const ROPE_Q = [1, 0, 0, 1, 1, 1, 1, 0];
export const ROPE_K = [1, 1, 1, 0, 0, 1, 1, 0];
