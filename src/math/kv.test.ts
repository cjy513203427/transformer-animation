import { describe, expect, it } from 'vitest';
import {
  coupledRopeRows,
  D_HEAD,
  D_LATENT,
  D_ROPE,
  DEEPSEEK_V2,
  formatBytes,
  groupedAttention,
  groupOf,
  KV_OPTIONS,
  kvCacheBytes,
  mlaAbsorbed,
  mlaCache,
  mlaCacheElements,
  mlaExpanded,
  MHA_WEIGHTS,
  MLA_WEIGHTS,
  poolHeads,
  projectionParams,
  Q_HEADS,
  repeatedKVReference,
  rope2,
  X,
} from './kv';

const TOL = 1e-12;
const maxDiff = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((row, i) => row.map((value, j) => Math.abs(value - b[i][j]))));

describe('KV 缓存字节账本', () => {
  it('与原书第 23 章的例子一致：32 层、d_h = 128、1024 token、2 字节', () => {
    const base = { batch: 1, layers: 32, tokens: 1024, dK: 128, dV: 128, bytesPerElement: 2 };
    expect(kvCacheBytes({ ...base, kvHeads: 32 })).toBe(536_870_912);
    expect(kvCacheBytes({ ...base, kvHeads: 8 })).toBe(134_217_728);
    expect(kvCacheBytes({ ...base, kvHeads: 1 })).toBe(16_777_216);
    expect(formatBytes(536_870_912)).toBe('512 MiB');
    expect(formatBytes(16_777_216)).toBe('16 MiB');
  });

  it('与原书第 22 章的 Llama 2 7B 手算一致：每 token 512 KiB，4096 token 2 GiB', () => {
    const llama = { batch: 1, layers: 32, kvHeads: 32, dK: 128, dV: 128, bytesPerElement: 2 };
    expect(kvCacheBytes({ ...llama, tokens: 1 })).toBe(524_288);
    expect(formatBytes(kvCacheBytes({ ...llama, tokens: 4096 }))).toBe('2 GiB');
  });

  it('DeepSeek-V2 的 MLA 每 token 每层缓存 d_c + d_h^R = 576 个元素，相当于 2.25 组的 GQA', () => {
    const { layers, latent, ropeDim, dHead } = DEEPSEEK_V2;
    const perLayer = mlaCacheElements({
      batch: 1,
      layers: 1,
      tokens: 1,
      latent,
      ropeDim,
      bytesPerElement: 2,
    });
    expect(perLayer).toBe(576);
    expect(perLayer).toBe((9 / 2) * dHead);
    expect(perLayer / (2 * dHead)).toBe(2.25);
    // Counting only the latent would understate the cache.
    expect(perLayer).toBeGreaterThan(latent);
    expect(
      mlaCacheElements({ batch: 1, layers, tokens: 1, latent, ropeDim, bytesPerElement: 2 }),
    ).toBe(34_560);
  });

  it('只有 W_K、W_V 随 KV 头数变化：H_Q = 32 时 MHA 4D²、GQA-8 2.5D²、MQA 2.0625D²', () => {
    const D = 4096;
    const p = (kvHeads: number) =>
      projectionParams({ dModel: D, qHeads: 32, kvHeads, dHead: 128 }).total;
    expect(p(32)).toBe(4 * D * D);
    expect(p(8)).toBe(2.5 * D * D);
    expect(p(1)).toBe(2.0625 * D * D);
    expect(() => projectionParams({ dModel: D, qHeads: 32, kvHeads: 3, dHead: 128 })).toThrow();
  });

  it('非正整数参数被拒绝', () => {
    expect(() =>
      kvCacheBytes({
        batch: 0,
        layers: 1,
        tokens: 1,
        kvHeads: 1,
        dK: 1,
        dV: 1,
        bytesPerElement: 2,
      }),
    ).toThrow();
    expect(() =>
      kvCacheBytes({
        batch: 1,
        layers: 1.5,
        tokens: 1,
        kvHeads: 1,
        dK: 1,
        dV: 1,
        bytesPerElement: 2,
      }),
    ).toThrow();
  });
});

describe('MHA / GQA / MQA', () => {
  it('分组映射：相邻的 Q 头共享同一组 K/V', () => {
    const map = (g: number) => Array.from({ length: Q_HEADS }, (_, h) => groupOf(h, Q_HEADS, g));
    expect(map(4)).toEqual([0, 1, 2, 3]);
    expect(map(2)).toEqual([0, 0, 1, 1]);
    expect(map(1)).toEqual([0, 0, 0, 0]);
    expect(() => groupOf(0, 4, 3)).toThrow();
  });

  it('组内取平均：G = H 时不变，G = 1 时每列是所有头对应列的均值', () => {
    expect(poolHeads(MHA_WEIGHTS.wk, Q_HEADS, Q_HEADS, D_HEAD)).toEqual(MHA_WEIGHTS.wk);
    const pooled = poolHeads(MHA_WEIGHTS.wk, Q_HEADS, 1, D_HEAD);
    pooled.forEach((row, r) =>
      row.forEach((value, d) => {
        const mean =
          [0, 1, 2, 3].reduce((sum, h) => sum + MHA_WEIGHTS.wk[r][h * D_HEAD + d], 0) / 4;
        expect(value).toBeCloseTo(mean, 12);
      }),
    );
  });

  it.each(KV_OPTIONS)('G = %s：按组查表与“显式复制 K/V 头”的参考计算一致（含因果 Mask）', (g) => {
    for (const causal of [true, false]) {
      const grouped = groupedAttention(X, MHA_WEIGHTS, g, causal);
      expect(maxDiff(grouped.output, repeatedKVReference(X, MHA_WEIGHTS, g, causal))).toBeLessThan(
        TOL,
      );
    }
  });

  it('KV 头数只改变缓存和 K/V 投影，不改变打分次数和 Q 头数', () => {
    const results = KV_OPTIONS.map((g) => groupedAttention(X, MHA_WEIGHTS, g));
    expect(results.map((r) => r.cachePerToken)).toEqual([16, 8, 4]);
    expect(new Set(results.map((r) => r.scorePairs))).toEqual(new Set([Q_HEADS * 10]));
    results.forEach((r) => {
      expect(r.q).toHaveLength(Q_HEADS);
      expect(r.probs).toHaveLength(Q_HEADS);
      r.probs.forEach((p) =>
        p.forEach((row, i) => {
          expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(TOL);
          row.forEach((value, j) => j > i && expect(value).toBe(0));
        }),
      );
    });
  });

  it('同组的 Q 头读同一份 K，但各自的权重可以不同', () => {
    const r = groupedAttention(X, MHA_WEIGHTS, 2);
    expect(maxDiff(r.probs[0], r.probs[1])).toBeGreaterThan(1e-6);
  });
});

describe('MLA', () => {
  it('缓存每个 token 只存 latent 与共享 RoPE Key', () => {
    const cache = mlaCache(X, MLA_WEIGHTS);
    expect(cache.latent[0]).toHaveLength(D_LATENT);
    expect(cache.ropeKey[0]).toHaveLength(D_ROPE);
  });

  it('RoPE 只旋转、不改变长度', () => {
    for (let pos = 0; pos < 6; pos++) {
      const v = rope2([0.6, -1.3], pos);
      expect(Math.hypot(v[0], v[1])).toBeCloseTo(Math.hypot(0.6, -1.3), 12);
    }
  });

  it('吸收路径（只读缓存）与展开路径的分数、权重和输出一致，含 RoPE 分支', () => {
    const expanded = mlaExpanded(X, MLA_WEIGHTS);
    const absorbed = mlaAbsorbed(X, MLA_WEIGHTS, expanded.cache);
    for (let i = 0; i < Q_HEADS; i++) {
      const finite = (m: number[][]) => m.map((row) => row.map((v) => (v === -Infinity ? 0 : v)));
      expect(maxDiff(finite(absorbed.scores[i]), finite(expanded.scores[i]))).toBeLessThan(TOL);
      expect(maxDiff(absorbed.probs[i], expanded.probs[i])).toBeLessThan(TOL);
    }
    expect(maxDiff(absorbed.output, expanded.output)).toBeLessThan(TOL);
  });

  it('RoPE 分支确实参与了打分：去掉它，分数会变', () => {
    const expanded = mlaExpanded(X, MLA_WEIGHTS);
    const noRope = mlaExpanded(X, {
      ...MLA_WEIGHTS,
      wkr: MLA_WEIGHTS.wkr.map((r) => r.map(() => 0)),
    });
    expect(maxDiff(noRope.probs[0], expanded.probs[0])).toBeGreaterThan(1e-6);
  });

  it('不解耦时（RoPE 直接加在还原出的 K 上），同一个吸收矩阵只在 j = t 处正确', () => {
    for (let head = 0; head < Q_HEADS; head++)
      for (let t = 0; t < X.length; t++) {
        const rows = coupledRopeRows(X, MLA_WEIGHTS, head, t);
        expect(rows).toHaveLength(t + 1);
        expect(rows[t].error).toBeLessThan(TOL);
      }
    const last = coupledRopeRows(X, MLA_WEIGHTS, 0, 3);
    expect(last.slice(0, 3).some((row) => row.error > 1e-3)).toBe(true);
  });
});
