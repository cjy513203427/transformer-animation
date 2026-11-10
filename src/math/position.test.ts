import { describe, expect, it } from 'vitest';
import {
  addVectors,
  alibiAttention,
  alibiBias,
  alibiSlopes,
  applyRope,
  crossTerms,
  dot,
  identityAttention,
  norm,
  ROPE_K,
  ROPE_Q,
  ropeScore,
  ropeScoreParts,
  ropeTheta,
  rotatePair,
  sinusoidalFrequency,
  sinusoidalPE,
  sinusoidalTable,
  TOY_EMBEDDINGS,
  truncate2,
} from './position';

const TOLERANCE = 1e-12;

describe('正弦位置编码（第 5 章）', () => {
  it('d_model = 64 的前 4 维，截断到两位小数后与书中表格逐格一致', () => {
    const book = [
      [0, 1, 0, 1],
      [0.84, 0.54, 0.68, 0.73],
      [0.9, -0.41, 0.99, 0.07],
      [0.14, -0.98, 0.77, -0.62],
      [-0.75, -0.65, 0.14, -0.98],
      [-0.95, 0.28, -0.57, -0.82],
    ];
    const table = sinusoidalTable(6, 64);
    table.forEach((row, pos) =>
      row.slice(0, 4).forEach((value, j) => expect(truncate2(value)).toBe(book[pos][j])),
    );
    // The book truncates: sin(2) = 0.909… would round to 0.91 but is printed 0.90.
    expect(Math.round(table[2][0] * 100) / 100).toBe(0.91);
  });
  it('每个 sin/cos 维度对落在单位圆上，频率随维度序号变低', () => {
    for (const dModel of [4, 8, 64, 512]) {
      for (let pos = 0; pos < 40; pos++) {
        const pe = sinusoidalPE(pos, dModel);
        for (let i = 0; i < dModel / 2; i++)
          expect(Math.abs(pe[2 * i] ** 2 + pe[2 * i + 1] ** 2 - 1)).toBeLessThan(TOLERANCE);
      }
      for (let i = 1; i < dModel / 2; i++)
        expect(sinusoidalFrequency(i, dModel)).toBeLessThan(sinusoidalFrequency(i - 1, dModel));
    }
    expect(sinusoidalFrequency(0, 64)).toBe(1);
    expect(() => sinusoidalFrequency(0, 5)).toThrow();
    expect(() => sinusoidalPE(-1, 4)).toThrow();
  });
  it('64 个位置的编码互不相同；PE(m)·PE(n) 只取决于 m − n', () => {
    const table = sinusoidalTable(64, 64);
    for (let a = 0; a < 64; a++)
      for (let b = a + 1; b < 64; b++)
        expect(table[a].some((x, j) => Math.abs(x - table[b][j]) > 1e-6)).toBe(true);
    for (const [m, n, shift] of [
      [3, 7, 5],
      [10, 2, 9],
      [0, 0, 30],
    ])
      expect(
        Math.abs(dot(table[m], table[n]) - dot(table[m + shift], table[n + shift])),
      ).toBeLessThan(1e-9);
  });
  it('加法编码的点积可拆成内容·内容、内容·位置、位置·内容、位置·位置四项', () => {
    const [em, en] = [TOY_EMBEDDINGS.我, TOY_EMBEDDINGS.你];
    const [pm, pn] = [sinusoidalPE(0, 4), sinusoidalPE(2, 4)];
    const terms = crossTerms(em, pm, en, pn);
    const sum =
      terms.contentContent + terms.contentPosition + terms.positionContent + terms.positionPosition;
    expect(Math.abs(sum - terms.total)).toBeLessThan(TOLERANCE);
    expect(terms.total).toBe(dot(addVectors(em, pm), addVectors(en, pn)));
  });
});

describe('位置盲：没有位置信息时 Attention 分不清顺序', () => {
  const E = TOY_EMBEDDINGS;
  it('双向读取：“我爱你”与“你爱我”中“我”的输出完全相同；加入位置编码后不同', () => {
    const a = identityAttention([E.我, E.爱, E.你], false);
    const b = identityAttention([E.你, E.爱, E.我], false);
    a.output[0].forEach((x, d) => expect(Math.abs(x - b.output[2][d])).toBeLessThan(TOLERANCE));
    const withPE = (tokens: number[][]) =>
      identityAttention(
        tokens.map((e, pos) => addVectors(e, sinusoidalPE(pos, 4))),
        false,
      );
    const pa = withPE([E.我, E.爱, E.你]);
    const pb = withPE([E.你, E.爱, E.我]);
    expect(pa.output[0].some((x, d) => Math.abs(x - pb.output[2][d]) > 1e-6)).toBe(true);
  });
  it('因果 Mask 只限制可见范围：“我爱你”与“爱我你”最后的“你”输出相同', () => {
    const a = identityAttention([E.我, E.爱, E.你], true);
    const b = identityAttention([E.爱, E.我, E.你], true);
    a.output[2].forEach((x, d) => expect(Math.abs(x - b.output[2][d])).toBeLessThan(TOLERANCE));
  });
});

describe('RoPE（第 25 章）', () => {
  it('d = 8 时 θ 恰为 1、0.1、0.01、0.001', () => {
    [1, 0.1, 0.01, 0.001].forEach((theta, i) =>
      expect(Math.abs(ropeTheta(i, 8) - theta)).toBeLessThan(TOLERANCE),
    );
  });
  it('旋转保持每个二维对的长度与整体范数；位置 0 不旋转', () => {
    for (let m = 0; m < 50; m += 7) {
      const rotated = applyRope(ROPE_Q, m);
      for (let i = 0; i < 4; i++)
        expect(
          Math.abs(
            Math.hypot(rotated[2 * i], rotated[2 * i + 1]) -
              Math.hypot(ROPE_Q[2 * i], ROPE_Q[2 * i + 1]),
          ),
        ).toBeLessThan(TOLERANCE);
      expect(Math.abs(norm(rotated) - norm(ROPE_Q))).toBeLessThan(TOLERANCE);
    }
    expect(applyRope(ROPE_Q, 0)).toEqual(ROPE_Q);
    expect(rotatePair([1, 0], Math.PI / 2).map((x) => Math.round(x * 1e12) / 1e12)).toEqual([0, 1]);
  });
  it('固定内容，同时平移 m 与 n，分数不变：(R_m q)·(R_n k) = q·(R_{n−m} k)', () => {
    for (const [m, n] of [
      [0, 3],
      [2, 9],
      [7, 1],
      [15, 15],
    ]) {
      const base = ropeScore(ROPE_Q, ROPE_K, m, n);
      for (const shift of [1, 5, 40, 1000])
        expect(Math.abs(ropeScore(ROPE_Q, ROPE_K, m + shift, n + shift) - base)).toBeLessThan(1e-9);
      expect(Math.abs(dot(ROPE_Q, applyRope(ROPE_K, n - m)) - base)).toBeLessThan(1e-9);
      const parts = ropeScoreParts(ROPE_Q, ROPE_K, m, n);
      expect(Math.abs(parts.reduce((a, b) => a + b, 0) - base)).toBeLessThan(TOLERANCE);
    }
  });
  it('单个分数随距离并不单调下降', () => {
    const scores = Array.from({ length: 33 }, (_, distance) =>
      ropeScore(ROPE_Q, ROPE_K, 0, distance),
    );
    const rises = scores.slice(1).filter((s, d) => s > scores[d] + 1e-9).length;
    const falls = scores.slice(1).filter((s, d) => s < scores[d] - 1e-9).length;
    expect(rises).toBeGreaterThan(0);
    expect(falls).toBeGreaterThan(0);
  });
  it('拒绝奇数维向量', () => {
    expect(() => applyRope([1, 2, 3], 1)).toThrow();
  });
});

describe('ALiBi（第 25 章）', () => {
  it('8 个头的斜率为 1/2 … 1/256；4 个头为 1/4 … 1/256', () => {
    expect(alibiSlopes(8)).toEqual([1, 2, 3, 4, 5, 6, 7, 8].map((h) => 2 ** -h));
    expect(alibiSlopes(4)).toEqual([2, 4, 6, 8].map((h) => 2 ** -h));
    expect(() => alibiSlopes(6)).toThrow();
  });
  it('偏置在对角线为 0，向左每远一格减少 m', () => {
    const bias = alibiBias(5, 0.5);
    expect(bias[4]).toEqual([-2, -1.5, -1, -0.5, 0]);
    bias.forEach((row, i) => expect(row[i]).toBe(0));
  });
  it('内容分数相同时，权重随距离按 e^(−m) 等比衰减；未来位置为 0', () => {
    const slope = 0.25;
    const result = alibiAttention(
      Array.from({ length: 6 }, () => Array(6).fill(0)),
      slope,
    );
    const row = result.probabilities[5];
    for (let j = 1; j <= 5; j++)
      expect(Math.abs(row[j - 1] / row[j] - Math.exp(-slope))).toBeLessThan(TOLERANCE);
    expect(result.probabilities[2].slice(3)).toEqual([0, 0, 0]);
  });
  it('同一公式可直接计算更长序列，行和为 1、没有 NaN', () => {
    const tokens = 200;
    const result = alibiAttention(
      Array.from({ length: tokens }, () => Array(tokens).fill(0)),
      2 ** -8,
    );
    result.probabilities.forEach((row) => {
      expect(row.every(Number.isFinite)).toBe(true);
      expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9);
    });
  });
});
