import { describe, expect, it } from 'vitest';
import {
  attention,
  computeExperiment,
  explainRow,
  matmul,
  seededInput,
  softmax,
  transpose,
} from './attention';

describe('数值基础', () => {
  it('矩阵乘法保留行与列的对应', () => {
    expect(
      matmul(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      ),
    ).toEqual([
      [22, 28],
      [49, 64],
    ]);
    expect(
      transpose([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow();
    expect(() => transpose([[1], [2, 3]])).toThrow();
  });
  it('Softmax 在极端分数下稳定，且对整体平移不变', () => {
    const expected = softmax([1, 2, 3]);
    const actual = softmax([10001, 10002, 10003]);
    actual.forEach((p, j) => expect(p).toBeCloseTo(expected[j], 12));
    expect(actual.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(softmax([10000, -10000])).toEqual([1, 0]);
  });
  it('全屏蔽行约定为零，无 NaN；拒绝非法值', () => {
    expect(softmax([-Infinity, -Infinity])).toEqual([0, 0]);
    expect(softmax([0, -Infinity])).toEqual([1, 0]);
    expect(() => softmax([NaN])).toThrow();
    expect(() => softmax([Infinity])).toThrow();
  });
});

describe('Attention 数学性质（JavaScript float64，容差 1e-12）', () => {
  it('导读贡献逐维合成真实输出，屏蔽位置贡献为零', () => {
    const result = computeExperiment(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      0,
      0,
      true,
      true,
    );
    const row = explainRow(result, 1);
    expect(row.keys[2].maskedScore).toBe(-Infinity);
    expect(row.keys[2].contribution).toEqual([0, 0]);
    for (let d = 0; d < 2; d++)
      expect(row.keys.reduce((sum, key) => sum + key.contribution[d], 0)).toBeCloseTo(
        result.output[1][d],
        12,
      );
    expect(row.denominator).toBeGreaterThan(0);
  });
  it.each([false, true])('与独立逐 Query/Key/维度参考实现一致，causal=%s', (causal) => {
    // Independent scalar reference: no production matmul/transpose/softmax calls.
    for (let seed = 0; seed < 20; seed++) {
      const q = seededInput(seed);
      const k = seededInput(seed + 50);
      const v = seededInput(seed + 100);
      const result = attention(q, k, v, causal);
      for (let i = 0; i < 3; i++) {
        const weights: number[] = [];
        for (let j = 0; j < 3; j++) {
          let dot = 0;
          for (let d = 0; d < 2; d++) dot += q[i][d] * k[j][d];
          weights.push(causal && j > i ? 0 : Math.exp(dot / Math.sqrt(2)));
        }
        const total = weights.reduce((a, b) => a + b, 0);
        for (let d = 0; d < 2; d++) {
          let output = 0;
          for (let j = 0; j < 3; j++) output += (weights[j] / total) * v[j][d];
          expect(Math.abs(result.output[i][d] - output)).toBeLessThan(1e-12);
        }
        expect(result.probabilities[i].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
        if (causal) for (let j = i + 1; j < 3; j++) expect(result.probabilities[i][j]).toBe(0);
      }
    }
  });
  it('只干预 V，P 不变且输出差等于对应权重乘以偏移', () => {
    const x = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const baseline = computeExperiment(x, 0, 0, false, true);
    const changed = computeExperiment(x, 0, 3, false, true);
    expect(changed.probabilities).toEqual(baseline.probabilities);
    for (let i = 0; i < 3; i++) {
      expect(changed.output[i][0] - baseline.output[i][0]).toBeCloseTo(
        3 * baseline.probabilities[i][1],
        12,
      );
      expect(changed.output[i][1]).toBe(baseline.output[i][1]);
    }
  });
  it('干预被屏蔽的 V 不会影响当前输出', () => {
    const x = seededInput(42);
    expect(computeExperiment(x, 0, 4, true, true).output[0]).toEqual(
      computeExperiment(x, 0, 0, true, true).output[0],
    );
  });
  it('只干预 K 的一行，只改变相应分数列但可改变整行归一化', () => {
    const x = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const a = computeExperiment(x, 0, 0, false, true);
    const b = computeExperiment(x, 2, 0, false, true);
    expect(b.scores[2][0]).toBe(a.scores[2][0]);
    expect(b.scores[2][2]).toBe(a.scores[2][2]);
    expect(b.scores[2][1]).not.toBe(a.scores[2][1]);
    b.probabilities[2].forEach((p, j) => expect(p).not.toBe(a.probabilities[2][j]));
  });
  it('相同分数产生均匀分布；首行因果输出等于自身 V', () => {
    const q = [
      [0, 0],
      [0, 0],
    ];
    const k = [
      [1, 2],
      [3, 4],
    ];
    const v = [
      [2, 4],
      [4, 8],
    ];
    expect(attention(q, k, v).output).toEqual([
      [3, 6],
      [3, 6],
    ]);
    expect(attention(q, k, v, true).output[0]).toEqual(v[0]);
  });
  it('随机示例可复现，且数值在规定范围', () => {
    expect(seededInput(42)).toEqual(seededInput(42));
    expect(seededInput(42)).not.toEqual(seededInput(43));
    expect(
      seededInput(0)
        .flat()
        .every((x) => Math.abs(x) <= 2 && Number.isInteger(x * 2)),
    ).toBe(true);
  });
});
