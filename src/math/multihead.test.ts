import { describe, expect, it } from 'vitest';
import { attention, matmul, type Matrix } from './attention';
import {
  attentionWeightCount,
  columnBlock,
  HEAD_OPTIONS,
  mergeHeads,
  MH_WEIGHTS,
  MH_X,
  multiHeadAttention,
  rowBlock,
  scoreMultiplyAdds,
  splitHeads,
  type MultiHeadWeights,
} from './multihead';

const TOLERANCE = 1e-12;

function lcg(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return ((value % 9) - 4) / 2;
  };
}
function randomMatrix(rows: number, columns: number, next: () => number): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: columns }, next));
}
function randomWeights(seed: number): MultiHeadWeights {
  const next = lcg(seed);
  return {
    wq: randomMatrix(4, 4, next),
    wk: randomMatrix(4, 4, next),
    wv: randomMatrix(4, 4, next),
    wo: randomMatrix(4, 4, next),
  };
}
function expectClose(actual: Matrix, expected: Matrix) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, i) =>
    row.forEach((value, j) => expect(Math.abs(value - expected[i][j])).toBeLessThan(TOLERANCE)),
  );
}

describe('切分多头与合并（view + transpose）', () => {
  it.each(HEAD_OPTIONS)('num_heads=%s：往返不丢数据，token 行顺序不变', (numHeads) => {
    const m = randomMatrix(3, 4, lcg(numHeads));
    const heads = splitHeads(m, numHeads);
    const dKey = 4 / numHeads;
    expect(heads).toHaveLength(numHeads);
    heads.forEach((head, h) => {
      expect(head).toHaveLength(3);
      head.forEach((row, t) => {
        expect(row).toHaveLength(dKey);
        // Element (t, j) of head h is element (t, h·d_key + j) of the full matrix.
        row.forEach((value, j) => expect(value).toBe(m[t][h * dKey + j]));
      });
    });
    expect(mergeHeads(heads)).toEqual(m);
  });
  it('拒绝不能整除的头数与不一致的头形状', () => {
    expect(() => splitHeads(MH_X, 3)).toThrow();
    expect(() => splitHeads(MH_X, 0)).toThrow();
    expect(() => mergeHeads([[[1, 2]], [[1]]])).toThrow();
    expect(() => mergeHeads([])).toThrow();
  });
});

describe('Multi-Head Attention 数学性质（float64，容差 1e-12）', () => {
  it.each([false, true])('“方式一”独立参考实现与“一次大投影再切分”一致，causal=%s', (causal) => {
    for (let seed = 0; seed < 15; seed++) {
      const x = randomMatrix(3, 4, lcg(seed + 300));
      const weights = randomWeights(seed);
      for (const numHeads of HEAD_OPTIONS) {
        const dKey = 4 / numHeads;
        const result = multiHeadAttention(x, weights, numHeads, causal);
        for (let h = 0; h < numHeads; h++) {
          // Scalar reference with per-head weights Wq_h = W_Q[:, h·d_key:(h+1)·d_key].
          const project = (w: Matrix) =>
            x.map((row) =>
              Array.from({ length: dKey }, (_, j) => {
                let sum = 0;
                for (let i = 0; i < 4; i++) sum += row[i] * w[i][h * dKey + j];
                return sum;
              }),
            );
          const [qh, kh, vh] = [project(weights.wq), project(weights.wk), project(weights.wv)];
          for (let i = 0; i < 3; i++) {
            const raw: number[] = [];
            for (let j = 0; j < 3; j++) {
              let dot = 0;
              for (let d = 0; d < dKey; d++) dot += qh[i][d] * kh[j][d];
              raw.push(causal && j > i ? -Infinity : dot / Math.sqrt(dKey));
            }
            const max = Math.max(...raw);
            const exps = raw.map((s) => (s === -Infinity ? 0 : Math.exp(s - max)));
            const total = exps.reduce((a, b) => a + b, 0);
            for (let d = 0; d < dKey; d++) {
              let out = 0;
              for (let j = 0; j < 3; j++) out += (exps[j] / total) * vh[j][d];
              expect(Math.abs(result.heads[h].output[i][d] - out)).toBeLessThan(TOLERANCE);
              expect(Math.abs(result.concat[i][h * dKey + d] - out)).toBeLessThan(TOLERANCE);
            }
            const rowSum = result.heads[h].probabilities[i].reduce((a, b) => a + b, 0);
            expect(Math.abs(rowSum - 1)).toBeLessThan(TOLERANCE);
            if (causal)
              for (let j = i + 1; j < 3; j++) expect(result.heads[h].probabilities[i][j]).toBe(0);
          }
          expectClose(
            result.heads[h].q,
            matmul(x, columnBlock(weights.wq, h * dKey, (h + 1) * dKey)),
          );
        }
      }
    }
  });
  it('Concat 后乘 Wo，等于各头输出乘 Wo 对应行块再相加', () => {
    for (const numHeads of HEAD_OPTIONS) {
      const result = multiHeadAttention(MH_X, MH_WEIGHTS, numHeads, true);
      const dKey = 4 / numHeads;
      const summed = result.heads
        .map((head, h) => matmul(head.output, rowBlock(MH_WEIGHTS.wo, h * dKey, (h + 1) * dKey)))
        .reduce((total, part) => total.map((row, i) => row.map((x, j) => x + part[i][j])));
      expectClose(result.output, summed);
    }
  });
  it('只有一个头时，就是 d_key = d_model 的单头 Attention 再乘 Wo', () => {
    const result = multiHeadAttention(MH_X, MH_WEIGHTS, 1, true);
    const single = attention(
      matmul(MH_X, MH_WEIGHTS.wq),
      matmul(MH_X, MH_WEIGHTS.wk),
      matmul(MH_X, MH_WEIGHTS.wv),
      true,
    );
    expectClose(result.concat, single.output);
    expectClose(result.output, matmul(single.output, MH_WEIGHTS.wo));
  });
  it('干预最后一个头的 K：其他头的 P 与 A 列不变，经 Wo 后输出各列都变化', () => {
    const intervention = { token: 0, column: 3, delta: -2 };
    const base = multiHeadAttention(MH_X, MH_WEIGHTS, 2, true);
    const changed = multiHeadAttention(MH_X, MH_WEIGHTS, 2, true, intervention);
    expect(changed.heads[0].probabilities).toEqual(base.heads[0].probabilities);
    // Row 0 can only see itself under the causal mask, so its weights stay [1, 0, 0].
    expect(changed.heads[1].probabilities[0]).toEqual(base.heads[1].probabilities[0]);
    for (const row of [1, 2]) {
      expect(changed.heads[1].probabilities[row]).not.toEqual(base.heads[1].probabilities[row]);
      expect(changed.concat[row].slice(0, 2)).toEqual(base.concat[row].slice(0, 2));
      changed.concat[row]
        .slice(2)
        .forEach((value, j) => expect(value).not.toBe(base.concat[row][2 + j]));
      changed.output[row].forEach((value, j) => expect(value).not.toBe(base.output[row][j]));
    }
    // The weights are untouched: only the activation K was edited.
    expect(changed.q).toEqual(base.q);
    expect(changed.v).toEqual(base.v);
  });
  it('教学示例的两个头在同一 Query 上关注不同位置', () => {
    const result = multiHeadAttention(MH_X, MH_WEIGHTS, 2, true);
    const argmax = (row: number[]) => row.indexOf(Math.max(...row));
    expect(argmax(result.heads[0].probabilities[2])).not.toBe(
      argmax(result.heads[1].probabilities[2]),
    );
  });
  it('打分乘加次数与头数无关；参数量为 4 × d_model²', () => {
    for (const numHeads of HEAD_OPTIONS) expect(scoreMultiplyAdds(3, 4, numHeads)).toBe(36);
    expect(scoreMultiplyAdds(16, 512, 4)).toBe(16 * 16 * 512);
    expect(attentionWeightCount(4)).toBe(64);
    expect(attentionWeightCount(768)).toBe(2_359_296);
  });
});
