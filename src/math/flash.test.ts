import { describe, expect, it } from 'vitest';
import {
  computeFlash,
  flashLedger,
  makeFlashInputs,
  onlineAttention,
  traceOnlineRow,
  type FlashOptions,
} from './flash';
import type { Matrix } from './attention';

const TOL = 1e-12;

// Deliberately independent, scalar dense reference: no attention, matmul, or softmax helpers.
function reference(q: Matrix, k: Matrix, v: Matrix, causal: boolean): Matrix {
  return q.map((query, i) => {
    const scores = k.map((key, j) => {
      if (causal && j > i) return -Infinity;
      let dot = 0;
      for (let d = 0; d < query.length; d++) dot += query[d] * key[d];
      return dot / Math.sqrt(query.length);
    });
    const maximum = Math.max(...scores);
    const weights = scores.map((s) => Math.exp(s - maximum));
    const denominator = weights.reduce((a, b) => a + b, 0);
    return v[0].map((_, d) => {
      let total = 0;
      for (let j = 0; j < v.length; j++) total += (weights[j] * v[j][d]) / denominator;
      return total;
    });
  });
}

function expectClose(actual: Matrix, expected: Matrix) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((row, i) =>
    row.forEach((value, d) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[i][d])).toBeLessThan(TOL);
    }),
  );
}

describe('分块 Attention 与完整参考计算', () => {
  it('不同分块、末块不足长度、因果边界与多种输入均等价', () => {
    for (const seed of [1, 42, 9999])
      for (const preset of ['rising', 'mixed'] as const)
        for (const causal of [true, false])
          for (const blockSize of [1, 2, 3, 4] as const)
            for (const keyDelta of [-2, 0, 2]) {
              const options: FlashOptions = {
                seed,
                config: { causal, blockSize },
                input: { preset, query: 3, keyDelta, block: 0 },
              };
              const result = computeFlash(options);
              expectClose(result.output, reference(result.q, result.k, result.v, causal));
              expect(result.maxError).toBeLessThan(TOL);
              result.rows.forEach((blocks, query) => {
                expect(blocks).toHaveLength(Math.ceil(4 / blockSize));
                expect(blocks.flatMap((block) => block.keys)).toEqual([0, 1, 2, 3]);
                if (causal)
                  blocks.forEach((block) =>
                    block.keys.forEach((key, j) => {
                      if (key > query) expect(block.exponentials[j]).toBe(0);
                    }),
                  );
              });
            }
  });

  it('支持非方阵与不同的 Key/Value 维度', () => {
    const q = [
      [1, 2, -1],
      [2, 0, 1],
    ];
    const k = [
      [0, 2, 3],
      [1, 0, -1],
      [-2, 1, 0],
      [2, 3, 4],
      [0, 1, 2],
    ];
    const v = [[0], [1], [2], [-1], [3]];
    for (const block of [1, 2, 3, 7])
      expectClose(onlineAttention(q, k, v, block).output, reference(q, k, v, false));
  });

  it('大分数仍然稳定；不会直接计算 exp(1000)', () => {
    const q = [[1], [2], [3]];
    const k = [[1000], [999], [1001]];
    const v = [
      [-2, 1],
      [0, 1],
      [4, 1],
    ];
    for (const block of [1, 2, 3])
      expectClose(onlineAttention(q, k, v, block).output, reference(q, k, v, false));
  });
});

describe('Online Softmax 状态', () => {
  it('后续块出现更大的最大值时，l 和 a 都重新缩放', () => {
    const v = [
      [1, 0],
      [0, 2],
      [2, 1],
      [-1, 3],
    ];
    const blocks = traceOnlineRow([0, 1, 2, 3], v, 2);
    const last = blocks[1];
    expect(last.previousM).toBe(1);
    expect(last.m).toBe(3);
    expect(last.alpha).toBeCloseTo(Math.exp(-2), 14);
    expect(last.rescaled).toBe(true);
    expect(last.l).toBeCloseTo(Math.exp(-3) + Math.exp(-2) + Math.exp(-1) + 1, 14);
    for (let d = 0; d < 2; d++) {
      const expectedA = v.reduce((sum, value, j) => sum + Math.exp(j - 3) * value[d], 0);
      expect(last.a[d]).toBeCloseTo(expectedA, 14);
    }
    // Omitting rescaling changes the answer, despite each tile's own stable exponentials.
    const wrongL = last.previousL + last.blockL;
    const wrongA = last.previousA.map((value, d) => value + last.blockA[d]);
    expect(
      Math.max(...wrongA.map((value, d) => Math.abs(value / wrongL - last.output[d]))),
    ).toBeGreaterThan(0.1);
  });

  it('每个块的统计量都代表已读前缀，而非最终完整概率', () => {
    const scores = [2, -1, 3, 0, 4];
    const v = [
      [1, 2],
      [3, 4],
      [-2, 1],
      [0, -1],
      [2, 3],
    ];
    traceOnlineRow(scores, v, 2).forEach((block) => {
      const end = block.keys[block.keys.length - 1] + 1;
      const max = Math.max(...scores.slice(0, end));
      expect(block.m).toBe(max);
      const l = scores.slice(0, end).reduce((sum, s) => sum + Math.exp(s - max), 0);
      expect(block.l).toBeCloseTo(l, 14);
      expectClose(
        [block.output],
        [
          v[0].map(
            (_, d) =>
              v.slice(0, end).reduce((sum, val, j) => sum + Math.exp(scores[j] - max) * val[d], 0) /
              l,
          ),
        ],
      );
    });
  });

  it('首块、夹在中间和末尾的全 Mask 块均跳过；全行 Mask 返回零', () => {
    const v = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    const all = traceOnlineRow([-Infinity, -Infinity, -Infinity, -Infinity], v, 2);
    all.forEach((block) => {
      expect(block.m).toBe(-Infinity);
      expect(block.alpha).toBe(1);
      expect(block.l).toBe(0);
      expect(block.output).toEqual([0, 0]);
      expect(block.a).toEqual([0, 0]);
      expect(block.exponentials).toEqual([0, 0]);
    });
    const sparse = traceOnlineRow([-Infinity, 1, -Infinity, 2], v, 1);
    expect(sparse[2].a).toEqual(sparse[1].a);
    expect(sparse[2].l).toBe(sparse[1].l);
    expect(sparse[2].alpha).toBe(1);
    expect(sparse[2].rescaled).toBe(false);
    expect(sparse[3].output.every(Number.isFinite)).toBe(true);
    const causal = onlineAttention([[1], [2], [3], [4]], [[1], [2], [3], [4]], v, 2, true);
    expect(causal.rows[0][1].validCount).toBe(0);
    expect(causal.rows[0][1].output).toEqual(causal.rows[0][0].output);
  });

  it('输入、形状和块大小错误会明确拒绝', () => {
    expect(() => traceOnlineRow([NaN], [[1]], 1)).toThrow();
    expect(() => traceOnlineRow([Infinity], [[1]], 1)).toThrow();
    expect(() => traceOnlineRow([1], [[1]], 0)).toThrow();
    expect(() => traceOnlineRow([1], [[1]], 1.5)).toThrow();
    expect(() => onlineAttention([[1]], [[1, 2]], [[1]], 1)).toThrow();
    expect(() => onlineAttention([[1]], [[1], [2]], [[1], [2]], 1, true)).toThrow();
    expect(() => onlineAttention([], [[1]], [[1]], 1)).toThrow();
    expect(() => onlineAttention([[1]], [[1], [2]], [[1]], 1)).toThrow();
  });
});

describe('可复现输入与教学存储账本', () => {
  it('混合输入由种子复现；固定递增预设不受种子影响', () => {
    expect(makeFlashInputs('mixed', 42)).toEqual(makeFlashInputs('mixed', 42));
    expect(makeFlashInputs('mixed', 42)).not.toEqual(makeFlashInputs('mixed', 43));
    expect(makeFlashInputs('rising', 1)).toEqual(makeFlashInputs('rising', 9999));
  });

  it('分块改变临时空间，不改变完整配对规模；账本只统计指定中间量', () => {
    const small = flashLedger(4, 2, 2, 1, false);
    const large = flashLedger(4, 2, 2, 4, false);
    expect(small.pairCount).toBe(16);
    expect(large.pairCount).toBe(small.pairCount);
    expect(small.naiveIntermediateElements).toBe(32);
    expect(small.naiveIntermediateTransfers).toBe(64);
    expect(small.tiledIntermediateTransfers).toBe(0);
    expect(small.bytesPerElement).toBe(4);
    expect(small.tiledScratchElements).toBe(14);
    expect(large.tiledScratchElements).toBe(29);
    expect(flashLedger(4, 2, 2, 2, true).validPairs).toBe(10);
    expect(flashLedger(4, 2, 2, 2, true).pairCount).toBe(16);
  });
});
