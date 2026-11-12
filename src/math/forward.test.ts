import { describe, expect, it } from 'vitest';
import {
  decoderForward,
  DECODER_EMBEDDING,
  DECODER_EPSILON,
  decoderLayers,
  forwardStepCount,
} from './forward';

/** Independent scalar reference, including per-head feature correspondence and vocabulary axis. */
function reference(tokens: number[], depth: number, causal: boolean, seed: number) {
  const product = (x: number[][], w: number[][]) =>
    x.map((row) => w[0].map((_, j) => row.reduce((s, v, k) => s + v * w[k][j], 0)));
  const ln = (x: number[][]) =>
    x.map((row) => {
      const mean = row.reduce((s, v) => s + v, 0) / row.length;
      const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
      return row.map((v) => (v - mean) / Math.sqrt(variance + DECODER_EPSILON));
    });
  const probability = (x: number[]) => {
    const max = Math.max(...x),
      exp = x.map((v) => Math.exp(v - max)),
      sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / sum);
  };
  let h = tokens.map((id, t) =>
    DECODER_EMBEDDING[id].map(
      (v, d) => v + (d % 2 ? Math.cos : Math.sin)(t / 10000 ** ((2 * Math.floor(d / 2)) / 4)),
    ),
  );
  for (const weights of decoderLayers(seed).slice(0, depth)) {
    const norm = ln(h),
      q = product(norm, weights.attention.wq),
      k = product(norm, weights.attention.wk),
      v = product(norm, weights.attention.wv);
    const joined = tokens.map((_, t) =>
      [0, 1].flatMap((head) => {
        const p = probability(
          tokens.map((_, j) =>
            causal && j > t
              ? -Infinity
              : [0, 1].reduce((s, d) => s + q[t][head * 2 + d] * k[j][head * 2 + d], 0) /
                Math.sqrt(2),
          ),
        );
        return [0, 1].map((d) => p.reduce((s, w, j) => s + w * v[j][head * 2 + d], 0));
      }),
    );
    const a = product(joined, weights.attention.wo);
    const residual = h.map((row, t) => row.map((v, d) => v + a[t][d]));
    const g = product(ln(residual), weights.w1).map((row) =>
      row.map(
        (x) => (x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))) / 2,
      ),
    );
    const f = product(g, weights.w2);
    h = residual.map((row, t) => row.map((v, d) => v + f[t][d]));
  }
  return ln(h).map((row) =>
    DECODER_EMBEDDING.map((e) => e.reduce((sum, v, d) => sum + v * row[d], 0)),
  );
}

describe('完整 Pre-Norm 前向链', () => {
  it('端到端结果与独立标量计算一致，所有位置沿词表归一化', () => {
    for (const depth of [1, 2] as const)
      for (const causal of [false, true])
        for (const seed of [0, 42, 255]) {
          const result = decoderForward([0, 1, 2], depth, causal, seed),
            expected = reference([0, 1, 2], depth, causal, seed);
          expect(result.frames.length).toBe(forwardStepCount(depth));
          result.logits.forEach((row, t) =>
            row.forEach((v, d) => expect(v).toBeCloseTo(expected[t][d], 12)),
          );
          result.probabilities.forEach((row) =>
            expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12),
          );
          expect(new Set(result.frames.map((f) => f.id)).size).toBe(result.frames.length);
        }
  });
  it('未来 token 不影响早期位置；关闭因果限制时确实存在影响', () => {
    const before = decoderForward([0, 1, 2], 2, true),
      after = decoderForward([0, 1, 3], 2, true);
    expect(after.logits[0]).toEqual(before.logits[0]);
    expect(after.logits[1]).toEqual(before.logits[1]);
    expect(after.logits[2]).not.toEqual(before.logits[2]);
    expect(decoderForward([0, 1, 3], 2, false).logits[0]).not.toEqual(
      decoderForward([0, 1, 2], 2, false).logits[0],
    );
  });
  it('输入与参数不被前向修改，重算轨迹可重建', () => {
    const tokens = [0, 1, 2],
      embedding = DECODER_EMBEDDING.map((row) => [...row]);
    expect(decoderForward(tokens, 2, true)).toEqual(decoderForward(tokens, 2, true));
    expect(tokens).toEqual([0, 1, 2]);
    expect(DECODER_EMBEDDING).toEqual(embedding);
  });
});
