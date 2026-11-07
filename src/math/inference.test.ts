import { describe, expect, it } from 'vitest';
import { attention, matmul } from './attention';
import {
  cacheFromForward,
  cachedStepWork,
  decodeStep,
  embedTokens,
  entryChanged,
  forwardFull,
  fullPassWork,
  generate,
  invalidatedByEdit,
  invalidatedByWindowSlide,
  MAX_DECODE_STEPS,
  NUM_LAYERS,
  TOY_MODEL,
  trainingRows,
  trainingWithEditedInput,
  VOCAB,
} from './inference';

const TOL = 1e-12;
const V = VOCAB.length;
/** Every 3-token prompt over the toy vocabulary. */
const PROMPTS = Array.from({ length: V ** 3 }, (_, n) => [
  Math.floor(n / (V * V)),
  Math.floor(n / V) % V,
  n % V,
]);

describe('完整因果前向', () => {
  it('第一层与独立的 Attention 参考实现一致，并加上残差', () => {
    const ids = [0, 3, 1, 5];
    const forward = forwardFull(TOY_MODEL, ids);
    const x = embedTokens(TOY_MODEL, ids);
    const { wq, wk, wv } = TOY_MODEL.layers[0];
    const reference = attention(matmul(x, wq), matmul(x, wk), matmul(x, wv), true, true);
    forward.layers[0].output.forEach((row, i) =>
      row.forEach((value, d) => expect(value).toBeCloseTo(x[i][d] + reference.output[i][d], 12)),
    );
  });

  it('每行输出分布的和为 1，且行数等于 token 数', () => {
    const forward = forwardFull(TOY_MODEL, [2, 2, 4, 0, 1]);
    expect(forward.probs).toHaveLength(5);
    forward.probs.forEach((row) =>
      expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(TOL),
    );
  });

  it('计数器与闭式公式一致：L·n 次 K/V 投影，L·n(n+1)/2 次打分', () => {
    for (let n = 1; n <= 8; n++)
      expect(forwardFull(TOY_MODEL, Array(n).fill(0)).work).toEqual(fullPassWork(n));
  });
});

describe('训练：一次前向算所有位置', () => {
  it('目标是右移一位的序列，loss = −ln p(目标)', () => {
    const rows = trainingRows(TOY_MODEL, [0, 1, 2, 3, 4]);
    expect(rows.map((r) => r.input)).toEqual([0, 1, 2, 3]);
    expect(rows.map((r) => r.target)).toEqual([1, 2, 3, 4]);
    rows.forEach((row) => {
      expect(row.loss).toBeCloseTo(-Math.log(row.probs[row.target]), 12);
      expect(row.visible).toEqual(Array.from({ length: row.position + 1 }, (_, i) => i));
    });
  });

  it('只改未来的输入，之前各行的输出分布逐位不变（被屏蔽的项乘以 0）', () => {
    for (let pos = 0; pos < 4; pos++)
      for (let token = 0; token < V; token++) {
        const { changedRows, rows } = trainingWithEditedInput(
          TOY_MODEL,
          [0, 1, 2, 3, 4],
          pos,
          token,
        );
        const base = trainingRows(TOY_MODEL, [0, 1, 2, 3, 4]);
        for (let i = 0; i < pos; i++) {
          expect(changedRows[i]).toBe(false);
          expect(rows[i].probs).toEqual(base[i].probs);
        }
        // Targets stay those of the original sample.
        expect(rows.map((r) => r.target)).toEqual([1, 2, 3, 4]);
      }
  });

  it('默认对照（改 x2 为 f）让第 2、3 行变化', () => {
    expect(trainingWithEditedInput(TOY_MODEL, [0, 1, 2, 3, 4], 2, 5).changedRows).toEqual([
      false,
      false,
      true,
      true,
    ]);
  });
});

describe('Prefill + Decode 与完整重算', () => {
  it('所有 3-token 提示、每一步 Decode：logits 与重算一致，生成的 token 相同', () => {
    for (const prompt of PROMPTS) {
      const generation = generate(TOY_MODEL, prompt, MAX_DECODE_STEPS);
      generation.steps.forEach((step) => {
        expect(step.logitsDiff).toBeLessThanOrEqual(TOL);
        const recomputed = step.recompute.probs[step.recompute.probs.length - 1];
        const best = recomputed.indexOf(Math.max(...recomputed));
        expect(step.produced).toBe(best);
      });
    }
  });

  it('Decode 只追加新位置：旧缓存条目原样保留，新条目等于重算得到的 K/V', () => {
    const prompt = [0, 1, 2];
    const prefill = forwardFull(TOY_MODEL, prompt);
    const cache = cacheFromForward(prefill);
    const step = decodeStep(TOY_MODEL, cache, 3);
    const full = cacheFromForward(forwardFull(TOY_MODEL, [...prompt, 3]));
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      expect(step.cache.k[layer].slice(0, 3)).toEqual(cache.k[layer]);
      expect(step.cache.v[layer].slice(0, 3)).toEqual(cache.v[layer]);
      for (let p = 0; p < 4; p++) expect(entryChanged(step.cache, full, layer, p, p)).toBe(false);
    }
    // The input cache object is not mutated.
    expect(cache.k[0]).toHaveLength(3);
  });

  it('缓存 Decode 每层只做 1 次 K/V 投影和 n 次打分', () => {
    const generation = generate(TOY_MODEL, [4, 4, 1], MAX_DECODE_STEPS);
    generation.steps.forEach((step, s) => {
      const n = 3 + s + 1;
      expect(step.decode.work).toEqual(cachedStepWork(n));
      expect(step.recompute.work).toEqual(fullPassWork(n));
    });
  });

  it('没有 Prefill 就不能 Decode；越界参数被拒绝', () => {
    expect(() => decodeStep(TOY_MODEL, { k: [[], []], v: [[], []] }, 0)).toThrow();
    expect(() => generate(TOY_MODEL, [0, 9], 1)).toThrow();
    expect(() => generate(TOY_MODEL, [0], MAX_DECODE_STEPS + 1)).toThrow();
  });
});

describe('缓存何时失效', () => {
  it('改第 j 个 token：第 1 层只有位置 j 变化，第 2 层位置 j 之前全部保留', () => {
    for (const prompt of PROMPTS.slice(0, 60)) {
      const ids = generate(TOY_MODEL, prompt, 3).tokens.slice(0, 6);
      for (let j = 0; j < 3; j++) {
        const token = (ids[j] + 1) % V;
        const changed = invalidatedByEdit(TOY_MODEL, ids, j, token);
        changed[0].forEach((flag, p) => expect(flag).toBe(p === j));
        for (let p = 0; p < j; p++) expect(changed[1][p]).toBe(false);
        expect(changed[1][j]).toBe(true);
      }
    }
  });

  it('默认对照：把位置 1 改成 f，第 2 层从位置 1 起全部失效', () => {
    const ids = generate(TOY_MODEL, [0, 1, 2], 3).tokens.slice(0, 6);
    expect(invalidatedByEdit(TOY_MODEL, ids, 1, 5)).toEqual([
      [false, true, false, false, false, false],
      [false, true, true, true, true, true],
    ]);
  });

  it('滑动窗口让位置编号整体平移：加法位置编码下，第 1 层每个旧条目都不再可用', () => {
    for (const prompt of PROMPTS) {
      const changed = invalidatedByWindowSlide(TOY_MODEL, prompt);
      expect(changed[0][0]).toBeNull();
      expect(changed[0].slice(1).every(Boolean)).toBe(true);
    }
  });
});
