import { describe, expect, it } from 'vitest';
import katex from 'katex';
import normalization from './normalization';
import ffn from './ffn';
import lora from './lora';
import quantization from './quantization';
import decoding from './decoding';
import preference from './preference';
import memory from './memory';
import ssm from './ssm';
import { explain, inputRange, lessonContext, type TopicDefinition } from './types';
import { initialLabState } from '../../state/p1';
import { BOOK_URL } from '../topics';

const topics: TopicDefinition[] = [
  normalization,
  ffn,
  lora,
  quantization,
  decoding,
  preference,
  memory,
  ssm,
];
const BARE_CONTROL_WORD =
  /(^|[^\\a-zA-Z])(sqrt|frac|mathrm|theta|omega|times|cdot|operatorname|sum|ldots|alpha|sigma|epsilon|phi|Delta)(?![a-zA-Z])/;
const inlineMath = (text: string) => [...text.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]);
function checkProse(text: string, where: string) {
  for (const tex of inlineMath(text)) {
    expect(
      () => katex.renderToString(tex, { throwOnError: true, strict: 'ignore' }),
      where,
    ).not.toThrow();
    expect(tex, `${where}：缺少反斜杠 ${tex}`).not.toMatch(BARE_CONTROL_WORD);
  }
}

it('P1 的48步公式都可严格解析，每组都有来源、边界与完整步骤', () => {
  for (const topic of topics) {
    expect(topic.steps.length).toBe(6);
    expect(topic.sources.length).toBeGreaterThan(0);
    expect(topic.limits.length).toBeGreaterThan(0);
    for (const step of topic.steps)
      expect(
        () =>
          katex.renderToString(step.formula, {
            throwOnError: true,
            strict: 'ignore',
            displayMode: true,
          }),
        `${topic.id}/${step.name}`,
      ).not.toThrow();
  }
});

describe('P1 讲解与原书对照', () => {
  it('每一步都链接到原书章节，开场段也有原书出处', () => {
    for (const topic of topics) {
      expect(topic.why.book[1].startsWith(BOOK_URL), topic.id).toBe(true);
      for (const step of topic.steps)
        expect(step.book[1].startsWith(BOOK_URL), `${topic.id}/${step.name}`).toBe(true);
    }
  });

  it('默认输入与每个预设下，逐步解释都能生成，且不含 NaN 或 undefined', () => {
    for (const topic of topics) {
      const base = initialLabState(topic);
      const [low, high] = inputRange(topic);
      for (const example of [
        { name: '默认', input: topic.input, parameters: {} },
        ...topic.examples,
      ]) {
        expect(example.input.length, `${topic.id}/${example.name}`).toBe(topic.input.length);
        expect(
          example.input.every((v) => v >= low && v <= high),
          `${topic.id}/${example.name}`,
        ).toBe(true);
        const config = { ...base.config, ...example.parameters };
        const result = topic.compute(example.input, config, base.seed);
        topic.steps.forEach((step, i) => {
          const where = `${topic.id}/${example.name}/${step.name}`;
          const text = explain(
            step.explanation,
            lessonContext(example.input, config, result.frames[i]),
          );
          expect(text, where).not.toMatch(/NaN|undefined|Infinity/);
          checkProse(text, where);
        });
      }
      for (const text of [
        ...topic.why.points,
        ...(topic.why.analogy ? [topic.why.analogy.text] : []),
        ...(topic.bookNotes ?? []),
        ...topic.steps.flatMap((s) => [s.prompt, s.takeaway, s.title]),
      ])
        checkProse(text, topic.id);
    }
  });

  it('书中预设复现书里印出的手算结果', () => {
    // 第 6.2.4 节：[22, 5, 6, 8] → [1.7105, −0.7643, −0.6187, −0.3275]（配图 4 位小数）
    const ln = normalization.compute([22, 5, 6, 8], initialLabState(normalization).config);
    const z = ln.frames[1].tensors[1].values[0];
    [1.7105, -0.7643, -0.6187, -0.3275].forEach((v, i) => expect(z[i]).toBeCloseTo(v, 4));
    // 第 6.3.4 节：[3.01, 0.09, 2.48, 1.95] → 50.28%、2.71%、29.59%、17.42%
    const sm = normalization.compute(
      [3.01, 0.09, 2.48, 1.95],
      initialLabState(normalization).config,
    );
    const p = sm.frames[4].tensors[2].values[0];
    [0.5028, 0.0271, 0.2959, 0.1742].forEach((v, i) => expect(p[i]).toBeCloseTo(v, 3));
    // 附录 B.4.2：logits [2, 1, 0.5]，τ = 1 → [0.63, 0.23, 0.14]
    const dec = decoding.compute([2, 1, 0.5], initialLabState(decoding).config, 42);
    [0.63, 0.23, 0.14].forEach((v, i) =>
      expect(dec.frames[0].tensors[1].values[0][i]).toBeCloseTo(v, 2),
    );
    // 第 31.4.3 节：[0.8, 0.3, 0.9, 0.5] 的组平均 0.625
    const grpo = preference.compute([0.8, 0.3, 0.9, 0.5], initialLabState(preference).config);
    expect(lessonContext([], {}, grpo.frames[3]).metric('组均值')).toBeCloseTo(0.625, 12);
  });
});
