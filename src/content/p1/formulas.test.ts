import { expect, it } from 'vitest';
import katex from 'katex';
import normalization from './normalization';
import ffn from './ffn';
import lora from './lora';
import quantization from './quantization';
import decoding from './decoding';
import preference from './preference';
import memory from './memory';
import ssm from './ssm';
it('P1 的48步公式都可严格解析，每组都有来源、边界与完整步骤', () => {
  for (const topic of [normalization, ffn, lora, quantization, decoding, preference, memory, ssm]) {
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
