import { describe, it, expect } from 'vitest';
import { initialLabState, validateLabState, restoreLabState, serializeLabState } from './p1';
import normalization from '../content/p1/normalization';
import ffn from '../content/p1/ffn';
import lora from '../content/p1/lora';
import quantization from '../content/p1/quantization';
import decoding from '../content/p1/decoding';
import preference from '../content/p1/preference';
import memory from '../content/p1/memory';
import ssm from '../content/p1/ssm';
import * as learning from '../content/p1/learning';
import type { TopicDefinition } from '../content/p1/types';
const topics: TopicDefinition[] = [
  normalization,
  ffn,
  lora,
  quantization,
  decoding,
  preference,
  memory,
  ssm,
  ...Object.values(learning),
];
describe('P1 可分享状态与完整计算', () => {
  for (const topic of topics) {
    it(`${topic.id} 状态、步骤、种子与字段往返，所有场景都可计算`, () => {
      const state = initialLabState(topic);
      state.step = 5;
      state.seed = 731;
      state.view = { explanations: false, speed: 2 };
      expect(restoreLabState(topic, serializeLabState(topic, state)).state).toEqual(state);
      const r = topic.compute(state.input, state.config, state.seed);
      expect(r.frames.length).toBe(topic.steps.length);
      for (const c of r.checks) expect(c.error, c.label).toBeLessThanOrEqual(c.tolerance);
      expect(r.signature.every(Number.isFinite)).toBe(true);
    });
    it(`${topic.id} 所有字段边界和预设都产生有限结果`, () => {
      const base = initialLabState(topic);
      for (const field of topic.fields)
        for (const value of field.options?.map((o) => o.value) ?? [field.min, field.max]) {
          const state = { ...base, config: { ...base.config, [field.key]: value } };
          expect(validateLabState(topic, state)).not.toBeNull();
          const r = topic.compute(state.input, state.config, state.seed);
          expect(r.signature.every(Number.isFinite), field.key).toBe(true);
          for (const f of r.frames)
            for (const t of f.tensors)
              for (const v of t.values.flat())
                expect(Number.isFinite(v) || v === -Infinity, field.key).toBe(true);
          for (const c of r.checks)
            expect(c.error, `${field.key}: ${c.label}`).toBeLessThanOrEqual(c.tolerance);
        }
      for (const preset of topic.examples) {
        const r = topic.compute(preset.input, { ...base.config, ...preset.parameters }, 42);
        expect(r.signature.every(Number.isFinite)).toBe(true);
      }
    });
  }
  it('拒绝跨主题、坏编码、版本、错误类型、越界与过长链接', () => {
    const state = initialLabState(normalization);
    expect(restoreLabState(ffn, serializeLabState(normalization, state)).warning).toBeTruthy();
    for (const hash of ['#lab=%ZZ', '#lab={oops', '#lab=' + 'x'.repeat(4000)])
      expect(restoreLabState(normalization, hash).state).toEqual(state);
    for (const patch of [
      { version: 2 },
      { step: 6 },
      { seed: 0 },
      { input: [NaN, 0, 0, 0] },
      { input: [1, 2] },
      { config: { ...state.config, temperature: 0 } },
      { config: { ...state.config, eps: 0 } },
      { view: { explanations: 'yes', speed: 1 } },
    ])
      expect(validateLabState(normalization, { ...state, ...patch })).toBeNull();
  });
  it('分享白名单不包含笔记或待执行预测', () => {
    const state = initialLabState(normalization),
      clean = validateLabState(normalization, {
        ...state,
        note: 'private',
        pending: 2,
        config: { ...state.config, secret: 'private' },
        view: { ...state.view, note: 'private' },
      });
    expect(clean).toEqual(state);
    expect(JSON.stringify(clean)).not.toContain('private');
  });
});
