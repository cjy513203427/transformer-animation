import { describe, expect, it } from 'vitest';
import {
  initialMoEState,
  moveMoEStep,
  restoreMoEState,
  serializeMoEState,
  validateMoEState,
} from './moe';

describe('MoE 可分享状态', () => {
  it('完整参数、种子、步骤与展示选项往返恢复', () => {
    const state = initialMoEState();
    state.step = 5;
    state.seed = 9999;
    state.config = { topK: 1, topology: 'single' };
    state.input = { preset: 'concentrated', token: 3, delta: -1.75, expertScale: 0.25 };
    state.view = { explanations: false, speed: 2 };
    expect(restoreMoEState(serializeMoEState(state))).toEqual({ state, warning: null });
    expect(restoreMoEState('')).toEqual({ state: initialMoEState(), warning: null });
  });

  it.each(['#moe=%', '#moe=null', '#moe=[]', '#moe={}', '#moe=' + 'x'.repeat(2001), '#kv=1'])(
    '异常链接安全回退：%s',
    (hash) => {
      const result = restoreMoEState(hash);
      expect(result.state).toEqual(initialMoEState());
      expect(result.warning).toContain('分享链接无效');
    },
  );

  it('拒绝错误版本、非有限数、范围与刻度错误、错误结构与类型', () => {
    const state = initialMoEState();
    const cases = [
      null,
      [],
      { ...state, version: 2 },
      { ...state, step: 6 },
      { ...state, step: -1 },
      { ...state, seed: 0 },
      { ...state, seed: 10000 },
      { ...state, seed: 1.5 },
      { ...state, config: null },
      { ...state, config: { ...state.config, topK: 3 } },
      { ...state, config: { ...state.config, topology: 'other' } },
      { ...state, input: { ...state.input, preset: 'other' } },
      { ...state, input: { ...state.input, token: 4 } },
      { ...state, input: { ...state.input, delta: Infinity } },
      { ...state, input: { ...state.input, delta: -2.25 } },
      { ...state, input: { ...state.input, delta: 0.1 } },
      { ...state, input: { ...state.input, expertScale: NaN } },
      { ...state, input: { ...state.input, expertScale: 2.25 } },
      { ...state, input: { ...state.input, expertScale: '1' } },
      { ...state, view: { ...state.view, explanations: 1 } },
      { ...state, view: { ...state.view, speed: 3 } },
    ];
    for (const value of cases) expect(validateMoEState(value)).toBeNull();
    expect(() => serializeMoEState({ ...state, step: 99 })).toThrow();
  });

  it('移除各层未知字段，私人笔记与答案不进入分享链接', () => {
    const base = initialMoEState();
    const clean = validateMoEState({
      ...base,
      notes: 'PRIVATE',
      config: { ...base.config, private: 'PRIVATE' },
      input: { ...base.input, prediction: 'PRIVATE' },
      view: { ...base.view, answer: 'PRIVATE' },
    });
    expect(clean).toEqual(base);
    expect(JSON.stringify(clean)).not.toContain('PRIVATE');
  });

  it('移动步骤不越界；默认状态之间不共享嵌套对象', () => {
    expect(moveMoEStep(initialMoEState(), -1).step).toBe(0);
    expect(moveMoEStep(initialMoEState(), 99).step).toBe(5);
    expect(moveMoEStep({ ...initialMoEState(), step: 4 }, -1).step).toBe(3);
    const a = initialMoEState();
    a.input.delta = 1;
    expect(initialMoEState().input.delta).toBe(0);
  });
});
