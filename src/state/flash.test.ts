import { describe, expect, it } from 'vitest';
import {
  initialFlashState,
  moveFlashStep,
  restoreFlashState,
  serializeFlashState,
  validateFlashState,
} from './flash';

describe('FlashAttention 可分享状态', () => {
  it('输入、Mask、分块、选中块、种子与阅读步骤完整往返', () => {
    const state = initialFlashState();
    state.step = 5;
    state.seed = 9999;
    state.config = { blockSize: 1, causal: false };
    state.input = { preset: 'mixed', query: 0, keyDelta: -1.75, block: 3 };
    state.view = { explanations: false, speed: 2 };
    expect(restoreFlashState(serializeFlashState(state))).toEqual({ state, warning: null });
    expect(restoreFlashState('')).toEqual({ state: initialFlashState(), warning: null });
  });

  it.each([
    '#flash=%',
    '#flash=null',
    '#flash=[]',
    '#flash={}',
    '#flash=' + 'x'.repeat(2001),
    '#moe=1',
  ])('坏链接恢复默认示例：%s', (hash) => {
    const result = restoreFlashState(hash);
    expect(result.state).toEqual(initialFlashState());
    expect(result.warning).toContain('分享链接无效');
  });

  it('拒绝版本、类型、非有限数、错误刻度与越界配置', () => {
    const state = initialFlashState();
    const invalid = [
      null,
      [],
      { ...state, version: 2 },
      { ...state, step: 6 },
      { ...state, step: -1 },
      { ...state, seed: 0 },
      { ...state, seed: 10000 },
      { ...state, seed: 2.5 },
      { ...state, config: null },
      { ...state, config: { ...state.config, blockSize: 0 } },
      { ...state, config: { ...state.config, blockSize: 5 } },
      { ...state, config: { ...state.config, blockSize: 1.5 } },
      { ...state, config: { ...state.config, causal: 'true' } },
      { ...state, input: { ...state.input, preset: 'other' } },
      { ...state, input: { ...state.input, query: 4 } },
      { ...state, input: { ...state.input, query: -1 } },
      { ...state, input: { ...state.input, keyDelta: Infinity } },
      { ...state, input: { ...state.input, keyDelta: NaN } },
      { ...state, input: { ...state.input, keyDelta: 2.25 } },
      { ...state, input: { ...state.input, keyDelta: 0.1 } },
      { ...state, input: { ...state.input, block: 2 } },
      { ...state, input: { ...state.input, block: -1 } },
      { ...state, view: { ...state.view, explanations: 1 } },
      { ...state, view: { ...state.view, speed: 3 } },
    ];
    for (const value of invalid) expect(validateFlashState(value)).toBeNull();
    expect(() => serializeFlashState({ ...state, step: 99 })).toThrow();
  });

  it('块索引随块大小校验，尾块不要求填满', () => {
    const state = initialFlashState();
    expect(
      validateFlashState({
        ...state,
        config: { ...state.config, blockSize: 3 },
        input: { ...state.input, block: 1 },
      }),
    ).not.toBeNull();
    expect(
      validateFlashState({
        ...state,
        config: { ...state.config, blockSize: 4 },
        input: { ...state.input, block: 1 },
      }),
    ).toBeNull();
  });

  it('分享字段白名单移除笔记和临时预测', () => {
    const base = initialFlashState();
    const clean = validateFlashState({
      ...base,
      notes: 'PRIVATE',
      config: { ...base.config, private: 'PRIVATE' },
      input: { ...base.input, prediction: 'PRIVATE' },
      view: { ...base.view, answer: 'PRIVATE' },
    });
    expect(clean).toEqual(base);
    expect(JSON.stringify(clean)).not.toContain('PRIVATE');
  });

  it('步进不越界且新默认状态不共享嵌套对象', () => {
    expect(moveFlashStep(initialFlashState(), -1).step).toBe(0);
    expect(moveFlashStep(initialFlashState(), 99).step).toBe(5);
    expect(moveFlashStep({ ...initialFlashState(), step: 4 }, -1).step).toBe(3);
    const state = initialFlashState();
    state.input.keyDelta = 1;
    expect(initialFlashState().input.keyDelta).toBe(0);
  });
});
