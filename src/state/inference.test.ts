import { describe, expect, it } from 'vitest';
import {
  initialInferenceState,
  moveInferenceStep,
  restoreInferenceState,
  serializeInferenceState,
  validateInferenceState,
} from './inference';

describe('训练与缓存实验的分享状态', () => {
  it('往返恢复全部公开参数', () => {
    const state = initialInferenceState();
    state.step = 5;
    state.view = { explanations: false, speed: 0.5 };
    state.train = { pos: 3, token: 0, applied: true };
    state.decode = { steps: 5 };
    state.cache = { mode: 'slide', pos: 2, token: 4 };
    expect(restoreInferenceState(serializeInferenceState(state))).toEqual({
      state,
      warning: null,
    });
  });

  it.each(['#inf=%', '#inf=null', '#inf=[]', '#inf={}', '#inf=' + 'x'.repeat(2001), '#pos=1'])(
    '异常链接安全回退：%s',
    (hash) => {
      const result = restoreInferenceState(hash);
      expect(result.state).toEqual(initialInferenceState());
      expect(result.warning).toBeTruthy();
    },
  );

  it('拒绝越界与类型错误', () => {
    const base = initialInferenceState();
    const cases = [
      { ...base, version: 2 },
      { ...base, step: 6 },
      { ...base, view: { ...base.view, speed: 4 } },
      { ...base, train: { ...base.train, pos: 4 } },
      { ...base, train: { ...base.train, token: 6 } },
      { ...base, train: { ...base.train, applied: 'yes' } },
      { ...base, decode: { steps: 0 } },
      { ...base, decode: { steps: 6 } },
      { ...base, decode: { steps: 2.5 } },
      { ...base, cache: { ...base.cache, mode: 'clear' } },
      { ...base, cache: { ...base.cache, pos: 3 } },
      { ...base, cache: { ...base.cache, token: -1 } },
    ];
    for (const value of cases) expect(validateInferenceState(value)).toBeNull();
  });

  it('丢弃白名单之外的字段', () => {
    const link = serializeInferenceState({
      ...initialInferenceState(),
      notes: 'PRIVATE',
    } as ReturnType<typeof initialInferenceState>);
    expect(link).not.toContain('PRIVATE');
  });

  it('步骤移动不越界，默认状态互不共享引用', () => {
    expect(moveInferenceStep(initialInferenceState(), -1).step).toBe(0);
    expect(moveInferenceStep(initialInferenceState(), 99).step).toBe(5);
    const first = initialInferenceState();
    first.decode.steps = 5;
    expect(initialInferenceState().decode.steps).toBe(3);
  });
});
