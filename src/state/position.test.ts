import { describe, expect, it } from 'vitest';
import {
  initialPositionState,
  movePositionStep,
  restorePositionState,
  serializePositionState,
  validatePositionState,
} from './position';

describe('位置编码实验的分享状态', () => {
  it('往返恢复全部公开参数', () => {
    const state = initialPositionState();
    state.step = 4;
    state.view = { explanations: false, speed: 2 };
    state.blind = { withPE: true, pair: 'causal' };
    state.sin = { dModel: 512, pos: 37, pair: 200 };
    state.rope = { m: 11, n: 3 };
    state.alibi = { head: 7, query: 2, content: 'example' };
    expect(restorePositionState(serializePositionState(state))).toEqual({ state, warning: null });
  });
  it.each(['#pos=%', '#pos=null', '#pos=[]', '#pos={}', '#pos=' + 'x'.repeat(3001), '#lab=1'])(
    '异常链接安全回退：%s',
    (hash) => {
      const result = restorePositionState(hash);
      expect(result.state).toEqual(initialPositionState());
      expect(result.warning).toBeTruthy();
    },
  );
  it('拒绝越界、类型错误与依赖 d_model 的维度对编号', () => {
    const base = initialPositionState();
    const cases = [
      { ...base, version: 2 },
      { ...base, step: 7 },
      { ...base, view: { ...base.view, speed: 3 } },
      { ...base, blind: { ...base.blind, pair: 'other' } },
      { ...base, sin: { ...base.sin, dModel: 32 } },
      { ...base, sin: { ...base.sin, pos: 64 } },
      { ...base, sin: { dModel: 4, pos: 1, pair: 2 } },
      { ...base, rope: { m: 41, n: 0 } },
      { ...base, rope: { m: 1.5, n: 0 } },
      { ...base, alibi: { ...base.alibi, head: 8 } },
      { ...base, alibi: { ...base.alibi, content: 'random' } },
    ];
    for (const value of cases) expect(validatePositionState(value)).toBeNull();
    expect(validatePositionState({ ...base, sin: { dModel: 4, pos: 1, pair: 1 } })).not.toBeNull();
  });
  it('丢弃白名单之外的字段', () => {
    const link = serializePositionState({
      ...initialPositionState(),
      notes: 'PRIVATE',
    } as ReturnType<typeof initialPositionState>);
    expect(link).not.toContain('PRIVATE');
  });
  it('步骤移动不越界，默认状态互不共享引用', () => {
    expect(movePositionStep(initialPositionState(), -1).step).toBe(0);
    expect(movePositionStep(initialPositionState(), 99).step).toBe(6);
    const first = initialPositionState();
    first.rope.m = 9;
    expect(initialPositionState().rope.m).toBe(2);
  });
});
