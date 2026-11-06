import { describe, expect, it } from 'vitest';
import {
  initialState,
  moveHeadStep,
  moveStep,
  presetState,
  restoreState,
  serializeState,
  validateState,
} from './experiment';

describe('分享状态与步骤', () => {
  it('往返恢复所有公开实验参数', () => {
    const state = presetState('value');
    state.seed = 919;
    state.view = { ...state.view, query: 0, key: 2, explanations: false, speed: 2 };
    state.input.x[0][0] = -1.5;
    expect(restoreState(serializeState(state))).toEqual({ state, warning: null });
  });
  it.each([
    '#lab=%',
    '#lab=null',
    '#lab=[]',
    '#lab={}',
    '#lab=' + 'x'.repeat(6001),
    '#unknown=123',
  ])('异常链接安全回退：%s', (hash) => {
    const result = restoreState(hash);
    expect(result.state).toEqual(initialState());
    expect(result.warning).toBeTruthy();
  });
  it('拒绝版本、维度、类型及范围错误', () => {
    const cases = [
      { ...initialState(), version: 4 },
      { ...initialState(), heads: undefined },
      { ...initialState(), heads: { ...initialState().heads, numHeads: 3 } },
      { ...initialState(), heads: { ...initialState().heads, step: 6 } },
      { ...initialState(), heads: { ...initialState().heads, query: 3 } },
      { ...initialState(), heads: { ...initialState().heads, keyDelta: 4.5 } },
      { ...initialState(), view: { ...initialState().view, lesson: 'multi' } },
      { ...initialState(), step: 5 },
      { ...initialState(), seed: 0.2 },
      { ...initialState(), config: { causal: 'true', scaled: true } },
      { ...initialState(), input: { x: [[1, 2]], keyDelta: 0, valueDelta: 0 } },
      { ...initialState(), input: { ...initialState().input, valueDelta: 5 } },
      { ...initialState(), view: { ...initialState().view, query: -1 } },
      { ...initialState(), view: { ...initialState().view, speed: 1.5 } },
      { ...initialState(), view: { ...initialState().view, lesson: 'unknown' } },
      { ...initialState(), view: { ...initialState().view, architectureNode: 'unknown' } },
    ];
    for (const value of cases) expect(validateState(value)).toBeNull();
  });
  it('丢弃不在白名单中的笔记等字段', () => {
    const result = serializeState({ ...initialState(), notes: 'PRIVATE' } as ReturnType<
      typeof initialState
    >);
    expect(result).not.toContain('PRIVATE');
  });
  it('回退再前进精确还原状态；边界不越界', () => {
    const state = presetState('value');
    expect(moveStep(moveStep(state, -1), 1)).toEqual(state);
    expect(moveStep(state, 1).step).toBe(4);
    expect(moveStep(initialState(), -1).step).toBe(0);
  });
  it('预设与重置不共享可变输入', () => {
    const first = initialState();
    first.input.x[0][0] = 4;
    expect(initialState().input.x[0][0]).toBe(1);
    expect(presetState('causal').config.causal).toBe(true);
    expect(presetState('value').input.valueDelta).toBe(2);
  });
  it('旧版分享迁移到数值视图，保留原来的输入与条件', () => {
    const old = {
      ...initialState(),
      version: 1,
      config: { causal: false, scaled: true },
      view: { query: 2, key: 1, explanations: true, speed: 1 },
    };
    const restored = restoreState('#lab=' + encodeURIComponent(JSON.stringify(old)));
    expect(restored.warning).toBeNull();
    expect(restored.state.version).toBe(3);
    expect(restored.state.heads).toEqual(initialState().heads);
    expect(restored.state.view.lesson).toBe('numbers');
    expect(restored.state.view.query).toBe(2);
    expect(restored.state.config.causal).toBe(false);
  });
  it('导读层次和选中架构模块可分享恢复', () => {
    const state = initialState();
    state.view.architectureNode = 'ffn-add';
    expect(restoreState(serializeState(state)).state).toEqual(state);
    state.view.lesson = 'guide';
    state.step = 3;
    expect(restoreState(serializeState(state)).state).toEqual(state);
  });
  it('v2 链接迁移到 v3，多头课程从默认示例开始', () => {
    const v2 = { ...initialState(), version: 2, heads: undefined };
    v2.view = { ...v2.view, lesson: 'guide' };
    v2.step = 2;
    const restored = restoreState('#lab=' + encodeURIComponent(JSON.stringify(v2)));
    expect(restored.warning).toBeNull();
    expect(restored.state.view.lesson).toBe('guide');
    expect(restored.state.step).toBe(2);
    expect(restored.state.heads).toEqual(initialState().heads);
  });
  it('多头课程的步骤、头数、观察位置与已执行的干预可分享恢复', () => {
    const state = initialState();
    state.view.lesson = 'heads';
    state.heads = { step: 5, numHeads: 4, query: 0, keyDelta: -2.5 };
    expect(restoreState(serializeState(state))).toEqual({ state, warning: null });
  });
  it('多头步骤独立于单头步骤，且不越界', () => {
    const state = initialState();
    const forward = moveHeadStep(state, 10);
    expect(forward.heads.step).toBe(5);
    expect(forward.step).toBe(state.step);
    expect(moveHeadStep(forward, -10).heads.step).toBe(0);
  });
});
