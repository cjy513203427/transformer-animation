import { describe, expect, it } from 'vitest';
import {
  initialLinearState,
  moveLinearStep,
  restoreLinearState,
  serializeLinearState,
  validateLinearState,
} from './linear';

describe('二维线性变换可分享状态', () => {
  it('完整状态往返序列化并还原', () => {
    const state = initialLinearState();
    state.step = 3;
    state.w = [
      [0.5, -1.25],
      [1.75, 0.25],
    ];
    state.x = [-1.5, 0.75];
    state.k = [0.5, -2];
    state.view = { explanations: false, speed: 2 };
    expect(restoreLinearState(serializeLinearState(state))).toEqual({ state, warning: null });
    expect(restoreLinearState('')).toEqual({ state: initialLinearState(), warning: null });
  });

  it.each(['#lin=%', '#lin=null', '#lin={}', '#lin=' + 'x'.repeat(2001), '#moe=1'])(
    '异常链接安全回退：%s',
    (hash) => {
      const result = restoreLinearState(hash);
      expect(result.state).toEqual(initialLinearState());
      expect(result.warning).toContain('分享链接无效');
    },
  );

  it('拒绝越界、非步长、错版本与未知字段污染', () => {
    const base = initialLinearState();
    for (const bad of [
      { ...base, version: 2 },
      { ...base, step: 5 },
      { ...base, w: [[2.5, 0], [0, 1]] },
      { ...base, x: [0.1, 0] },
      { ...base, k: [NaN, 0] },
      { ...base, view: { explanations: true, speed: 4 } },
    ]) {
      expect(validateLinearState(bad)).toBeNull();
    }
    const cleaned = validateLinearState({ ...base, secret: 'PRIVATE' });
    expect(cleaned).toEqual(base);
    expect(moveLinearStep(base, -1).step).toBe(0);
    expect(moveLinearStep(base, 99).step).toBe(4);
  });
});
