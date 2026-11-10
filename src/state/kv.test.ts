import { describe, expect, it } from 'vitest';
import {
  initialKVState,
  kvHeadChoices,
  moveKVStep,
  restoreKVState,
  serializeKVState,
  validateKVState,
} from './kv';

describe('KV 架构实验的分享状态', () => {
  it('往返恢复全部公开参数', () => {
    const state = initialKVState();
    state.step = 4;
    state.view = { explanations: false, speed: 2 };
    state.calc = {
      batch: 8,
      layers: 80,
      tokens: 32768,
      qHeads: 64,
      kvHeads: 8,
      dHead: 128,
      bytes: 1,
    };
    state.group = { kvHeads: 1, head: 3, query: 0 };
    state.mla = { head: 2, query: 1, coupled: false };
    state.compare = { tokens: 131072 };
    expect(restoreKVState(serializeKVState(state))).toEqual({ state, warning: null });
  });

  it.each(['#kv=%', '#kv=null', '#kv=[]', '#kv={}', '#kv=' + 'x'.repeat(2001), '#inf=1'])(
    '异常链接安全回退：%s',
    (hash) => {
      const result = restoreKVState(hash);
      expect(result.state).toEqual(initialKVState());
      expect(result.warning).toBeTruthy();
    },
  );

  it('拒绝越界、不能整除的分组与类型错误', () => {
    const base = initialKVState();
    const cases = [
      { ...base, version: 2 },
      { ...base, step: 6 },
      { ...base, view: { ...base.view, speed: 3 } },
      { ...base, calc: { ...base.calc, kvHeads: 3 } },
      { ...base, calc: { ...base.calc, kvHeads: 64 } },
      { ...base, calc: { ...base.calc, bytes: 3 } },
      { ...base, calc: { ...base.calc, tokens: 0 } },
      { ...base, calc: { ...base.calc, layers: 1.5 } },
      { ...base, group: { ...base.group, kvHeads: 3 } },
      { ...base, group: { ...base.group, head: 4 } },
      { ...base, mla: { ...base.mla, query: 4 } },
      { ...base, mla: { ...base.mla, coupled: 'yes' } },
      { ...base, compare: { tokens: 2000 } },
    ];
    for (const value of cases) expect(validateKVState(value)).toBeNull();
  });

  it('丢弃白名单之外的字段', () => {
    const link = serializeKVState({
      ...initialKVState(),
      notes: 'PRIVATE',
    } as ReturnType<typeof initialKVState>);
    expect(link).not.toContain('PRIVATE');
  });

  it('KV 头数候选只包含能整除 Q 头数的值；步骤移动不越界', () => {
    expect(kvHeadChoices(32)).toEqual([32, 16, 8, 4, 2, 1]);
    expect(kvHeadChoices(12)).toEqual([12, 6, 4, 3, 2, 1]);
    expect(moveKVStep(initialKVState(), -1).step).toBe(0);
    expect(moveKVStep(initialKVState(), 99).step).toBe(5);
    const first = initialKVState();
    first.calc.layers = 1;
    expect(initialKVState().calc.layers).toBe(32);
  });
});
