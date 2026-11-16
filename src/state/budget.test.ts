import { expect, it } from 'vitest';
import {
  BUDGET_PRESETS,
  budgetProblem,
  initialBudgetState,
  restoreBudgetState,
  serializeBudgetState,
} from './budget';

it('分享状态往返，额外字段被丢弃', () => {
  const state = initialBudgetState();
  state.config = { ...BUDGET_PRESETS.llama70b, context: 8192 };
  state.view.recall = true;
  expect(restoreBudgetState(serializeBudgetState(state))).toEqual({ state, invalid: false });
  const extra = JSON.parse(decodeURIComponent(serializeBudgetState(state).slice(8)));
  extra.config.secret = 'x';
  const restored = restoreBudgetState('#budget=' + encodeURIComponent(JSON.stringify(extra)));
  expect(restored.state.config).not.toHaveProperty('secret');
});

it('拒绝错误版本、类型、范围、长度与不成立的头数组合', () => {
  const bad = (patch: Record<string, unknown>, top: Record<string, unknown> = {}) => {
    const s = { ...initialBudgetState(), ...top } as Record<string, unknown>;
    s.config = { ...BUDGET_PRESETS.gpt2, ...patch };
    return restoreBudgetState('#budget=' + encodeURIComponent(JSON.stringify(s))).invalid;
  };
  expect(bad({}, { version: 2 })).toBe(true);
  expect(bad({ layers: 2.5 })).toBe(true);
  expect(bad({ vocab: '50257' })).toBe(true);
  expect(bad({ ffn: 'relu' })).toBe(true);
  expect(bad({ context: 0 })).toBe(true);
  expect(bad({ heads: 7 })).toBe(true); // 768 / 7 is not an integer
  expect(bad({ heads: 12, kvHeads: 5 })).toBe(true);
  expect(bad({ heads: 12, kvHeads: 4 })).toBe(false);
  expect(restoreBudgetState('#budget=' + 'x'.repeat(1300)).invalid).toBe(true);
  expect(restoreBudgetState('#budget=%E0').invalid).toBe(true);
  expect(restoreBudgetState('').invalid).toBe(false);
  for (const preset of Object.values(BUDGET_PRESETS)) expect(budgetProblem(preset)).toBeNull();
});
