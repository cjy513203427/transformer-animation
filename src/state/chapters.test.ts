import { expect, it } from 'vitest';
import { initialForwardState, restoreForwardState, serializeForwardState } from './forward';
import {
  initialArchitectureState,
  restoreArchitectureState,
  serializeArchitectureState,
} from './architecture';

it('章节实验分享状态往返，非法长度、类型和越界步骤回退', () => {
  const f = initialForwardState();
  f.step = 22;
  f.input.tokens = [3, 1, 0];
  expect(restoreForwardState(serializeForwardState(f))).toEqual({ state: f, invalid: false });
  const a = initialArchitectureState();
  a.input.signal = [-4, 1, 2, 4];
  a.view = { node: 6, explanations: false };
  expect(restoreArchitectureState(serializeArchitectureState(a))).toEqual({
    state: a,
    invalid: false,
  });
  for (const hash of [
    '#forward=%',
    '#forward=' + 'a'.repeat(2000),
    '#forward=' + encodeURIComponent(JSON.stringify({ ...f, step: 23 })),
    '#forward=' + encodeURIComponent(JSON.stringify({ ...f, seed: -1 })),
    '#forward=' + encodeURIComponent(JSON.stringify({ ...f, view: { ...f.view, speed: '1' } })),
  ])
    expect(restoreForwardState(hash).invalid).toBe(true);
  for (const config of [
    { ...a.config, tokens: Infinity },
    { ...a.config, mixers: [] },
    { ...a.config, batch: 99 },
    { ...a.config, bytes: 8 },
  ])
    expect(
      restoreArchitectureState(
        '#architecture=' + encodeURIComponent(JSON.stringify({ ...a, config })),
      ).invalid,
    ).toBe(true);
  expect(restoreArchitectureState('#ssd').invalid).toBe(false);
  expect(restoreArchitectureState('#family-rwkv').invalid).toBe(false);
  expect(restoreArchitectureState('#family-not-a-model').invalid).toBe(true);
});
