import { describe, expect, it } from 'vitest';
import { tokenizationSteps } from '../content/tokenization';
import {
  TOKENIZATION_STEPS,
  initialTokenizationState,
  restoreTokenizationState,
  serializeTokenizationState,
} from './tokenization';

describe('BPE share state', () => {
  it('restores the text, seed, corpus, stage and round', () => {
    const state = initialTokenizationState();
    state.input.text = '猫 😺\n';
    state.seed = 17;
    state.step = 2;
    state.view.round = 3;
    state.view.explanations = false;
    state.view.speed = 2;
    expect(TOKENIZATION_STEPS).toBe(tokenizationSteps.length);
    expect(restoreTokenizationState(serializeTokenizationState(state))).toEqual({
      state,
      invalid: false,
    });
  });

  it('rejects malformed text or bounds and filters extra fields', () => {
    const state = initialTokenizationState();
    const raw = (value: unknown) => '#bpe=' + encodeURIComponent(JSON.stringify(value));
    expect(
      restoreTokenizationState(raw({ ...state, input: { text: state.input.text, note: 'secret' } }))
        .state,
    ).toEqual(state);
    for (const hash of [
      '#bpe=%ZZ',
      '#bpe=' + 'A'.repeat(1700),
      raw({ ...state, version: 2 }),
      raw({ ...state, input: { text: '\uD800' } }),
      raw({ ...state, view: { ...state.view, round: 9 } }),
      raw({ ...state, config: { ...state.config, merges: -1 } }),
    ])
      expect(restoreTokenizationState(hash)).toEqual({ state, invalid: true });
  });
});
