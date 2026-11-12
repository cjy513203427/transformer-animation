import { forwardStepCount } from '../math/forward';

export interface ForwardState {
  version: 1;
  config: { depth: 1 | 2; causal: boolean };
  input: { tokens: number[] };
  step: number;
  seed: number;
  view: { query: number; explanations: boolean; speed: number };
}
export const initialForwardState = (): ForwardState => ({
  version: 1,
  config: { depth: 2, causal: true },
  input: { tokens: [0, 1, 2] },
  step: 0,
  seed: 42,
  view: { query: 2, explanations: true, speed: 1 },
});
const record = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === 'object' && !Array.isArray(x);
const integer = (x: unknown, min: number, max: number): x is number =>
  typeof x === 'number' && Number.isInteger(x) && x >= min && x <= max;

export const serializeForwardState = (state: ForwardState): string =>
  '#forward=' + encodeURIComponent(JSON.stringify(state));
export function restoreForwardState(hash: string): { state: ForwardState; invalid: boolean } {
  const fallback = { state: initialForwardState(), invalid: !!hash && hash !== '#scope' };
  if (!hash) return fallback;
  if (!hash.startsWith('#forward=') || hash.length > 1800) return fallback;
  try {
    const x: unknown = JSON.parse(decodeURIComponent(hash.slice(9)));
    if (!record(x) || x.version !== 1 || !record(x.config) || !record(x.input) || !record(x.view))
      return fallback;
    if (
      (x.config.depth !== 1 && x.config.depth !== 2) ||
      typeof x.config.causal !== 'boolean' ||
      !Array.isArray(x.input.tokens) ||
      x.input.tokens.length !== 3 ||
      x.input.tokens.some((t) => !integer(t, 0, 3)) ||
      !integer(x.seed, 0, 0xffffffff) ||
      !integer(x.step, 0, forwardStepCount(x.config.depth) - 1) ||
      !integer(x.view.query, 0, 2) ||
      typeof x.view.explanations !== 'boolean' ||
      ![0.5, 1, 2].includes(Number(x.view.speed)) ||
      typeof x.view.speed !== 'number'
    )
      return fallback;
    // Reconstruct only the public schema; ignore unrecognized fields.
    return {
      invalid: false,
      state: {
        version: 1,
        config: { depth: x.config.depth, causal: x.config.causal },
        input: { tokens: [...x.input.tokens] },
        seed: x.seed,
        step: x.step,
        view: { query: x.view.query, explanations: x.view.explanations, speed: x.view.speed },
      },
    };
  } catch {
    return fallback;
  }
}
