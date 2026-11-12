import type { ArchitectureConfig } from '../math/architecture';

export interface ArchitectureState {
  version: 1;
  config: ArchitectureConfig;
  input: { signal: number[]; retention: number };
  seed: 0;
  view: { explanations: boolean; node: number };
}
export const initialArchitectureState = (): ArchitectureState => ({
  version: 1,
  config: {
    mixers: ['attention', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm'],
    moe: [false, true, false, true, false, true, false, true],
    tokens: 1024,
    batch: 1,
    bytes: 2,
  },
  input: { signal: [1, 0, 2, -1], retention: 0.5 },
  seed: 0,
  view: { explanations: true, node: 0 },
});
const record = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === 'object' && !Array.isArray(x);
const finite = (x: unknown, min: number, max: number): x is number =>
  typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max;
export const serializeArchitectureState = (state: ArchitectureState): string =>
  '#architecture=' + encodeURIComponent(JSON.stringify(state));
export const isArchitectureAnchor = (hash: string): boolean =>
  /^#(overview|mamba|ssd|hybrid|alternatives|review|family-(attention|s4|mamba|ssd|rwkv|retnet|hyena|hybrid))$/.test(
    hash,
  );
export function restoreArchitectureState(hash: string): {
  state: ArchitectureState;
  invalid: boolean;
} {
  const fallback = {
    state: initialArchitectureState(),
    invalid: !!hash && !isArchitectureAnchor(hash),
  };
  if (!hash || !hash.startsWith('#architecture=') || hash.length > 2400) return fallback;
  try {
    const x: unknown = JSON.parse(decodeURIComponent(hash.slice(14)));
    if (
      !record(x) ||
      x.version !== 1 ||
      x.seed !== 0 ||
      !record(x.config) ||
      !record(x.input) ||
      !record(x.view)
    )
      return fallback;
    const c = x.config,
      input = x.input;
    if (
      !Array.isArray(c.mixers) ||
      c.mixers.length !== 8 ||
      c.mixers.some((m) => m !== 'attention' && m !== 'ssm') ||
      !Array.isArray(c.moe) ||
      c.moe.length !== 8 ||
      c.moe.some((v) => typeof v !== 'boolean') ||
      !finite(c.tokens, 1, 131072) ||
      !Number.isInteger(c.tokens) ||
      !finite(c.batch, 1, 8) ||
      !Number.isInteger(c.batch) ||
      (c.bytes !== 2 && c.bytes !== 4) ||
      !Array.isArray(input.signal) ||
      input.signal.length !== 4 ||
      input.signal.some((v) => !finite(v, -4, 4)) ||
      !finite(input.retention, 0, 1) ||
      typeof x.view.explanations !== 'boolean' ||
      !finite(x.view.node, 0, 6) ||
      !Number.isInteger(x.view.node)
    )
      return { ...fallback, invalid: true };
    return {
      invalid: false,
      state: {
        version: 1,
        seed: 0,
        config: {
          mixers: [...c.mixers],
          moe: [...c.moe],
          tokens: c.tokens,
          batch: c.batch,
          bytes: c.bytes,
        },
        input: { signal: [...input.signal], retention: input.retention },
        view: { explanations: x.view.explanations, node: x.view.node },
      },
    };
  } catch {
    return { ...fallback, invalid: true };
  }
}
