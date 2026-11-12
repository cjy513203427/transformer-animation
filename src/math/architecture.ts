import { matmul } from './attention';

export type Mixer = 'attention' | 'ssm';
export interface ArchitectureConfig {
  mixers: Mixer[];
  moe: boolean[];
  tokens: number;
  batch: number;
  bytes: 2 | 4;
}

/** Exact element counts under the displayed, deliberately small architectural assumptions. */
export function architectureLedger(config: ArchitectureConfig) {
  const { mixers, moe, tokens, batch, bytes } = config;
  if (
    !mixers.length ||
    mixers.length !== moe.length ||
    mixers.some((m) => m !== 'attention' && m !== 'ssm') ||
    !Number.isInteger(tokens) ||
    tokens < 1 ||
    !Number.isInteger(batch) ||
    batch < 1 ||
    (bytes !== 2 && bytes !== 4)
  )
    throw new Error('Invalid architecture configuration.');
  const attentionLayers = mixers.filter((m) => m === 'attention').length;
  const ssmLayers = mixers.length - attentionLayers;
  // Attention: 2 Query heads, 2 KV heads, d_k = d_v = 2. SSM: inner width 8, state width 4.
  const kvElements = batch * attentionLayers * tokens * 2 * (2 + 2);
  const ssmElements = batch * ssmLayers * 8 * 4;
  const convElements = batch * ssmLayers * 8 * (4 - 1);
  // Only FFN/MoE sublayers: D = 4, F = 8, E = 4 experts, K = 2 selected experts, no biases.
  const denseParameters = 2 * 4 * 8;
  const routerParameters = 4 * 4;
  const allFfnParameters = moe.reduce(
    (sum, sparse) => sum + (sparse ? routerParameters + 4 * denseParameters : denseParameters),
    0,
  );
  const activeFfnParameters = moe.reduce(
    (sum, sparse) => sum + (sparse ? routerParameters + 2 * denseParameters : denseParameters),
    0,
  );
  return {
    attentionLayers,
    ssmLayers,
    kvElements,
    ssmElements,
    convElements,
    kvBytes: kvElements * bytes,
    ssmBytes: ssmElements * bytes,
    convBytes: convElements * bytes,
    historyBytes: (kvElements + ssmElements + convElements) * bytes,
    allFfnParameters,
    activeFfnParameters,
  };
}

/** Scalar-identity SSM slice of SSD: zero initial state, B = C = 1, shared retention a. */
export function structuredRecurrence(input: readonly number[], retention: number) {
  if (
    !input.length ||
    input.some((x) => !Number.isFinite(x)) ||
    !Number.isFinite(retention) ||
    retention < 0 ||
    retention > 1
  )
    throw new Error('Invalid recurrence input.');
  let state = 0;
  const recurrent = input.map((x) => {
    state = retention * state + x;
    return state;
  });
  const matrix = input.map((_, t) => input.map((_, j) => (j > t ? 0 : retention ** (t - j))));
  const expanded = matmul(
    matrix,
    input.map((x) => [x]),
  ).flat();
  const error = Math.max(...recurrent.map((value, t) => Math.abs(value - expanded[t])));
  return { recurrent, matrix, expanded, error };
}
