import { TOKENS, type MoEConfig, type MoEInput } from '../math/moe';
import { T } from '../i18n';

/** Router → Top-K → FFN → combine → load → expert parallel ledger. */
export const MOE_STEP_COUNT = 6;
export interface MoEState {
  version: 1;
  step: number;
  seed: number;
  config: MoEConfig;
  input: MoEInput;
  view: { explanations: boolean; speed: number };
}

export function initialMoEState(): MoEState {
  return {
    version: 1,
    step: 0,
    seed: 42,
    config: { topK: 2, topology: 'split' },
    input: { preset: 'balanced', token: 0, delta: 0, expertScale: 1 },
    view: { explanations: true, speed: 1 },
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function int(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}
function quarter(value: unknown, min: number, max: number): value is number {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= min &&
    value <= max &&
    Number.isInteger(value * 4)
  );
}

export function validateMoEState(value: unknown): MoEState | null {
  if (
    !record(value) ||
    value.version !== 1 ||
    !record(value.config) ||
    !record(value.input) ||
    !record(value.view)
  )
    return null;
  const { config, input, view } = value;
  if (!int(value.step, 0, MOE_STEP_COUNT - 1) || !int(value.seed, 1, 9999)) return null;
  if (
    (config.topK !== 1 && config.topK !== 2) ||
    (config.topology !== 'single' && config.topology !== 'split')
  )
    return null;
  if (
    (input.preset !== 'balanced' && input.preset !== 'concentrated') ||
    !int(input.token, 0, TOKENS - 1) ||
    !quarter(input.delta, -2, 2) ||
    !quarter(input.expertScale, 0, 2)
  )
    return null;
  if (typeof view.explanations !== 'boolean' || ![0.5, 1, 2].includes(view.speed as number))
    return null;
  // Rebuild all nested objects from a whitelist; private fields cannot enter shared links.
  return {
    version: 1,
    step: value.step,
    seed: value.seed,
    config: { topK: config.topK, topology: config.topology },
    input: {
      preset: input.preset,
      token: input.token,
      delta: input.delta,
      expertScale: input.expertScale,
    },
    view: { explanations: view.explanations, speed: view.speed as number },
  };
}

export function serializeMoEState(state: MoEState): string {
  const clean = validateMoEState(state);
  if (!clean) throw new Error('MoE 实验状态超出允许范围。');
  return '#moe=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreMoEState(hash: string): { state: MoEState; warning: string | null } {
  if (!hash) return { state: initialMoEState(), warning: null };
  try {
    if (hash.length > 2000 || !hash.startsWith('#moe=')) throw new Error('Invalid link');
    const state = validateMoEState(JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    // Malformed or unsupported URLs restore a complete, usable example.
  }
  return { state: initialMoEState(), warning: T('分享链接无效或版本不受支持，已恢复默认示例。') };
}

export function moveMoEStep(state: MoEState, delta: number): MoEState {
  return { ...state, step: Math.max(0, Math.min(MOE_STEP_COUNT - 1, state.step + delta)) };
}
