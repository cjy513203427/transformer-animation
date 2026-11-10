import type { FlashConfig, FlashInput } from '../math/flash';
import { T } from '../i18n';

/** Materialized attention → tiles → online statistics → rescaling → comparison → ledger. */
export const FLASH_STEP_COUNT = 6;
export interface FlashState {
  version: 1;
  step: number;
  seed: number;
  config: FlashConfig;
  input: FlashInput;
  view: { explanations: boolean; speed: number };
}

export function initialFlashState(): FlashState {
  return {
    version: 1,
    step: 0,
    seed: 42,
    config: { blockSize: 2, causal: true },
    input: { preset: 'rising', query: 3, keyDelta: 0, block: 0 },
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

export function validateFlashState(value: unknown): FlashState | null {
  if (
    !record(value) ||
    value.version !== 1 ||
    !record(value.config) ||
    !record(value.input) ||
    !record(value.view)
  )
    return null;
  const { config, input, view } = value;
  if (!int(value.step, 0, FLASH_STEP_COUNT - 1) || !int(value.seed, 1, 9999)) return null;
  if (
    !int(config.blockSize, 1, 4) ||
    typeof config.causal !== 'boolean' ||
    (input.preset !== 'rising' && input.preset !== 'book' && input.preset !== 'mixed') ||
    !int(input.query, 0, 3) ||
    !quarter(input.keyDelta, -2, 2) ||
    !int(input.block, 0, Math.ceil(4 / config.blockSize) - 1)
  )
    return null;
  if (typeof view.explanations !== 'boolean' || ![0.5, 1, 2].includes(view.speed as number))
    return null;
  // Whitelist every nested field: transient predictions and private notes cannot enter a link.
  return {
    version: 1,
    step: value.step,
    seed: value.seed,
    config: { blockSize: config.blockSize as FlashConfig['blockSize'], causal: config.causal },
    input: {
      preset: input.preset,
      query: input.query,
      keyDelta: input.keyDelta,
      block: input.block,
    },
    view: { explanations: view.explanations, speed: view.speed as number },
  };
}

export function serializeFlashState(state: FlashState): string {
  const clean = validateFlashState(state);
  if (!clean) throw new Error('FlashAttention 实验状态超出允许范围。');
  return '#flash=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreFlashState(hash: string): { state: FlashState; warning: string | null } {
  if (!hash) return { state: initialFlashState(), warning: null };
  try {
    if (hash.length > 2000 || !hash.startsWith('#flash=')) throw new Error('Invalid link');
    const state = validateFlashState(JSON.parse(decodeURIComponent(hash.slice(7))));
    if (state) return { state, warning: null };
  } catch {
    // Unsupported or malformed links always restore a usable, deterministic example.
  }
  return { state: initialFlashState(), warning: T('分享链接无效或版本不受支持，已恢复默认示例。') };
}

export function moveFlashStep(state: FlashState, delta: number): FlashState {
  return { ...state, step: Math.max(0, Math.min(FLASH_STEP_COUNT - 1, state.step + delta)) };
}
