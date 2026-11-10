import { T } from '../i18n';
/** 位置盲 → 正弦表 → 相加 → RoPE 旋转 → RoPE 相对位置 → ALiBi → 对照 */
export const POSITION_STEP_COUNT = 7;
export const D_MODEL_OPTIONS = [4, 64, 512] as const;
export type DModel = (typeof D_MODEL_OPTIONS)[number];
export const MAX_POS = 63;
export const ROPE_MAX = 40;
export const ALIBI_TOKENS = 6;
export const ALIBI_HEADS = 8;

export interface PositionState {
  version: 1;
  step: number;
  view: { explanations: boolean; speed: number };
  blind: { withPE: boolean; pair: 'book' | 'causal' };
  sin: { dModel: DModel; pos: number; pair: number };
  rope: { m: number; n: number };
  alibi: { head: number; query: number; content: 'same' | 'example' };
}

export function initialPositionState(): PositionState {
  return {
    version: 1,
    step: 0,
    view: { explanations: true, speed: 1 },
    blind: { withPE: false, pair: 'book' },
    // Pair 8 of d_model = 64 turns 0.1 rad per position: one smooth cycle over 0–63.
    sin: { dModel: 64, pos: 1, pair: 8 },
    rope: { m: 2, n: 5 },
    alibi: { head: 0, query: ALIBI_TOKENS - 1, content: 'same' },
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function int(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

export function validatePositionState(value: unknown): PositionState | null {
  if (
    !record(value) ||
    value.version !== 1 ||
    !record(value.view) ||
    !record(value.blind) ||
    !record(value.sin) ||
    !record(value.rope) ||
    !record(value.alibi)
  )
    return null;
  const { view, blind, sin, rope, alibi } = value;
  if (!int(value.step, 0, POSITION_STEP_COUNT - 1)) return null;
  if (typeof view.explanations !== 'boolean' || ![0.5, 1, 2].includes(view.speed as number))
    return null;
  if (typeof blind.withPE !== 'boolean' || (blind.pair !== 'book' && blind.pair !== 'causal'))
    return null;
  if (!D_MODEL_OPTIONS.some((option) => option === sin.dModel)) return null;
  const dModel = sin.dModel as DModel;
  if (!int(sin.pos, 0, MAX_POS) || !int(sin.pair, 0, dModel / 2 - 1)) return null;
  if (!int(rope.m, 0, ROPE_MAX) || !int(rope.n, 0, ROPE_MAX)) return null;
  if (
    !int(alibi.head, 0, ALIBI_HEADS - 1) ||
    !int(alibi.query, 0, ALIBI_TOKENS - 1) ||
    (alibi.content !== 'same' && alibi.content !== 'example')
  )
    return null;
  // Rebuild from a whitelist so unrelated or private fields never enter a shared link.
  return {
    version: 1,
    step: value.step,
    view: { explanations: view.explanations, speed: view.speed as number },
    blind: { withPE: blind.withPE, pair: blind.pair },
    sin: { dModel, pos: sin.pos, pair: sin.pair },
    rope: { m: rope.m, n: rope.n },
    alibi: { head: alibi.head, query: alibi.query, content: alibi.content },
  };
}

export function serializePositionState(state: PositionState): string {
  const clean = validatePositionState(state);
  if (!clean) throw new Error('位置编码实验状态超出允许范围。');
  return '#pos=' + encodeURIComponent(JSON.stringify(clean));
}

export function restorePositionState(hash: string): {
  state: PositionState;
  warning: string | null;
} {
  if (!hash) return { state: initialPositionState(), warning: null };
  try {
    if (hash.length > 3000 || !hash.startsWith('#pos=')) throw new Error('Invalid link');
    const state = validatePositionState(JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    /* Invalid links fall back to the complete default example. */
  }
  return {
    state: initialPositionState(),
    warning: T('分享链接无效或版本不受支持，已恢复默认示例。'),
  };
}

export function movePositionStep(state: PositionState, delta: number): PositionState {
  return {
    ...state,
    step: Math.max(0, Math.min(POSITION_STEP_COUNT - 1, state.step + delta)),
  };
}
