import { KV_OPTIONS, Q_HEADS, TOKENS, type KVHeads } from '../math/kv';

/** 缓存账本 → 共享 KV 头 → MLA 压缩 → 解耦 RoPE → 矩阵吸收 → 对照 */
export const KV_STEP_COUNT = 6;
/** Bytes per element: FP32, FP16/BF16, FP8. */
export const BYTE_OPTIONS = [4, 2, 1] as const;
export const COMPARE_TOKENS = [1024, 4096, 32768, 131072] as const;

export interface CalcConfig {
  batch: number;
  layers: number;
  tokens: number;
  qHeads: number;
  kvHeads: number;
  dHead: number;
  bytes: (typeof BYTE_OPTIONS)[number];
}

/** Configurations whose numbers the book works through (chapters 22 and 23). */
export const CALC_PRESETS: Record<'book23' | 'llama7b', { label: string; config: CalcConfig }> = {
  book23: {
    label: '原书第 23 章：32 层、1024 token',
    config: { batch: 1, layers: 32, tokens: 1024, qHeads: 32, kvHeads: 32, dHead: 128, bytes: 2 },
  },
  llama7b: {
    label: '原书第 22 章：Llama 2 7B、4096 token',
    config: { batch: 1, layers: 32, tokens: 4096, qHeads: 32, kvHeads: 32, dHead: 128, bytes: 2 },
  },
};

export const LIMITS = {
  batch: 256,
  layers: 256,
  tokens: 1_048_576,
  qHeads: 256,
  dHead: 1024,
} as const;

export interface KVState {
  version: 1;
  step: number;
  view: { explanations: boolean; speed: number };
  calc: CalcConfig;
  group: { kvHeads: KVHeads; head: number; query: number };
  mla: { head: number; query: number; coupled: boolean };
  compare: { tokens: (typeof COMPARE_TOKENS)[number] };
}

export function initialKVState(): KVState {
  return {
    version: 1,
    step: 0,
    view: { explanations: true, speed: 1 },
    calc: { ...CALC_PRESETS.book23.config },
    group: { kvHeads: 2, head: 1, query: TOKENS - 1 },
    mla: { head: 0, query: TOKENS - 1, coupled: true },
    compare: { tokens: 4096 },
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function int(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

export function validCalc(value: unknown): CalcConfig | null {
  if (!record(value)) return null;
  const { batch, layers, tokens, qHeads, kvHeads, dHead, bytes } = value;
  if (
    !int(batch, 1, LIMITS.batch) ||
    !int(layers, 1, LIMITS.layers) ||
    !int(tokens, 1, LIMITS.tokens) ||
    !int(qHeads, 1, LIMITS.qHeads) ||
    !int(kvHeads, 1, qHeads) ||
    qHeads % kvHeads !== 0 ||
    !int(dHead, 1, LIMITS.dHead) ||
    !BYTE_OPTIONS.some((option) => option === bytes)
  )
    return null;
  return { batch, layers, tokens, qHeads, kvHeads, dHead, bytes: bytes as CalcConfig['bytes'] };
}

export function validateKVState(value: unknown): KVState | null {
  if (
    !record(value) ||
    value.version !== 1 ||
    !record(value.view) ||
    !record(value.group) ||
    !record(value.mla) ||
    !record(value.compare)
  )
    return null;
  const { view, group, mla, compare } = value;
  const calc = validCalc(value.calc);
  if (!calc || !int(value.step, 0, KV_STEP_COUNT - 1)) return null;
  if (typeof view.explanations !== 'boolean' || ![0.5, 1, 2].includes(view.speed as number))
    return null;
  if (
    !KV_OPTIONS.some((option) => option === group.kvHeads) ||
    !int(group.head, 0, Q_HEADS - 1) ||
    !int(group.query, 0, TOKENS - 1)
  )
    return null;
  if (!int(mla.head, 0, Q_HEADS - 1) || !int(mla.query, 0, TOKENS - 1)) return null;
  if (typeof mla.coupled !== 'boolean') return null;
  if (!COMPARE_TOKENS.some((option) => option === compare.tokens)) return null;
  // Rebuild from a whitelist so unrelated fields never travel inside a shared link.
  return {
    version: 1,
    step: value.step,
    view: { explanations: view.explanations, speed: view.speed as number },
    calc,
    group: { kvHeads: group.kvHeads as KVHeads, head: group.head, query: group.query },
    mla: { head: mla.head, query: mla.query, coupled: mla.coupled },
    compare: { tokens: compare.tokens as KVState['compare']['tokens'] },
  };
}

export function serializeKVState(state: KVState): string {
  const clean = validateKVState(state);
  if (!clean) throw new Error('KV 架构实验状态超出允许范围。');
  return '#kv=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreKVState(hash: string): { state: KVState; warning: string | null } {
  if (!hash) return { state: initialKVState(), warning: null };
  try {
    if (hash.length > 2000 || !hash.startsWith('#kv=')) throw new Error('Invalid link');
    const state = validateKVState(JSON.parse(decodeURIComponent(hash.slice(4))));
    if (state) return { state, warning: null };
  } catch {
    /* Invalid links fall back to the complete default example. */
  }
  return { state: initialKVState(), warning: '分享链接无效或版本不受支持，已恢复默认示例。' };
}

export function moveKVStep(state: KVState, delta: number): KVState {
  return { ...state, step: Math.max(0, Math.min(KV_STEP_COUNT - 1, state.step + delta)) };
}

/** KV head counts that divide the query head count, largest first (MHA … MQA). */
export function kvHeadChoices(qHeads: number): number[] {
  return Array.from({ length: qHeads }, (_, i) => qHeads - i).filter((g) => qHeads % g === 0);
}
