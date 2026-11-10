import { MAX_DECODE_STEPS, VOCAB } from '../math/inference';
import { T } from '../i18n';

/** 训练并行 → 无缓存基线 → Prefill → Decode 一步 → 缓存对照重算 → 缓存何时失效 */
export const INFERENCE_STEP_COUNT = 6;
/** The prompt used by every generation scene: a b c. */
export const PROMPT = [0, 1, 2];
/** The training sample: inputs a b c d, targets b c d e. */
export const TRAIN_SEQUENCE = [0, 1, 2, 3, 4];

export interface InferenceState {
  version: 1;
  step: number;
  view: { explanations: boolean; speed: number };
  /** Training counterfactual: replace input x[pos] with `token`, targets unchanged. */
  train: { pos: number; token: number; applied: boolean };
  /** Number of Decode steps shown (each step feeds one token and appends one cache entry). */
  decode: { steps: number };
  /** Cache scene: replace prompt token `pos`, or slide the window by one token. */
  cache: { mode: 'edit' | 'slide'; pos: number; token: number };
}

export function initialInferenceState(): InferenceState {
  return {
    version: 1,
    step: 0,
    view: { explanations: true, speed: 1 },
    train: { pos: 2, token: 5, applied: false },
    decode: { steps: 3 },
    cache: { mode: 'edit', pos: 1, token: 5 },
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function int(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

export function validateInferenceState(value: unknown): InferenceState | null {
  if (
    !record(value) ||
    value.version !== 1 ||
    !record(value.view) ||
    !record(value.train) ||
    !record(value.decode) ||
    !record(value.cache)
  )
    return null;
  const { view, train, decode, cache } = value;
  const lastToken = VOCAB.length - 1;
  if (!int(value.step, 0, INFERENCE_STEP_COUNT - 1)) return null;
  if (typeof view.explanations !== 'boolean' || ![0.5, 1, 2].includes(view.speed as number))
    return null;
  if (
    !int(train.pos, 0, TRAIN_SEQUENCE.length - 2) ||
    !int(train.token, 0, lastToken) ||
    typeof train.applied !== 'boolean'
  )
    return null;
  if (!int(decode.steps, 1, MAX_DECODE_STEPS)) return null;
  if (
    (cache.mode !== 'edit' && cache.mode !== 'slide') ||
    !int(cache.pos, 0, PROMPT.length - 1) ||
    !int(cache.token, 0, lastToken)
  )
    return null;
  // Rebuild from a whitelist so unrelated fields never travel inside a shared link.
  return {
    version: 1,
    step: value.step,
    view: { explanations: view.explanations, speed: view.speed as number },
    train: { pos: train.pos, token: train.token, applied: train.applied },
    decode: { steps: decode.steps },
    cache: { mode: cache.mode, pos: cache.pos, token: cache.token },
  };
}

export function serializeInferenceState(state: InferenceState): string {
  const clean = validateInferenceState(state);
  if (!clean) throw new Error('训练与缓存实验状态超出允许范围。');
  return '#inf=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreInferenceState(hash: string): {
  state: InferenceState;
  warning: string | null;
} {
  if (!hash) return { state: initialInferenceState(), warning: null };
  try {
    if (hash.length > 2000 || !hash.startsWith('#inf=')) throw new Error('Invalid link');
    const state = validateInferenceState(JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    /* Invalid links fall back to the complete default example. */
  }
  return {
    state: initialInferenceState(),
    warning: T('分享链接无效或版本不受支持，已恢复默认示例。'),
  };
}

export function moveInferenceStep(state: InferenceState, delta: number): InferenceState {
  return {
    ...state,
    step: Math.max(0, Math.min(INFERENCE_STEP_COUNT - 1, state.step + delta)),
  };
}
