import type { Matrix } from '../math/attention';
import { HEAD_OPTIONS, type HeadCount } from '../math/multihead';
import { ARCHITECTURE_NODES, LESSON_MODES, type ArchitectureNode, type LessonMode } from './lesson';

export const STEP_COUNT = 5;
/** 投影 → 切分多头 → 矩阵变形 → 每头 Attention → Concatenate → Wo */
export const HEAD_STEP_COUNT = 6;
export type Preset = 'basic' | 'value' | 'causal' | 'uniform' | 'bidirectional';

/** The multi-head lesson keeps its own timeline, so it never mixes with the single-head steps. */
export interface HeadsState {
  step: number;
  numHeads: HeadCount;
  query: number;
  /** Executed activation intervention K[t0, d3] += keyDelta (the draft slider is not shared). */
  keyDelta: number;
}

export interface ExperimentState {
  version: 3;
  config: { causal: boolean; scaled: boolean };
  input: { x: Matrix; keyDelta: number; valueDelta: number };
  step: number;
  seed: number;
  view: {
    query: number;
    key: number;
    explanations: boolean;
    speed: number;
    lesson: LessonMode;
    architectureNode: ArchitectureNode;
  };
  heads: HeadsState;
}

function initialHeads(): HeadsState {
  return { step: 0, numHeads: 2, query: 2, keyDelta: 0 };
}

export function initialState(): ExperimentState {
  return {
    version: 3,
    config: { causal: true, scaled: true },
    input: {
      x: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      keyDelta: 0,
      valueDelta: 0,
    },
    step: 0,
    seed: 42,
    view: {
      query: 1,
      key: 0,
      explanations: true,
      speed: 1,
      lesson: 'overview',
      architectureNode: 'attention',
    },
    heads: initialHeads(),
  };
}

export function presetState(preset: Preset): ExperimentState {
  const state = initialState();
  state.view.lesson = 'numbers';
  if (preset === 'value') {
    state.input.valueDelta = 2;
    state.step = 4;
  }
  if (preset === 'causal') {
    state.config.causal = true;
    state.view.query = 1;
    state.step = 2;
  }
  if (preset === 'uniform') {
    state.config.causal = false;
    state.input.x = [
      [0, 0],
      [0, 0],
      [0, 0],
    ];
    state.step = 3;
  }
  if (preset === 'bidirectional') {
    state.config.causal = false;
    state.step = 2;
  }
  return state;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function bounded(value: unknown, min: number, max: number, integer = false): value is number {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= min &&
    value <= max &&
    (!integer || Number.isInteger(value))
  );
}

function validateHeads(value: unknown): HeadsState | null {
  if (!record(value)) return null;
  const { step, numHeads, query, keyDelta } = value;
  if (
    !bounded(step, 0, HEAD_STEP_COUNT - 1, true) ||
    !HEAD_OPTIONS.some((option) => option === numHeads) ||
    !bounded(query, 0, 2, true) ||
    !bounded(keyDelta, -4, 4)
  )
    return null;
  return { step, numHeads: numHeads as HeadCount, query, keyDelta };
}

export function validateState(value: unknown): ExperimentState | null {
  if (
    !record(value) ||
    (value.version !== 1 && value.version !== 2 && value.version !== 3) ||
    !record(value.config) ||
    !record(value.input) ||
    !record(value.view)
  )
    return null;
  const { config, input, view } = value;
  const lesson = value.version === 1 ? 'numbers' : view.lesson;
  const architectureNode = value.version === 1 ? 'attention' : view.architectureNode;
  if (
    !LESSON_MODES.some((mode) => mode === lesson) ||
    !ARCHITECTURE_NODES.some((node) => node === architectureNode)
  )
    return null;
  // Links created before the multi-head lesson existed start it from its default example.
  const heads = value.version === 3 ? validateHeads(value.heads) : initialHeads();
  if (!heads) return null;
  if (typeof config.causal !== 'boolean' || typeof config.scaled !== 'boolean') return null;
  if (!bounded(input.keyDelta, -4, 4) || !bounded(input.valueDelta, -4, 4)) return null;
  if (
    !Array.isArray(input.x) ||
    input.x.length !== 3 ||
    !input.x.every(
      (row) => Array.isArray(row) && row.length === 2 && row.every((x) => bounded(x, -4, 4)),
    )
  )
    return null;
  if (!bounded(value.step, 0, STEP_COUNT - 1, true) || !bounded(value.seed, 0, 999999, true))
    return null;
  if (
    !bounded(view.query, 0, 2, true) ||
    !bounded(view.key, 0, 2, true) ||
    typeof view.explanations !== 'boolean' ||
    !bounded(view.speed, 0.5, 2) ||
    ![0.5, 1, 2].includes(view.speed)
  )
    return null;
  // Reconstruct a whitelist: unrelated/private fields are never serialized.
  return {
    version: 3,
    config: { causal: config.causal, scaled: config.scaled },
    input: {
      x: input.x.map((row) => [...row]),
      keyDelta: input.keyDelta,
      valueDelta: input.valueDelta,
    },
    step: value.step,
    seed: value.seed,
    view: {
      query: view.query,
      key: view.key,
      explanations: view.explanations,
      speed: view.speed,
      lesson: lesson as LessonMode,
      architectureNode: architectureNode as ArchitectureNode,
    },
    heads,
  };
}

export function serializeState(state: ExperimentState): string {
  const clean = validateState(state);
  if (!clean) throw new Error('实验状态超出允许范围。');
  return '#lab=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreState(hash: string): { state: ExperimentState; warning: string | null } {
  if (!hash) return { state: initialState(), warning: null };
  try {
    if (hash.length > 6000 || !hash.startsWith('#lab=')) throw new Error('Invalid link');
    const state = validateState(JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    /* Invalid links always fall back to a safe, complete example. */
  }
  return { state: initialState(), warning: '分享链接无效或版本不受支持，已恢复基础示例。' };
}

export function moveStep(state: ExperimentState, delta: number): ExperimentState {
  return { ...state, step: Math.max(0, Math.min(STEP_COUNT - 1, state.step + delta)) };
}

export function moveHeadStep(state: ExperimentState, delta: number): ExperimentState {
  const step = Math.max(0, Math.min(HEAD_STEP_COUNT - 1, state.heads.step + delta));
  return { ...state, heads: { ...state.heads, step } };
}
