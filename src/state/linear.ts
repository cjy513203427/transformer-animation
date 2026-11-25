import { LINEAR_PRESETS, type Mat2, type Vec2 } from '../math/linear';
import { T } from '../i18n';

export const LINEAR_STEP_COUNT = 5;

export interface LinearState {
  version: 1;
  step: number;
  w: Mat2;
  x: Vec2;
  k: Vec2;
  view: { explanations: boolean; speed: number };
}

export function initialLinearState(): LinearState {
  const p = LINEAR_PRESETS.shear;
  return {
    version: 1,
    step: 0,
    w: [
      [...p.w[0]],
      [...p.w[1]],
    ],
    x: [...p.x],
    k: [...p.k],
    view: { explanations: true, speed: 1 },
  };
}

const record = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const int = (v: unknown, min: number, max: number): v is number =>
  typeof v === 'number' && Number.isInteger(v) && v >= min && v <= max;
const quarter = (v: unknown, min = -2, max = 2): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max && Number.isInteger(v * 4);
const vec2 = (v: unknown): v is Vec2 =>
  Array.isArray(v) && v.length === 2 && quarter(v[0]) && quarter(v[1]);
const mat2 = (v: unknown): v is Mat2 =>
  Array.isArray(v) && v.length === 2 && vec2(v[0]) && vec2(v[1]);

export function validateLinearState(value: unknown): LinearState | null {
  if (!record(value) || value.version !== 1 || !record(value.view)) return null;
  if (!int(value.step, 0, LINEAR_STEP_COUNT - 1)) return null;
  if (!mat2(value.w) || !vec2(value.x) || !vec2(value.k)) return null;
  if (
    typeof value.view.explanations !== 'boolean' ||
    ![0.5, 1, 2].includes(value.view.speed as number)
  )
    return null;
  return {
    version: 1,
    step: value.step,
    w: [
      [value.w[0][0], value.w[0][1]],
      [value.w[1][0], value.w[1][1]],
    ],
    x: [value.x[0], value.x[1]],
    k: [value.k[0], value.k[1]],
    view: { explanations: value.view.explanations, speed: value.view.speed as number },
  };
}

export function serializeLinearState(state: LinearState): string {
  const clean = validateLinearState(state);
  if (!clean) throw new Error('线性变换实验状态超出允许范围。');
  return '#lin=' + encodeURIComponent(JSON.stringify(clean));
}

export function restoreLinearState(hash: string): { state: LinearState; warning: string | null } {
  if (!hash) return { state: initialLinearState(), warning: null };
  try {
    if (hash.length > 2000 || !hash.startsWith('#lin=')) throw new Error('Invalid link');
    const state = validateLinearState(JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    // Fallback to default example on invalid link.
  }
  return {
    state: initialLinearState(),
    warning: T('分享链接无效或版本不受支持，已恢复默认示例。'),
  };
}

export function moveLinearStep(state: LinearState, delta: number): LinearState {
  return { ...state, step: Math.max(0, Math.min(LINEAR_STEP_COUNT - 1, state.step + delta)) };
}
