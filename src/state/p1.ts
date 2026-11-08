import { inputRange, type TopicDefinition } from '../content/p1/types';
import type { Parameters } from '../math/p1/shared';
export interface LabState {
  version: 1;
  topic: string;
  step: number;
  seed: number;
  input: number[];
  config: Parameters;
  view: { explanations: boolean; speed: number };
}
export function initialLabState(topic: TopicDefinition): LabState {
  return {
    version: 1,
    topic: topic.id,
    step: 0,
    seed: 42,
    input: topic.input.slice(),
    config: Object.fromEntries(topic.fields.map((f) => [f.key, f.value])),
    view: { explanations: true, speed: 1 },
  };
}
const record = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown, min: number, max: number): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max;
export function validateLabState(topic: TopicDefinition, value: unknown): LabState | null {
  const [low, high] = inputRange(topic);
  if (
    !record(value) ||
    value.version !== 1 ||
    value.topic !== topic.id ||
    !finite(value.step, 0, topic.steps.length - 1) ||
    !Number.isInteger(value.step) ||
    !finite(value.seed, 1, 9999) ||
    !Number.isInteger(value.seed) ||
    !Array.isArray(value.input) ||
    value.input.length !== topic.input.length ||
    !value.input.every((v) => finite(v, low, high)) ||
    !record(value.config) ||
    !record(value.view) ||
    typeof value.view.explanations !== 'boolean' ||
    ![0.5, 1, 2].includes(value.view.speed as number)
  )
    return null;
  const config: Parameters = {};
  for (const field of topic.fields) {
    const v = value.config[field.key];
    if (
      !finite(v, field.min, field.max) ||
      (field.options && !field.options.some((o) => o.value === v)) ||
      (!field.options &&
        Math.abs((v - field.min) / field.step - Math.round((v - field.min) / field.step)) > 1e-8)
    )
      return null;
    config[field.key] = v;
  }
  return {
    version: 1,
    topic: topic.id,
    step: value.step,
    seed: value.seed,
    input: value.input.slice() as number[],
    config,
    view: { explanations: value.view.explanations, speed: value.view.speed as number },
  };
}
export function serializeLabState(topic: TopicDefinition, state: LabState) {
  const clean = validateLabState(topic, state);
  if (!clean) throw new Error('实验状态超出允许范围。');
  return '#lab=' + encodeURIComponent(JSON.stringify(clean));
}
export function restoreLabState(
  topic: TopicDefinition,
  hash: string,
): { state: LabState; warning: string | null } {
  if (!hash) return { state: initialLabState(topic), warning: null };
  try {
    if (hash.length > 4000 || !hash.startsWith('#lab=')) throw new Error('Invalid link');
    const state = validateLabState(topic, JSON.parse(decodeURIComponent(hash.slice(5))));
    if (state) return { state, warning: null };
  } catch {
    /* Restore a complete example. */
  }
  return { state: initialLabState(topic), warning: '分享链接无效或版本不受支持，已恢复默认示例。' };
}
