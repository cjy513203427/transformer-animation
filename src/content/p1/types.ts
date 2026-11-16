import type { FormulaSymbol } from '../notation/shared';
import type { WhyContent } from '../../components/WhyIntro';
import { formatNumber } from '../../math/attention';
import type { BookLink } from '../book';
import type { Computation, Frame, Parameters } from '../../math/p1/shared';
export interface Field {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  options?: { value: number; label: string }[];
  steps?: number[];
}
/** Values available to an explanation that quotes the current numbers. */
export interface LessonContext {
  input: number[];
  config: Parameters;
  frame: Frame;
  /** A numeric metric of the current frame, looked up by its label. */
  metric: (label: string) => number;
  /** A text metric of the current frame, such as a chosen token. */
  text: (label: string) => string;
  /** A tensor of the current frame, looked up by its name. */
  values: (name: string) => number[];
}
export type LessonText = string | ((context: LessonContext) => string);
export interface Lesson {
  /** Short label for the step tabs. */
  name: string;
  /** Plain-language heading that states what this step answers. */
  title: string;
  formula: string;
  symbols: FormulaSymbol[];
  explanation: LessonText;
  prompt: string;
  takeaway: string;
  /** Book section that explains this step. */
  book: BookLink;
  /** Paper or official documentation used to check the formula, when it differs from the book. */
  source?: BookLink;
}
export interface Example {
  name: string;
  input: number[];
  parameters?: Parameters;
  /** True when the numbers are copied from the book, so the reader can check against it. */
  book?: boolean;
}
export interface TopicDefinition {
  id:
    | 'normalization'
    | 'ffn'
    | 'lora'
    | 'quantization'
    | 'decoding'
    | 'preference'
    | 'memory'
    | 'ssm'
    | 'training'
    | 'policy'
    | 'prompt-chain'
    | 'prompt-vote'
    | 'prompt-search';
  path: string;
  title: string;
  chapter: string;
  intro: string;
  /** The one question the experiment keeps returning to. */
  question: string;
  /** One-line summary of the computation chain. */
  headline: string;
  why: WhyContent;
  introExpanded?: boolean;
  verifiedDate?: string;
  input: number[];
  /** Allowed range of each input value; defaults to [−8, 8]. */
  inputRange?: [number, number];
  /** Visible names of the input values, e.g. “d0” or “回答 0”. */
  inputNames?: string[];
  fields: Field[];
  steps: Lesson[];
  examples: Example[];
  compute: (input: number[], parameters: Parameters, seed: number) => Computation;
  prediction: { key: string; target: number; question: string; label: string };
  sources: { title: string; url: string }[];
  limits: string[];
  /** Where the page deliberately differs from the book's wording, with the reason. */
  bookNotes?: string[];
}
export const DEFAULT_INPUT_RANGE: [number, number] = [-8, 8];
export const inputRange = (topic: TopicDefinition) => topic.inputRange ?? DEFAULT_INPUT_RANGE;
export const field = (
  key: string,
  label: string,
  value: number,
  min: number,
  max: number,
  step: number,
  steps?: number[],
): Field => ({ key, label, value, min, max, step, steps });
export const choice = (
  key: string,
  label: string,
  value: number,
  options: { value: number; label: string }[],
  steps?: number[],
): Field => ({
  key,
  label,
  value,
  min: Math.min(...options.map((v) => v.value)),
  max: Math.max(...options.map((v) => v.value)),
  step: 0,
  options,
  steps,
});
export const lesson = (definition: Lesson): Lesson => definition;
export function lessonContext(input: number[], config: Parameters, frame: Frame): LessonContext {
  return {
    input,
    config,
    frame,
    metric(label) {
      const found = frame.metrics.find(([name]) => name === label);
      if (!found || typeof found[1] !== 'number') throw new Error(`缺少数值指标：${label}`);
      return found[1];
    },
    text(label) {
      const found = frame.metrics.find(([name]) => name === label);
      if (!found) throw new Error(`缺少指标：${label}`);
      return String(found[1]);
    },
    values(name) {
      const found = frame.tensors.find((t) => t.name === name);
      if (!found) throw new Error(`缺少张量：${name}`);
      return found.values.flat();
    },
  };
}
export const explain = (text: LessonText, context: LessonContext) =>
  typeof text === 'string' ? text : text(context);
/** “[1, -0.5]” with the display rounding used everywhere else. */
export const vec = (values: number[]) => '[' + values.map((v) => formatNumber(v)).join(', ') + ']';
export const verifiedDate = '2026-11-08';
