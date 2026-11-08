import type { Computation, Parameters } from '../../math/p1/shared';
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
export interface Lesson {
  name: string;
  formula: string;
  explanation: string;
  prompt: string;
  takeaway: string;
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
    | 'ssm';
  path: string;
  title: string;
  chapter: string;
  intro: string;
  input: number[];
  fields: Field[];
  steps: Lesson[];
  examples: { name: string; input: number[]; parameters?: Parameters }[];
  compute: (input: number[], parameters: Parameters, seed: number) => Computation;
  prediction: { key: string; target: number; question: string; label: string };
  sources: { title: string; url: string }[];
  limits: string[];
}
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
export const lesson = (
  name: string,
  formula: string,
  explanation: string,
  prompt: string,
  takeaway: string,
): Lesson => ({ name, formula, explanation, prompt, takeaway });
export const verifiedDate = '2026-11-08';
