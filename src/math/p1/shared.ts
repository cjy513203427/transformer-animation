import { softmax, type Matrix } from '../attention';
export type Parameters = Record<string, number>;
export interface Tensor {
  name: string;
  values: Matrix;
  columns?: string[];
  rows?: string[];
}
export interface Frame {
  title: string;
  tensors: Tensor[];
  metrics: [string, number | string][];
  note?: string;
  points?: { before: Matrix; after: Matrix };
  connections?: Matrix;
  tree?: { root: number[]; conditional: Matrix; joint: Matrix; selected: string };
  scan?: {
    input: number[];
    states: number[];
    retention: number[];
    drive: number[];
    cursor: number;
    initial: number;
  };
  branch?: { base: Matrix; delta: Matrix; total: Matrix };
}
export interface Check {
  label: string;
  error: number;
  tolerance: number;
}
export interface Computation {
  frames: Frame[];
  checks: Check[];
  signature: number[];
}
export const tensor = (name: string, values: number[] | Matrix): Tensor => ({
  name,
  values: typeof values[0] === 'number' ? [values as number[]] : (values as Matrix),
});
export const frame = (
  title: string,
  tensors: Tensor[],
  metrics: Frame['metrics'] = [],
  note?: string,
): Frame => ({ title, tensors, metrics, note });
export const sum = (x: number[]) => x.reduce((a, b) => a + b, 0);
export const mean = (x: number[]) => sum(x) / x.length;
export const dot = (a: number[], b: number[]) => sum(a.map((x, i) => x * b[i]));
export const error = (a: number[], b: number[]) =>
  Math.max(0, ...a.map((x, i) => Math.abs(x - b[i])));
export const check = (label: string, a: number[], b: number[], tolerance = 1e-10): Check => ({
  label,
  error: error(a, b),
  tolerance,
});
export function randomValues(seed: number, count: number) {
  let n = seed >>> 0;
  return Array.from({ length: count }, () => {
    n = (Math.imul(n, 1664525) + 1013904223) >>> 0;
    return ((n % 17) - 8) / 4;
  });
}
export function uniforms(seed: number, count: number) {
  let n = seed >>> 0;
  return Array.from({ length: count }, () => {
    n = (Math.imul(n, 1664525) + 1013904223) >>> 0;
    return n / 4294967296;
  });
}
export const pairRows = (x: number[]) =>
  Array.from({ length: x.length / 2 }, (_, i) => x.slice(i * 2, i * 2 + 2));
export const flat = (x: Matrix) => x.flat();
export const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
export { softmax };
