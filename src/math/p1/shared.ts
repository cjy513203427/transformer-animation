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
  branch?: { base: Matrix; delta: Matrix; total: Matrix; baseLabel?: string; totalLabel?: string };
  line?: NumberLine;
  plot?: Plot;
  ranking?: Ranking;
  strip?: Strip;
  chain?: Chain;
  flow?: { nodes: { label: string; value: string }[]; active: number; caption: string };
  distribution?: { names: string[]; before: number[]; after?: number[]; caption: string };
  search?: { root: number; middle: number[]; operations: number[]; leaves: {label:string;value:number;state:'unvisited'|'match'|'miss'}[];caption:string };
}
/** Values drawn on horizontal number lines; the same index is the same coordinate in every row. */
export interface NumberLine {
  rows: { label: string; values: number[] }[];
  /** Coordinate names shown next to the points. */
  names: string[];
  /** One scale for every row (to compare values), or one scale per row (to compare shapes). */
  shared: boolean;
  /** Explicit [low, high] per row, overriding `shared`; e.g. to align a row with its affine image. */
  domains?: ([number, number] | undefined)[];
  /** Representable grid values, e.g. quantization levels; drawn on every row. */
  ticks?: number[];
  /** Labelled reference values such as μ or a clipping bound, each on one row. */
  markers?: { row: number; value: number; label: string }[];
  caption: string;
}
/**
 * An x–y chart. Curves are sampled from the same function the step computes, so the current
 * point always lies on its curve; nothing on the chart is drawn by hand.
 */
export interface Plot {
  x: Axis;
  y: Axis;
  /** Up to three curves, told apart by line style and an end label, not by colour alone. */
  curves: { label: string; points: [number, number][] }[];
  /** Values computed for the current input; `series` picks the marker shape. */
  points: { label: string; x: number; y: number; series?: number }[];
  /** Reference lines such as a threshold, a clipping bound or the current parameter. */
  guides?: { axis: 'x' | 'y'; value: number; label: string }[];
  caption: string;
}
export interface Axis {
  label: string;
  min: number;
  max: number;
}
/** Candidates sorted by probability, before and after a filter that renormalizes. */
export interface Ranking {
  items: {
    name: string;
    before: number;
    after: number;
    kept: boolean;
    /** Running sum of `before` in this order; drawn as a staircase when `threshold` is set. */
    cumulative: number;
  }[];
  /** Top-P threshold on the cumulative sum. */
  threshold?: number;
  caption: string;
}
/** [0, 1) split into consecutive segments in token order; one uniform number picks a segment. */
export interface Strip {
  segments: { name: string; start: number; end: number }[];
  u: number;
  chosen: number;
  caption: string;
}
/** A recurrence unrolled in time: every step keeps part of the old state and writes new input. */
export interface Chain {
  rows: {
    label: string;
    initial: number;
    steps: { input: number; keep: number; drive: number; state: number }[];
  }[];
  /** Steps after this index are not computed yet; absent means every step is shown. */
  cursor?: number;
  /** Two rows whose final states are equal, e.g. colliding histories. */
  collide?: boolean;
  caption: string;
}
/** `count` evenly spaced samples of f on [from, to], both ends included. */
export const sample = (from: number, to: number, count: number, f: (x: number) => number) =>
  Array.from({ length: count }, (_, i): [number, number] => {
    const x = from + ((to - from) * i) / (count - 1);
    return [x, f(x)];
  });
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
