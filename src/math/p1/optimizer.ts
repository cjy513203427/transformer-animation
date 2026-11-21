/**
 * SGD, Adam with L2 and AdamW on L(w) = ½(a·w0² + b·w1²), a = 1.
 * Follows the PyTorch formulation: AdamW first scales the weights by (1 − ηλ), then applies the
 * Adam step; "Adam + L2" adds λw to the gradient before the moments.
 */
export const EPSILON = 1e-8;

export interface OptimizerConfig {
  eta: number;
  beta1: number;
  beta2: number;
  lambda: number;
  /** Curvature b of the second coordinate; a = 1. */
  curvature: number;
  /** 0: AdamW (decoupled decay); 1: Adam with L2 added to the gradient. */
  mode: number;
  /** Multiplies the whole loss; Adam's direction should not care. */
  lossScale?: number;
}
export interface AdamRecord {
  t: number;
  w: number[];
  lossGradient: number[];
  g: number[];
  m: number[];
  v: number[];
  mHat: number[];
  vHat: number[];
  adaptive: number[];
  decay: number[];
  next: number[];
}

export const loss = (w: number[], curvature: number, scale = 1) =>
  (scale * (w[0] ** 2 + curvature * w[1] ** 2)) / 2;
export const lossGradient = (w: number[], curvature: number, scale = 1) => [
  scale * w[0],
  scale * curvature * w[1],
];

export function adamTrajectory(w0: number[], c: OptimizerConfig, steps: number): AdamRecord[] {
  const scale = c.lossScale ?? 1;
  let w = w0.slice(),
    m = [0, 0],
    v = [0, 0];
  const records: AdamRecord[] = [];
  for (let t = 1; t <= steps; t++) {
    const lg = lossGradient(w, c.curvature, scale);
    const g = c.mode === 1 ? lg.map((x, i) => x + c.lambda * w[i]) : lg;
    m = m.map((x, i) => c.beta1 * x + (1 - c.beta1) * g[i]);
    v = v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] ** 2);
    const mHat = m.map((x) => x / (1 - c.beta1 ** t)),
      vHat = v.map((x) => x / (1 - c.beta2 ** t));
    const adaptive = mHat.map((x, i) => (-c.eta * x) / (Math.sqrt(vHat[i]) + EPSILON));
    const decay = c.mode === 1 ? [0, 0] : w.map((x) => -c.eta * c.lambda * x);
    const next = w.map((x, i) => x + decay[i] + adaptive[i]);
    records.push({ t, w, lossGradient: lg, g, m, v, mHat, vHat, adaptive, decay, next });
    w = next;
  }
  return records;
}

/** Plain SGD. decoupled: w ← (1 − ηλ)w − η∇L; otherwise w ← w − η(∇L + λw). */
export function sgdTrajectory(w0: number[], c: OptimizerConfig, steps: number, decoupled: boolean) {
  const path = [w0.slice()];
  let w = w0.slice();
  for (let t = 1; t <= steps; t++) {
    const g = lossGradient(w, c.curvature, c.lossScale ?? 1);
    w = decoupled
      ? w.map((x, i) => (1 - c.eta * c.lambda) * x - c.eta * g[i])
      : w.map((x, i) => x - c.eta * (g[i] + c.lambda * x));
    path.push(w);
  }
  return path;
}
