/**
 * RoPE context extension following YaRN (arXiv 2309.00071): θ_d = b^(−2d/|D|), s = L′/L.
 * PI divides positions by s (Appendix A.1, Eq. 16), equivalently θ/s. NTK-aware changes the base
 * to b·s^(|D|/(|D|−2)). NTK-by-parts blends θ/s and θ with the ramp γ(r), r = L/λ (Eq. 10–13),
 * α = 1, β = 32 as recommended for Llama. YaRN adds √(1/t) = 0.1·ln s + 1 (Eq. 15).
 */
export const ALPHA = 1;
export const BETA = 32;

export const ropeFrequencies = (dims: number, base: number) =>
  Array.from({ length: dims / 2 }, (_, d) => base ** ((-2 * d) / dims));
export const ramp = (r: number) => (r < ALPHA ? 0 : r > BETA ? 1 : (r - ALPHA) / (BETA - ALPHA));

/** q · R(φ) k for one two-dimensional pair. */
export function rotatedScore(q: number[], k: number[], angle: number) {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  return q[0] * (c * k[0] - s * k[1]) + q[1] * (s * k[0] + c * k[1]);
}

export function contextExtension(dims: number, base: number, trained: number, scale: number) {
  const theta = ropeFrequencies(dims, base);
  const wavelength = theta.map((t) => (2 * Math.PI) / t);
  const target = trained * scale;
  const ntkBase = base * scale ** (dims / (dims - 2));
  const ratio = wavelength.map((l) => trained / l);
  const gamma = ratio.map(ramp);
  const methods = {
    extrapolate: theta,
    pi: theta.map((t) => t / scale),
    ntk: ropeFrequencies(dims, ntkBase),
    yarn: theta.map((t, d) => ((1 - gamma[d]) * t) / scale + gamma[d] * t),
  };
  /**
   * Angles the model met in training for dimension d: every angle once a full turn fits in L
   * (λ ≤ L); otherwise only [0, L·θ_d), the range covered by positions 0 … L − 1.
   */
  const seen = (frequency: number[]) =>
    frequency.map((f, d) => wavelength[d] <= trained || (target - 1) * f < trained * theta[d]);
  return {
    theta,
    wavelength,
    target,
    ntkBase,
    ratio,
    gamma,
    methods,
    seen,
    /** √(1/t); q and k are both scaled by it, so logits are multiplied by its square. */
    temperatureFactor: 0.1 * Math.log(scale) + 1,
  };
}
