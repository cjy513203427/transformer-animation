import { describe, expect, it } from 'vitest';
import { adamTrajectory, loss, sgdTrajectory, type OptimizerConfig } from './optimizer';
import { contextExtension, ramp, rotatedScore } from './context';

const base: OptimizerConfig = { eta: 0.1, beta1: 0.9, beta2: 0.999, lambda: 0, curvature: 10, mode: 0 };

describe('Adam 与 AdamW', () => {
  it('第一步与手算一致：两个坐标都走约 η，尽管梯度差 10 倍', () => {
    // g = [2, 20]; m₁ = 0.1g; v₁ = 0.001g²; m̂ = g; v̂ = g²; step = −η·g/(|g| + ε).
    const [r] = adamTrajectory([2, 2], base, 1);
    expect(r.g).toEqual([2, 20]);
    expect(r.m[0]).toBeCloseTo(0.2, 14);
    expect(r.v[1]).toBeCloseTo(0.4, 14);
    expect(r.adaptive[0]).toBeCloseTo(-0.1, 8);
    expect(r.adaptive[1]).toBeCloseTo(-0.1, 8);
    // AdamW adds −ηλw outside the normalization: 2 − 0.1 − 0.1·0.1·2 = 1.88.
    const [d] = adamTrajectory([2, 2], { ...base, lambda: 0.1 }, 1);
    expect(d.next[0]).toBeCloseTo(1.88, 8);
  });

  it('λ > 0 时 L2 与解耦衰减不同，λ = 0 时完全相同', () => {
    const steps = 5;
    const w = (c: OptimizerConfig) => adamTrajectory([2, 2], c, steps)[steps - 1].next;
    expect(w({ ...base, mode: 1 })).toEqual(w(base));
    const decoupled = w({ ...base, lambda: 0.1 }),
      coupled = w({ ...base, lambda: 0.1, mode: 1 });
    expect(Math.abs(decoupled[0] - coupled[0])).toBeGreaterThan(1e-3);
  });

  it('b = 100、η = 0.1 时 SGD 发散而 Adam 下降', () => {
    const c = { ...base, curvature: 100, lambda: 0.1 };
    const start = loss([2, 2], 100);
    expect(loss(sgdTrajectory([2, 2], c, 20, false)[20], 100)).toBeGreaterThan(start * 1e6);
    expect(loss(adamTrajectory([2, 2], c, 20)[19].next, 100)).toBeLessThan(start);
  });
});

describe('RoPE 上下文扩展', () => {
  const r = contextExtension(8, 10000, 4096, 8);
  it('频率、NTK 底数、斜坡与温度与论文公式一致', () => {
    r.theta.forEach((t, d) => expect(t).toBeCloseTo([1, 0.1, 0.01, 0.001][d], 15));
    expect(r.ntkBase).toBeCloseTo(160000, 6); // 10000 · 8^(8/6)
    expect(r.methods.ntk[3]).toBeCloseTo(0.001 / 8, 15);
    expect(r.methods.ntk[0]).toBe(1);
    const turns = (4096 * 0.01) / (2 * Math.PI);
    // θ₂ = 10000^(−1/2) differs from 0.01 in the last bit, so compare within tolerance.
    [1, 1, ramp(turns), 0].forEach((g, d) => expect(r.gamma[d]).toBeCloseTo(g, 14));
    expect(r.gamma[2]).toBeCloseTo((turns - 1) / 31, 15);
    expect(r.temperatureFactor).toBeCloseTo(1.2079, 4); // paper: ≈ 1.208 at s = 8
  });

  it('只有没转满一圈的维度会在外推时越界，PI 与 YaRN 都不越界', () => {
    expect(r.seen(r.methods.extrapolate)).toEqual([true, true, true, false]);
    expect(r.seen(r.methods.pi)).toEqual([true, true, true, true]);
    expect(r.seen(r.methods.yarn)).toEqual([true, true, true, true]);
  });

  it('YaRN 的高频维度与 s 无关', () => {
    expect(contextExtension(8, 10000, 4096, 16).methods.yarn[0]).toBe(r.methods.yarn[0]);
  });

  it('二维旋转打分等于 cos 与 sin 的展开', () => {
    const q = [0.3, -1.2],
      k = [2, 0.5],
      phi = 0.9;
    const expected =
      q[0] * (k[0] * Math.cos(phi) - k[1] * Math.sin(phi)) +
      q[1] * (k[0] * Math.sin(phi) + k[1] * Math.cos(phi));
    expect(rotatedScore(q, k, phi)).toBeCloseTo(expected, 14);
    expect(rotatedScore([1, 0], [1, 0], phi)).toBeCloseTo(Math.cos(phi), 15);
  });
});
