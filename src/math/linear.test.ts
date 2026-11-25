import { describe, expect, it } from 'vitest';
import {
  computeLinear,
  determinant2x2,
  dotGeometry,
  LINEAR_PRESETS,
  matmul2x2,
  SECOND_MATRIX,
  transformVec,
  type Mat2,
} from './linear';

describe('二维线性变换与点积几何', () => {
  it('行向量乘法等于用输入分量线性组合 W 的两行', () => {
    const w: Mat2 = [
      [1.25, -0.5],
      [0.75, 1.5],
    ];
    const x: [number, number] = [2, -1];
    const y = transformVec(x, w);
    expect(y[0]).toBeCloseTo(2 * w[0][0] + -1 * w[1][0], 12);
    expect(y[1]).toBeCloseTo(2 * w[0][1] + -1 * w[1][1], 12);
    expect(transformVec([1, 0], w)).toEqual(w[0]);
    expect(transformVec([0, 1], w)).toEqual(w[1]);
  });

  it('行列式反映面积缩放、镜像翻转与降维奇异矩阵', () => {
    expect(determinant2x2(LINEAR_PRESETS.identity.w)).toBeCloseTo(1, 12);
    expect(determinant2x2(LINEAR_PRESETS.rotate90.w)).toBeCloseTo(1, 12);
    expect(determinant2x2(LINEAR_PRESETS.shear.w)).toBeCloseTo(1, 12);
    expect(determinant2x2(LINEAR_PRESETS.mirror.w)).toBeCloseTo(-1, 12);
    expect(determinant2x2(LINEAR_PRESETS.collapse.w)).toBeCloseTo(0, 12);
    expect(computeLinear(LINEAR_PRESETS.collapse.w, [1, 0], [1, 1]).orientation).toBe('collapsed');
    expect(computeLinear(LINEAR_PRESETS.mirror.w, [1, 0], [1, 1]).orientation).toBe('flipped');
  });

  it('代数点积与几何公式一致；只拉长 k 使点积翻倍而余弦不变；零向量无 NaN', () => {
    const g = dotGeometry([1.5, 0.5], [1, 1]);
    expect(g.dot).toBeCloseTo(g.geometricDot, 12);
    expect(g.cosine).toBeCloseTo(2 / (Math.hypot(1.5, 0.5) * Math.SQRT2), 12);
    const doubled = dotGeometry([1.5, 0.5], [2, 2]);
    expect(doubled.dot).toBeCloseTo(g.dot * 2, 12);
    expect(doubled.cosine).toBeCloseTo(g.cosine!, 12);

    const zero = dotGeometry([0, 0], [1, 2]);
    expect(zero.dot).toBe(0);
    expect(zero.cosine).toBeNull();
    expect(zero.angleDeg).toBeNull();
  });

  it('两次线性变换满足结合律 (xW₁)W₂ = x(W₁W₂)，且一般不满足交换律', () => {
    for (const preset of Object.values(LINEAR_PRESETS)) {
      const r = computeLinear(preset.w, preset.x, preset.k);
      expect(r.composeError).toBeLessThan(1e-12);
      expect(r.composedW).toEqual(matmul2x2(preset.w, SECOND_MATRIX));
    }
    const shear = computeLinear(LINEAR_PRESETS.shear.w, LINEAR_PRESETS.shear.x, LINEAR_PRESETS.shear.k);
    expect(shear.commuteGap).toBeGreaterThan(0.1);
  });
});
