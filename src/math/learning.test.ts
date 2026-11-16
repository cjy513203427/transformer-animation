import { expect, it } from 'vitest';
import { policyObjective, policyStep, supervisedStep, thoughtSearch } from './learning';
import { softmax } from './attention';

it('输出层反向梯度与独立中心差分一致，更新后真实损失下降', () => {
  const hidden = [1, -1], target = 1;
  const result = supervisedStep(hidden, target, 0.25);
  const loss = (weights: number[][]) => {
    const logits = weights[0].map((_, j) => hidden.reduce((sum, value, i) => sum + value * weights[i][j], 0));
    return -Math.log(softmax(logits)[target]);
  };
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    const plus = result.weights.map((row) => [...row]), minus = result.weights.map((row) => [...row]);
    plus[i][j] += 1e-6; minus[i][j] -= 1e-6;
    expect(result.weightGradient[i][j]).toBeCloseTo((loss(plus) - loss(minus)) / 2e-6, 8);
  }
  expect(result.nextLoss).toBeLessThan(result.loss);
  expect(result.nextProbabilities[target]).toBeGreaterThan(result.probabilities[target]);
  expect(supervisedStep([0, 0], target, 1).weightGradient.flat()).toEqual([0, -0, 0, 0, -0, 0]);
  expect(supervisedStep(hidden, target, 0.25, 3).nextProbabilities).toEqual(expect.arrayContaining(result.nextProbabilities.map((v) => expect.closeTo(v, 12))));
});

it('策略梯度包含参考KL且与独立差分一致，完整期望下价值基线抵消', () => {
  const r = [1, -0.5], before = policyStep(r, 0.5, 0.25, 0.5, 0.2, 0.2);
  for (let i = 0; i < 2; i++) {
    const plus = [...before.oldLogits], minus = [...before.oldLogits];
    plus[i] += 1e-6; minus[i] -= 1e-6;
    const difference = (policyObjective(plus, before.old, r, 0.25, 0.2, 0.2).objective -
      policyObjective(minus, before.old, r, 0.25, 0.2, 0.2).objective) / 2e-6;
    expect(before.gradient[i]).toBeCloseTo(difference, 8);
  }
  const shifted = policyStep(r, 0.5, -2, 0.5, 0.2, 0.2);
  shifted.after.probabilities.forEach((value, i) => expect(value).toBeCloseTo(before.after.probabilities[i], 13));
  expect(before.after.probabilities[0]).toBeGreaterThan(before.old[0]);
  expect(policyStep([1, 1], 0, 0, 1, 0.2, 0.2).after.probabilities).toEqual([0.5, 0.5]);
  expect(policyStep([1, 0], 0, 0, 4, 0.2, 0.2).after.ratios[0]).toBeGreaterThan(1.2);
});

it('有界搜索可因预算不足失败，扩展到第三条路径后得到解', () => {
  expect(thoughtSearch([2, 3, 1], 0, 1).solution).toBeNull();
  expect(thoughtSearch([2, 3, 1], 0, 3).solution?.result).toBe(0);
  expect(thoughtSearch([2, 3, 1], 12, 4).solution).toBeNull();
});
