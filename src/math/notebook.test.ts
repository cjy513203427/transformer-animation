import { expect, it } from 'vitest';
import { gradientTrajectory, tallyAnswers, trainingBudget } from './notebook';

it('投票处理平局、提取失败和一致但错误的答案', () => {
  expect(tallyAnswers([5, 3, null, 5, 3], 2).winners).toEqual([3, 5]);
  expect(tallyAnswers([5, 3, null, 5, 3], 2).winner).toBeNull();
  expect(tallyAnswers([null, null], 2)).toMatchObject({ valid: 0, share: 0, winner: null });
  expect(tallyAnswers([3, 3, 5, 3, 5], 5)).toMatchObject({ winner: 3, share: 0.6 });
  expect(tallyAnswers([5, 5, null, 3, 3], 3)).toMatchObject({ valid: 2, invalid: 1, share: 1 });
});
it('二次损失SGD满足闭式解，过大学习率发散，warmup初期确实减小步长', () => {
  for (const eta of [0.05, 0.5, 1, 1.5, 2, 2.5]) {
    const path = gradientTrajectory(eta, 12, false);
    path.forEach((row) => expect(row.w).toBeCloseTo(2 * (1 - eta) ** row.step, 12));
  }
  expect(gradientTrajectory(2.5, 12, false)[12].loss).toBeGreaterThan(2);
  expect(gradientTrajectory(2, 12, false)[12].loss).toBe(2);
  const scheduled = gradientTrajectory(0.6, 12, true);
  expect(scheduled[0].rate).toBeCloseTo(0.2);
  expect(scheduled[2].rate).toBeCloseTo(0.6);
  expect(scheduled[12].rate).toBe(0);
});
it('训练账本按参数和token分别线性缩放，不把20:1当作限制', () => {
  expect(trainingBudget(7, 140)).toEqual({ flops: 5.88e21, tokensPerParameter: 20 });
  expect(trainingBudget(14, 140).flops).toBe(2 * trainingBudget(7, 140).flops);
  expect(trainingBudget(7, 280).flops).toBe(2 * trainingBudget(7, 140).flops);
});
