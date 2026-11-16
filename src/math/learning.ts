import { matmul, softmax, type Matrix } from './attention';

export const TEACHING_WEIGHTS: Matrix = [[0.5, -0.5, 0], [0, 0.5, -0.5]];

export function supervisedStep(hidden: number[], target: number, rate: number, shift = 0) {
  const weights = TEACHING_WEIGHTS.map((row) => [...row]);
  const logits = matmul([hidden], weights)[0].map((value) => value + shift);
  const probabilities = softmax(logits);
  const maximum = Math.max(...logits);
  const loss = maximum + Math.log(logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0)) - logits[target];
  const logitGradient = probabilities.map((value, index) => value - Number(index === target));
  const weightGradient = hidden.map((value) => logitGradient.map((gradient) => value * gradient));
  const nextWeights = weights.map((row, i) => row.map((value, j) => value - rate * weightGradient[i][j]));
  const nextLogits = matmul([hidden], nextWeights)[0].map((value) => value + shift);
  const nextProbabilities = softmax(nextLogits);
  const nextMaximum = Math.max(...nextLogits);
  const nextLoss = nextMaximum + Math.log(nextLogits.reduce((sum, value) => sum + Math.exp(value - nextMaximum), 0)) - nextLogits[target];
  return { hidden, weights, logits, probabilities, loss, logitGradient, weightGradient, nextWeights,
    nextLogits, nextProbabilities, nextLoss };
}

export function policyObjective(logits: number[], old: number[], rewards: number[], baseline: number, beta: number, epsilon: number) {
  const probabilities = softmax(logits);
  const reference = [0.5, 0.5];
  const advantages = rewards.map((reward) => reward - baseline);
  const ratios = probabilities.map((value, i) => value / old[i]);
  const clipped = ratios.map((ratio) => Math.max(1 - epsilon, Math.min(1 + epsilon, ratio)));
  const terms = ratios.map((ratio, i) => Math.min(ratio * advantages[i], clipped[i] * advantages[i]));
  const klTerms = probabilities.map((value, i) => value * Math.log(value / reference[i]));
  const kl = klTerms.reduce((sum, value) => sum + value, 0);
  const objective = terms.reduce((sum, value, i) => sum + old[i] * value, 0) - beta * kl;
  return { probabilities, reference, advantages, ratios, clipped, terms, klTerms, kl, objective };
}

/** Exact enumeration of two terminal actions. One gradient step starts at the rollout policy. */
export function policyStep(rewards: number[], oldLogit: number, baseline: number, rate: number, beta: number, epsilon: number) {
  const oldLogits = [oldLogit, 0];
  const old = softmax(oldLogits);
  const before = policyObjective(oldLogits, old, rewards, baseline, beta, epsilon);
  const expectedReward = old.reduce((sum, probability, i) => sum + probability * rewards[i], 0);
  const expectedAdvantage = expectedReward - baseline;
  const gradient = old.map((probability, i) => probability *
    (before.advantages[i] - expectedAdvantage - beta * (Math.log(probability / before.reference[i]) - before.kl)));
  const nextLogits = oldLogits.map((value, i) => value + rate * gradient[i]);
  const after = policyObjective(nextLogits, old, rewards, baseline, beta, epsilon);
  const dpoMargin = beta * (Math.log(after.probabilities[0] / after.reference[0]) - Math.log(after.probabilities[1] / after.reference[1]));
  const dpoLoss = Math.max(-dpoMargin, 0) + Math.log1p(Math.exp(-Math.abs(dpoMargin)));
  const criticLoss = old.reduce((sum, probability, i) => sum + probability * (rewards[i] - baseline) ** 2, 0);
  return { oldLogits, old, before, expectedReward, expectedAdvantage, gradient, nextLogits, after, dpoMargin, dpoLoss, criticLoss };
}

/** Four complete paths through the ordered expression a ± b ± c. */
export function thoughtSearch(numbers: number[], target: number, budget: number) {
  const [a, b, c] = numbers;
  const paths = [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([first, second]) => ({
    first, second, intermediate: a + first * b, result: a + first * b + second * c,
  }));
  const visited = paths.slice(0, budget);
  const found = visited.findIndex((path) => path.result === target);
  return { paths, visited, found, solution: found < 0 ? null : visited[found],
    errors: visited.map((path) => Math.abs(path.result - target)) };
}
