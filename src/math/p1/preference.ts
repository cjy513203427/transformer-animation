import { formatNumber as fmt } from '../attention';
import {
  check,
  frame,
  mean,
  sample,
  softmax,
  softplus,
  tensor,
  type Computation,
  type Parameters,
} from './shared';
export const sigmoid = (x: number) => Math.exp(-softplus(-x));
export function groupAdvantages(rewards: number[]) {
  const mu = mean(rewards),
    std = Math.sqrt(mean(rewards.map((r) => (r - mu) ** 2)));
  return { mu, std, advantages: rewards.map((r) => (std === 0 ? 0 : (r - mu) / std)) };
}
export function ppoTerm(ratio: number, advantage: number, clip: number) {
  return Math.min(ratio * advantage, Math.max(1 - clip, Math.min(1 + clip, ratio)) * advantage);
}
const ANSWERS = ['回答0', '回答1', '回答2', '回答3'];
/** A symmetric range that keeps `value` inside with some room, at least ±`least`. */
const around = (value: number, least: number) => Math.max(least, Math.ceil(Math.abs(value)) + 1);
/** [low, high] around the values with 8% padding; a constant row gets ±1. */
function span(values: number[]): [number, number] {
  const low = Math.min(...values),
    high = Math.max(...values);
  if (high - low < 1e-12) return [low - 1, high + 1];
  const pad = (high - low) * 0.08;
  return [low - pad, high + pad];
}
/** The domain of `(r − μ) / σ` that puts every standardized point under its raw point. */
const standardized = ([low, high]: [number, number], mu: number, std: number): [number, number] =>
  std === 0 ? [-1, 1] : [(low - mu) / std, (high - mu) / std];
export function preference(x: number[], p: Parameters): Computation {
  const reward = x.map((v) => v + p.shift),
    group = groupAdvantages(reward),
    original = groupAdvantages(x),
    old = softmax([0.5, 0, -0.5, 0]),
    policy = softmax([p.policy, 0, -0.5, 0]),
    reference = [0.25, 0.25, 0.25, 0.25];
  const margin =
      p.beta *
      (Math.log(policy[0]) -
        Math.log(reference[0]) -
        (Math.log(policy[1]) - Math.log(reference[1]))),
    dpoLoss = softplus(-margin),
    ratios = policy.map((v, i) => v / old[i]),
    kl = policy.map((v, i) => {
      const r = reference[i] / v;
      return r - Math.log(r) - 1;
    }),
    clipped = ratios.map((r, i) => ppoTerm(r, group.advantages[i], p.clip)),
    objective = mean(clipped.map((v, i) => v - p.kl * kl[i]));
  const teacher = softmax(x),
    crossEntropy = -teacher.reduce((s, v, i) => s + v * Math.log(policy[i]), 0);
  const gap = reward[0] - reward[1],
    btRange = around(gap, 6),
    dpoRange = around(margin, 3),
    ppoCurves = [
      { label: 'PPO 目标', points: sample(0, 2.5, 101, (r) => ppoTerm(r, p.advantage, p.clip)) },
      { label: '不裁剪 ρA', points: sample(0, 2.5, 101, (r) => r * p.advantage) },
    ],
    ppoValues = ppoCurves.flatMap((c) => c.points.map(([, y]) => y)),
    ppoY = span([...ppoValues, 0]);
  const rewardDomain = span([...reward, group.mu - group.std, group.mu + group.std]),
    shiftDomain = span([...x, ...reward]);
  return {
    frames: [
      {
        ...frame(
          '同一个 prompt，比较两个回答的偏好',
          [tensor('给定奖励 r', reward)],
          [['P(回答0 优于回答1)', sigmoid(gap)]],
          'Bradley–Terry 用奖励差构造偏好概率；这是给定分数的模型，不是人类判断的真值。',
        ),
        plot: {
          x: { label: '奖励差 r₀ − r₁', min: -btRange, max: btRange },
          y: { label: 'P(回答0 更好)', min: 0, max: 1 },
          curves: [{ label: 'σ', points: sample(-btRange, btRange, 121, sigmoid) }],
          points: [
            { label: '当前', x: gap, y: sigmoid(gap) },
            { label: '书中 5 vs 3', x: 2, y: sigmoid(2), series: 1 },
          ],
          guides: [{ axis: 'y', value: 0.5, label: '0.5：一样好' }],
          caption: `横轴只有奖励差 r₀ − r₁ = ${fmt(gap)}，所以两个奖励同时加同一个数，点的位置不变。差越大，概率越接近 1，但不会等于 1。`,
        },
      },
      {
        ...frame(
          'DPO 直接使用偏好对与固定参考策略',
          [tensor('当前 π', policy), tensor('参考 π_ref', reference)],
          [
            ['偏好对', '回答0 被偏好，回答1 被拒绝'],
            ['β × log-ratio 差', margin],
            ['DPO loss', dpoLoss],
          ],
          '这里把完整回答当作单动作；真实回答的 log 概率是各 token 条件 log 概率之和。DPO 这条路径不单独训练奖励模型。',
        ),
        plot: {
          x: { label: 'm = β × log 比值差', min: -dpoRange, max: dpoRange },
          y: { label: 'DPO loss', min: 0, max: softplus(dpoRange) },
          curves: [
            { label: '−log σ(m)', points: sample(-dpoRange, dpoRange, 121, (m) => softplus(-m)) },
          ],
          points: [{ label: '当前', x: margin, y: dpoLoss }],
          guides: [{ axis: 'x', value: 0, label: `m = 0：loss = ${fmt(Math.LN2)}` }],
          caption: `当前 m = ${fmt(margin)}。被偏好回答相对参考策略的概率比越高，m 越大，损失越小；损失一直大于 0，没有“到此为止”的点。`,
        },
      },
      {
        ...frame(
          'PPO-Clip 是目标中的 min，不是强制裁剪策略',
          [tensor('单项比较', [p.ratio * p.advantage, ppoTerm(p.ratio, p.advantage, p.clip)])],
          [
            ['概率比 ρ', p.ratio],
            ['优势 A', p.advantage],
            ['裁剪宽度 ε', p.clip],
          ],
          'A 的正负会改变哪一侧受限制。这里只算单项 surrogate，省略 critic、GAE、采样和优化。',
        ),
        plot: {
          x: { label: '概率比 ρ', min: 0, max: 2.5 },
          y: { label: '单项目标', min: ppoY[0], max: ppoY[1] },
          curves: ppoCurves,
          points: [{ label: '当前', x: p.ratio, y: ppoTerm(p.ratio, p.advantage, p.clip) }],
          guides: [
            { axis: 'x', value: 1 - p.clip, label: '1 − ε' },
            { axis: 'x', value: 1 + p.clip, label: '1 + ε' },
          ],
          caption:
            p.advantage > 0
              ? `A = ${fmt(p.advantage)} > 0：ρ 超过 1 + ε 后实线变平，再增大 ρ 也得不到更多目标；虚线是不裁剪的 ρA。`
              : p.advantage < 0
                ? `A = ${fmt(p.advantage)} < 0：ρ 低于 1 − ε 后实线变平；ρ 偏大时仍按 ρA 计入，目标继续下降。`
                : 'A = 0：两项都是 0，目标与 ρ 无关。',
        },
      },
      {
        ...frame(
          'GRPO 的组：同一 prompt 的四个回答',
          [tensor('组内奖励', reward), tensor('标准化优势 A', group.advantages)],
          [
            ['组均值', group.mu],
            ['总体标准差（除以 G）', group.std],
          ],
          '采用总体标准差作为教学约定。零方差时优势设为 0，避免除零；不能混用不同 prompt 的回答做这一组。',
        ),
        line: {
          rows: [
            { label: '组内奖励 r', values: reward },
            { label: '标准化优势 A', values: group.advantages },
          ],
          names: ANSWERS,
          shared: false,
          domains: [rewardDomain, standardized(rewardDomain, group.mu, group.std)],
          markers:
            group.std === 0
              ? [
                  { row: 0, value: group.mu, label: '均值' },
                  { row: 1, value: 0, label: '0' },
                ]
              : [
                  { row: 0, value: group.mu - group.std, label: '均值−σ' },
                  { row: 0, value: group.mu, label: '均值' },
                  { row: 0, value: group.mu + group.std, label: '均值+σ' },
                  { row: 1, value: -1, label: '−1' },
                  { row: 1, value: 0, label: '0' },
                  { row: 1, value: 1, label: '+1' },
                ],
          caption:
            group.std === 0
              ? '四个奖励相同，标准差为 0：本页约定优势全为 0，不做除法。'
              : `两行的刻度对齐了：减去均值 ${fmt(group.mu)}、除以标准差 ${fmt(group.std)}，上面的“均值 ± σ”正好落在下面的 ±1，每个回答的相对位置不变。`,
        },
      },
      frame(
        '组优势乘概率比，再减参考约束估计项',
        [
          tensor('π / π_old', ratios),
          tensor('各回答 clipped 项', clipped),
          tensor('参考 KL 样本估计项', kl),
        ],
        [
          ['GRPO 简化目标（最大化）', objective],
          ['KL 系数', p.kl],
        ],
        '用 R1 的 ref/current − log(ref/current) − 1 样本项；这不是单个回答上的完整分布 KL。零优势时参考约束仍可能生效。',
      ),
      {
        ...frame(
          '奖励平移不改组内相对优势；蒸馏是另一条监督路径',
          [
            tensor('平移后优势', group.advantages),
            tensor('玩具教师分布', teacher),
            tensor('学生分布', policy),
          ],
          [['玩具蒸馏交叉熵', crossEntropy]],
          '教师分布取给定奖励的 Softmax，只用于示意概率匹配。没有闭源模型训练细节或隐藏推理；增加采样预算也不保证答案更好。',
        ),
        line: {
          rows: [
            { label: '原奖励 r', values: x },
            { label: `平移后 r + c（c = ${fmt(p.shift)}）`, values: reward },
            { label: '优势 A', values: group.advantages },
          ],
          names: ANSWERS,
          shared: false,
          domains: [shiftDomain, shiftDomain, standardized(shiftDomain, group.mu, group.std)],
          markers: [
            { row: 0, value: original.mu, label: '均值' },
            { row: 1, value: group.mu, label: '均值' },
          ],
          caption:
            p.shift === 0
              ? '当前 c = 0，前两行重合。拖动“奖励共同平移 c”：点和均值一起平移，优势一行不动。蒸馏的两个分布见右侧表格。'
              : `每个奖励都加 c = ${fmt(p.shift)}，组均值也加 c：差值不变，所以第三行的优势和平移前一样。蒸馏的两个分布见右侧表格。`,
        },
      },
    ],
    checks: [
      check('奖励共同平移后优势不变', original.advantages, group.advantages),
      check('零方差边界优势为零', groupAdvantages([2, 2, 2, 2]).advantages, [0, 0, 0, 0]),
    ],
    signature: group.advantages,
  };
}
