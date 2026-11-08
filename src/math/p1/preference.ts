import {
  check,
  frame,
  mean,
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
  return {
    frames: [
      frame(
        '同一个 prompt，比较两个回答的偏好',
        [tensor('给定奖励 r', reward)],
        [['P(回答0 优于回答1)', sigmoid(reward[0] - reward[1])]],
        'Bradley–Terry 用奖励差构造偏好概率；这是给定分数的模型，不是人类判断的真值。',
      ),
      frame(
        'DPO 直接使用偏好对与固定参考策略',
        [tensor('当前 π', policy), tensor('参考 π_ref', reference)],
        [
          ['偏好对', '回答0 被偏好，回答1 被拒绝'],
          ['β × log-ratio 差', margin],
          ['DPO loss', dpoLoss],
        ],
        '这里把完整回答当作单动作；真实回答的 log 概率是各 token 条件 log 概率之和。DPO 这条路径不单独训练奖励模型。',
      ),
      frame(
        'PPO-Clip 是目标中的 min，不是强制裁剪策略',
        [tensor('单项比较', [p.ratio * p.advantage, ppoTerm(p.ratio, p.advantage, p.clip)])],
        [
          ['概率比 ρ', p.ratio],
          ['优势 A', p.advantage],
          ['裁剪宽度 ε', p.clip],
        ],
        'A 的正负会改变哪一侧受限制。这里只算单项 surrogate，省略 critic、GAE、采样和优化。',
      ),
      frame(
        'GRPO 的组：同一 prompt 的四个回答',
        [tensor('组内奖励', reward), tensor('标准化优势 A', group.advantages)],
        [
          ['组均值', group.mu],
          ['总体标准差（除以 G）', group.std],
        ],
        '采用总体标准差作为教学约定。零方差时优势设为 0，避免除零；不能混用不同 prompt 的回答做这一组。',
      ),
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
      frame(
        '奖励平移不改组内相对优势；蒸馏是另一条监督路径',
        [
          tensor('平移后优势', group.advantages),
          tensor('玩具教师分布', teacher),
          tensor('学生分布', policy),
        ],
        [['玩具蒸馏交叉熵', crossEntropy]],
        '教师分布取给定奖励的 Softmax，只用于示意概率匹配。没有闭源模型训练细节或隐藏推理；增加采样预算也不保证答案更好。',
      ),
    ],
    checks: [
      check('奖励共同平移后优势不变', original.advantages, group.advantages),
      check('零方差边界优势为零', groupAdvantages([2, 2, 2, 2]).advantages, [0, 0, 0, 0]),
    ],
    signature: group.advantages,
  };
}
