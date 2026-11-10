import { formatNumber as fmt } from '../../math/attention';
import { preference } from '../../math/p1/preference';
import { BOOK } from '../book';
import { field, lesson, vec, type TopicDefinition } from './types';
import { T } from '../../i18n';
const DPO = 'https://arxiv.org/html/2305.18290v3',
  PPO = 'https://spinningup.openai.com/en/latest/algorithms/ppo.html',
  R1 = 'https://arxiv.org/html/2501.12948v1';
export default {
  id: 'preference',
  path: '/preference',
  title: T('偏好学习：奖励怎样进入训练目标？'),
  chapter: T('第 29、31 章'),
  intro: T(
    '给同一个 prompt 的四个回答打分，拆开偏好概率、DPO、PPO 裁剪与 GRPO 的组内优势。这里只核对公开目标，不运行训练。',
  ),
  question: T('同一个问题：人的偏好怎样变成可以计算的训练目标'),
  headline: T('偏好概率 → DPO → PPO 裁剪 → 组内优势 → GRPO 目标 → 平移与蒸馏。'),
  why: {
    question: T('模型已经会说话了，怎样让它更符合人的偏好？'),
    points: [
      T(
        '原书 29.1 的说法：预训练让模型“会说话”，对齐让模型“说人话”。常见做法是让人比较两个回答哪个更好，再把这种偏好变成训练目标。',
      ),
      T(
        'RLHF 先训练奖励模型给回答打分，再用 PPO 优化策略；DPO 直接用偏好对构造损失；DeepSeek-R1 用 GRPO：同一道题生成一组回答，用组内平均作基线，不再需要单独的 Critic 模型。本页只计算这些公开目标里的关键数字。',
      ),
    ],
    analogy: {
      title: T('比较几次作答，不请额外的老师（31.4.3）'),
      text: T(
        'PPO 像让学生做一道题，再请另一位老师来评判；GRPO 像让学生把同一类题做好几遍，看哪几遍比平均水平好，不需要额外的“老师”（Critic 模型）。',
      ),
    },
    book: [T('第 29.1 节 · 为什么需要对齐'), BOOK.ch29],
  },
  input: [0.8, 0.3, 0.9, 0.5],
  inputNames: [T('回答 0 的奖励'), T('回答 1 的奖励'), T('回答 2 的奖励'), T('回答 3 的奖励')],
  fields: [
    field('policy', T('当前回答0的 logit'), 0.5, -2, 2, 0.25, [1, 4, 5]),
    field('beta', 'DPO β', 0.1, 0.05, 1, 0.05, [1]),
    field('ratio', T('单项 PPO 比率 ρ'), 1.5, 0.25, 2, 0.25, [2]),
    field('advantage', T('单项 PPO 优势 A'), 1, -2, 2, 0.25, [2]),
    field('clip', T('裁剪宽度 ε'), 0.2, 0.1, 0.4, 0.1, [2, 4]),
    field('kl', T('GRPO 参考约束系数'), 0.02, 0, 0.2, 0.01, [4]),
    field('shift', T('奖励共同平移 c'), 0, -3, 3, 0.25, [3, 5]),
  ],
  steps: [
    lesson({
      name: T('偏好概率'),
      title: T('两个回答谁更好：只看奖励差'),
      formula: String.raw`P(y_w\succ y_l)=\sigma(r_w-r_l)`,
      explanation: (c) => {
        const r = c.values(T('给定奖励 r'));
        return T(
          'Bradley–Terry 模型假设：回答 0 胜过回答 1 的概率只取决于两者的奖励差，P = σ(r₀ − r₁)。当前 r₀ = {#0}、r₁ = {#1}，P = {#2}。原书 29.3.2 的例子：r(A) = 5、r(B) = 3 时 P = σ(2) ≈ 0.88；两者相等时是 0.5。',
          fmt(r[0]),
          fmt(r[1]),
          fmt(c.metric(T('P(回答0 优于回答1)'))),
        );
      },
      prompt: T('同一对奖励都加 2，偏好概率会变吗？'),
      takeaway: T('偏好概率只看奖励差，不看奖励的绝对零点。'),
      book: [T('第 29.3.2 节 · Bradley-Terry 模型'), BOOK.ch29],
    }),
    lesson({
      name: 'DPO',
      title: T('DPO：不训练奖励模型，直接用偏好对算损失'),
      formula: String.raw`L=-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right]\right)`,
      explanation: (c) =>
        T(
          '固定回答 0 被偏好、回答 1 被拒绝。DPO 比较当前策略相对参考策略的 log 概率比：被偏好回答的比值越高、被拒绝回答的越低，损失越小。当前 β × log 比值差 = {#0}，DPO loss = {#1}。拖动“当前回答0的 logit”，看损失怎样变化。',
          fmt(c.metric(T('β × log-ratio 差'))),
          fmt(c.metric('DPO loss')),
        ),
      prompt: T('提高被偏好回答相对于参考策略的概率，损失怎样变化？'),
      takeaway: T('DPO 从偏好对直接构造策略损失，不单独训练奖励模型。'),
      book: [T('第 29.5.2 节 · DPO 的核心思想'), BOOK.ch29],
      source: [T('DPO 论文 Eq.7'), DPO],
    }),
    lesson({
      name: 'PPO-Clip',
      title: T('PPO：一次更新不要走得太远'),
      formula: String.raw`J=\min(\rho A,clip(\rho,1-\epsilon,1+\epsilon)A)`,
      explanation: (c) =>
        T(
          'ρ 是新旧策略给同一个回答的概率之比，A 是这个回答比预期好多少（优势）。PPO 在“不裁剪”和“把 ρ 限制在 [1 − ε, 1 + ε]”两项里取较小的一个。当前 ρ = {#0}、A = {#1}、ε = {#2}。A 为正时，ρ 超过 1 + ε 后再增大，目标也不再提高；A 为负时，ρ 低于 1 − ε 后再减小，目标同样不再提高。',
          fmt(c.metric(T('概率比 ρ'))),
          fmt(c.metric(T('优势 A'))),
          fmt(c.metric(T('裁剪宽度 ε'))),
        ),
      prompt: T('A 为负且比率低于下限时，min 选中哪一项？'),
      takeaway: T('裁剪限制的是目标给出的激励，不是强行截断 ρ；不能都简写成 clip(ρ)A。'),
      book: [T('第 29.4.3 节 · PPO 优化的直觉'), BOOK.ch29],
      source: [T('PPO 作者文档'), PPO],
    }),
    lesson({
      name: T('组内优势'),
      title: T('GRPO：同一道题的几个回答，和组平均比'),
      formula: String.raw`A_i=\frac{r_i-\bar r}{\sqrt{\frac1G\sum_j(r_j-\bar r)^2}}`,
      explanation: (c) =>
        T(
          'GRPO 的“组”是同一个 prompt 的多个回答。先算组平均 {#0}，每个回答减去平均，得到“比平均好多少”；DeepSeek-R1 论文还会除以组内标准差 {#1}，得到 A = {#2}。原书 31.4.3 的例子 [0.8, 0.3, 0.9, 0.5] 平均 0.625，书中只写了减平均这一步：+0.175、−0.325、+0.275、−0.125。分数全部相同时标准差为 0，本页约定优势为 0。',
          fmt(c.metric(T('组均值'))),
          fmt(c.metric(T('总体标准差（除以 G）'))),
          vec(c.values(T('标准化优势 A'))),
        ),
      prompt: T('奖励全部相同，为什么不能直接除以标准差？'),
      takeaway: T('组内减平均、除以标准差，得到的是相对优势；不能混用不同 prompt 的回答。'),
      book: [T('第 31.4.3 节 · GRPO 算法'), BOOK.ch31],
      source: ['DeepSeek-R1 §2.2.1', R1],
    }),
    lesson({
      name: T('GRPO 约束'),
      title: T('优势乘概率比，再扣掉离参考模型太远的部分'),
      formula: String.raw`J=\frac1G\sum_i\left[\min(\rho_iA_i,clip(\rho_i)A_i)-\beta k_i\right]`,
      explanation: (c) =>
        T(
          '每个回答的贡献 = PPO 式裁剪项 − β × KL 估计项；当前 β = {#0}，简化目标 = {#1}。KL 项惩罚策略离参考模型太远，即使所有优势都是 0，它也可能不为 0。本例把每个完整回答当作一个动作，省略逐 token 平均和优化过程。',
          fmt(c.metric(T('KL 系数'))),
          fmt(c.metric(T('GRPO 简化目标（最大化）'))),
        ),
      prompt: T('优势都为 0 时，参考约束是否一定也为 0？'),
      takeaway: T('相对奖励项与参考约束是两份独立的贡献。'),
      book: [T('第 31.4.3 节 · GRPO 算法'), BOOK.ch31],
      source: ['DeepSeek-R1 §2.2.1', R1],
    }),
    lesson({
      name: T('平移与蒸馏'),
      title: T('奖励整体平移不改优势；蒸馏是另一条路'),
      formula: String.raw`k_i=\frac{\pi_{ref,i}}{\pi_{\theta,i}}-\log\frac{\pi_{ref,i}}{\pi_{\theta,i}}-1`,
      explanation: (c) =>
        T(
          '先预测：所有奖励都加上同一个 c，组内优势会不会变？减平均这一步会把 c 抵消掉。表中另列一个玩具教师分布，它与学生分布的交叉熵是 {#0}：蒸馏让学生模仿教师的输出（原书 31.7），和用组奖励优化是两种不同的训练信号。',
          fmt(c.metric(T('玩具蒸馏交叉熵'))),
        ),
      prompt: T('奖励平移、策略概率变化、教师目标变化分别影响哪条路径？'),
      takeaway: T('公开目标可以计算；不能据此推断闭源模型的内部训练或隐藏推理。'),
      book: [T('第 31.7.1 节 · 蒸馏的动机'), BOOK.ch31],
    }),
  ],
  examples: [
    { name: T('书中 31.4.3：[0.8, 0.3, 0.9, 0.5]'), input: [0.8, 0.3, 0.9, 0.5], book: true },
    { name: T('不同奖励'), input: [0, 1, 2, 3] },
    { name: T('零方差组'), input: [2, 2, 2, 2] },
  ],
  compute: preference,
  prediction: {
    key: 'shift',
    target: 2,
    question: T('奖励共同加常数，组内优势会保持不变吗？'),
    label: T('GRPO 组内优势'),
  },
  sources: [
    { title: T('原书第 29 章 · RLHF 与偏好学习'), url: BOOK.ch29 },
    { title: T('原书第 31 章 · 推理模型革命'), url: BOOK.ch31 },
    { title: 'DPO Eq.7', url: DPO },
    { title: T('PPO 作者文档'), url: PPO },
    { title: 'DeepSeek-R1 §2.2.1', url: R1 },
  ],
  limits: [
    T('只计算给定样本目标，没有采样、critic、GAE、反向或优化器。'),
    T('R1 原文 std 未指定总体或样本分母；本页明确用总体标准差，不冒充精确训练代码。'),
    T('KL 样本估计项不等于单个回答上的完整分布 KL。蒸馏分布为教学构造，没有隐藏推理内容。'),
  ],
  bookNotes: [
    T(
      '原书 31.4.3 的 GRPO 优势只写了“减去组平均”；DeepSeek-R1 论文的公式还除以组内标准差。本页按论文计算，说明里同时列出书中只减平均的结果。',
    ),
    T(
      '原书 31.4.3 的示意图写“生成 8 个回答”，但只列出 4 个分数；本页的组就是这 4 个回答，平均值与书中的 0.625 一致。',
    ),
  ],
} satisfies TopicDefinition;
