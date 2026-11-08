import { formatNumber as fmt } from '../../math/attention';
import { ssm } from '../../math/p1/ssm';
import { BOOK } from '../book';
import { choice, field, lesson, vec, type TopicDefinition } from './types';
const MAMBA = 'https://arxiv.org/html/2312.00752v2';
export default {
  id: 'ssm',
  path: '/ssm-mamba',
  title: 'SSM 与 Mamba：过去怎样进入当前状态？',
  chapter: '第 32 章',
  intro:
    '从一个标量状态的递推开始，再让步长、写入与读出随输入变化。核对离散化、旧状态衰减，以及固定状态的历史碰撞。',
  question: '同一个问题：不回头读取全部历史，过去怎样进入当前状态',
  headline: '状态递推 → 选择性参数 → 离散化 → 逐步扫描 → 记忆衰减 → 与 Attention 对照。',
  why: {
    question: '不用 Attention，模型还能怎样记住前文？',
    points: [
      'Attention 每一步都回头读取全部历史，序列越长，计算和 KV 缓存越大（原书 32.1）。RNN 走另一条路：用一个“隐状态”压缩历史，每一步只需要上一步的状态和当前输入。',
      'SSM（状态空间模型）沿用这个思路；Mamba 再让状态的更新方式随输入变化：重要的输入多写进状态，不重要的少写。本页用一个标量状态手算这个过程。',
    ],
    analogy: {
      title: '用同样的注意力读每个字（32.3.2）',
      text: '传统 SSM 对 “The” 和 “fox” 用同样的参数，就像一个人用同样的注意力读每一个字，分不出重要和不重要。Mamba 的选择性参数让它可以按内容决定记住什么、忘掉什么。',
    },
    book: ['第 32.1、32.2.1 节 · 从 RNN 说起', BOOK.ch32],
  },
  input: [1, 0, 2, -1],
  inputNames: ['x₀', 'x₁', 'x₂', 'x₃'],
  fields: [
    field('cursor', '查看时间步 t', 3, 0, 3, 1, [3]),
    field('A', '连续衰减 A', -1, -2, 0, 0.25),
    field('delta', '基础步长 Δ', 0.5, 0.25, 2, 0.25),
    field('h0', '初始状态 h_init', 1, -2, 2, 0.25, [0, 3, 5]),
    choice(
      'selective',
      '参数模式',
      1,
      [
        { value: 0, label: '固定参数' },
        { value: 1, label: '输入相关' },
      ],
      [1, 2, 3, 4, 5],
    ),
  ],
  steps: [
    lesson({
      name: '状态递推',
      title: '把过去折进一个数，再读出输出',
      formula: String.raw`h_t=\bar A h_{t-1}+\bar Bx_t,\qquad y_t=C h_t`,
      explanation: (c) =>
        `本页的状态 h 只是一个数。每读一个输入，先把旧状态乘一个保留系数，再加上这次写入的量；输出由当前状态读出。初始 h = ${fmt(c.metric('初始 h₀'))}，固定参数时四步的状态依次为 ${vec(c.values('固定参数状态 h_t'))}。模型保存的是这一个聚合后的状态，不是历史列表。`,
      prompt: '新输入到来时，保存的是历史列表还是一个聚合状态？',
      takeaway: '状态是过去的压缩表示；读出时不能直接访问全部历史。',
      book: ['第 32.2.2 节 · 什么是 State Space Models', BOOK.ch32],
    }),
    lesson({
      name: '选择性参数',
      title: 'Mamba：步长 Δ、写入 B、读出 C 随输入变化',
      formula: String.raw`\Delta_t=\Delta\,softplus(0.5x_t),\ B_t=1+0.25x_t,\ C_t=1-0.1x_t`,
      explanation: (c) =>
        `输入相关模式下，每个时间步的 Δ、B、C 都由当前输入算出；当前四步的 Δ 为 ${vec(c.values('Δ_t（始终为正）'))}。Δ 越大，这一步越“关注当前输入”；Δ 越小，越“保持历史”（原书 32.3.3）。A 始终固定；这里的投影规则是本页的教学设定，不是 Mamba 的训练权重。`,
      prompt: '把输入改大，会同时影响哪些系数？',
      takeaway: '选择性来自参数对输入的依赖，不是预设的语义开关。',
      book: ['第 32.3.3 节 · Mamba 的解决方案：选择性 SSM', BOOK.ch32],
      source: ['Mamba Eq.1–4、Algorithm 2', MAMBA],
    }),
    lesson({
      name: '离散化',
      title: '离散化：Δ 同时决定留多少旧的、写多少新的',
      formula: String.raw`\bar A_t=e^{\Delta_tA},\qquad\bar B_t=\frac{e^{\Delta_t A}-1}{A}B_t`,
      explanation: (c) =>
        `连续形式的 A、B 要换成一步一步的系数：旧状态乘 $e^{\\Delta A}$，新输入乘 $\\bar B$。当前四步的保留系数为 ${vec(c.values('保留系数 exp(ΔA)'))}。A < 0 时，Δ 越大，旧状态保留得越少，新输入写得越多。本页用 expm1 计算，避免小数相减损失精度；A = 0 时取极限 ΔB。`,
      prompt: '为什么 Δ 变化时，新输入的写入系数也要跟着变？',
      takeaway: '离散化同时改变保留项和写入项，不能只改其中一个。',
      book: ['第 32.2.3 节 · 连续 vs 离散', BOOK.ch32],
      source: ['Mamba Eq.4（零阶保持）', MAMBA],
    }),
    lesson({
      name: '逐步扫描',
      title: '一步一步扫：每一步只用上一步的状态',
      formula: String.raw`h_t=\bar A_th_{t-1}+\bar B_tx_t`,
      explanation: (c) => {
        const states = c.values('已计算状态 h_t');
        return `看第 ${fmt(c.metric('当前时间步（从0开始）'))} 步：上一状态 ${fmt(c.metric('上一状态'))} × 保留 ${fmt(c.metric('保留系数'))} + 写入 ${fmt(c.metric('输入注入'))} = ${fmt(states[states.length - 1])}。拖动“查看时间步 t”，左图逐步显示状态，未来的状态还没算出来。改动第 2 个输入，只会影响第 2 步及之后的状态。`;
      },
      prompt: '如果改动第 2 个输入，之前的输出会被改变吗？',
      takeaway: '时间方向的依赖决定了哪些结果需要重算。',
      book: ['第 32.2.4 节 · SSM 的两种视角', BOOK.ch32],
    }),
    lesson({
      name: '记忆衰减',
      title: 'Δ 越大，旧状态忘得越快',
      formula: String.raw`A<0,\ \Delta\uparrow\ \Longrightarrow e^{\Delta A}\downarrow`,
      explanation: (c) =>
        `基础步长 Δ = ${fmt(c.metric('基础步长 Δ'))}。A < 0 时，保留系数 $e^{\\Delta A}$ 随 Δ 增大而变小：大 Δ 让旧状态衰减更快、更关注当前输入，小 Δ 更接近保留旧状态。原书 32.3.4 写成“小 Δ 快速遗忘”，与公式和 32.3.3 节相反，本页以公式为准。`,
      prompt: '旧状态衰减得更快，是否意味着当前输出一定更小？',
      takeaway: '保留系数只描述旧状态；新输入的写入量也在同时变化。',
      book: ['第 32.3.4 节 · 选择性机制的力量（本页已修正）', BOOK.ch32],
    }),
    lesson({
      name: '历史访问对照',
      title: '固定大小的状态：不同的历史可能变成同一个数',
      formula: String.raw`[a,c]\ \text{与}\ [a+\delta,c-\lambda\delta]\ \Rightarrow h_2\text{相同}`,
      explanation: (c) =>
        `表中两段不同的历史，用固定参数扫描后最终状态都是 ${fmt(c.metric('历史 A 最终 h'), 4)}，之后的计算再也分不清它们。因果 Attention 则每次直接读取全部历史，是另一种机制。最后先预测：只改变基础步长，旧状态的保留系数会不会变？`,
      prompt: '相同的最终状态能区分所有历史细节吗？',
      takeaway: 'Mamba 的选择性 SSM 更灵活，但固定大小的状态不代表能无损记住任意长历史。',
      book: ['第 32.8 节 · Transformer vs Mamba vs Hybrid 对比', BOOK.ch32],
    }),
  ],
  examples: [
    { name: '脉冲与变化', input: [1, 0, 2, -1] },
    { name: '单个脉冲', input: [1, 0, 0, 0] },
    { name: '零输入', input: [0, 0, 0, 0] },
  ],
  compute: ssm,
  prediction: {
    key: 'delta',
    target: 1,
    question: '只改变基础步长，旧状态保留系数会保持不变吗？',
    label: '旧状态保留系数',
  },
  sources: [
    { title: '原书第 32 章 · 后 Transformer 架构', url: BOOK.ch32 },
    { title: 'Mamba Eq.1–4、Algorithm 2', url: MAMBA },
  ],
  limits: [
    '标量选择性 SSM 机制示例；省略 Mamba 的卷积、门控、扩展投影与多维状态，不能称完整模型复现。',
    '固定参数历史碰撞只证明本示例映射的边界；不作为所有任务的架构优劣排序。',
    '序列扫描、额外状态空间和 Attention 历史访问分别讨论；没有真实硬件性能测量。',
  ],
  bookNotes: [
    '原书 32.3.4 写“对 The（冠词）：小 Δ → 快速遗忘”。按 $\\bar A = e^{\\Delta A}$（A < 0），小 Δ 让保留系数接近 1，旧状态保留更多；大 Δ 才让旧状态衰减更快。这与书中 32.3.3 节“Δ 很小时隐状态保持稳定”一致，本页以公式为准。',
  ],
} satisfies TopicDefinition;
