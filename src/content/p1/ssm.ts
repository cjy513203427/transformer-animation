import { ssm } from '../../math/p1/ssm';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'ssm',
  path: '/ssm-mamba',
  title: 'SSM 与 Mamba：过去怎样进入当前状态？',
  chapter: '第 32 章',
  intro:
    '从标量状态递推开始，再让步长、写入与读出随输入变化。核对离散化、旧状态衰减，以及固定状态的历史碰撞。',
  input: [1, 0, 2, -1],
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
    lesson(
      '状态递推',
      String.raw`h_t=\bar A h_{t-1}+\bar Bx_t,\qquad y_t=C h_t`,
      '先看固定系数如何把历史折进一个状态。输入行依次是四个时间步，每次输出通过当前状态计算。',
      '新输入到来时，保存的是历史列表还是一个聚合状态？',
      '状态是过去的压缩表示，读出不是直接访问全部历史。',
    ),
    lesson(
      '选择性参数',
      String.raw`\Delta_t=\Delta\,softplus(0.5x_t),\ B_t=1+0.25x_t,\ C_t=1-0.1x_t`,
      '输入相关模式让写入、读出和步长随当前输入变化。投影规则是本项目的教学设定，A 始终保持固定。',
      '把输入改大，会同时影响哪些系数？',
      '选择性来自参数对输入的依赖，而不是预设的语义开关。',
    ),
    lesson(
      '离散化',
      String.raw`\bar A_t=e^{\Delta_tA},\qquad\bar B_t=\frac{e^{\Delta_t A}-1}{A}B_t`,
      '采用零阶保持离散化，使用 expm1 避免小数相减损失。A=0 时用连续极限 ΔB。',
      '为什么 Δ 变化时，新输入的注入系数也要改变？',
      '连续参数需要离散化，不能只改保留系数而忽略输入项。',
    ),
    lesson(
      '逐步扫描',
      String.raw`h_t=\bar A_th_{t-1}+\bar B_tx_t`,
      '依次查看状态与输出。即使参数依赖当前输入，这一页的每个数仍可按上一状态重建；没有隐藏的模型服务。',
      '如果改第2个输入，之前的输出会被改变吗？',
      '时间方向的依赖约束决定哪些结果需要重算。',
    ),
    lesson(
      '记忆衰减',
      String.raw`A<0,\ \Delta\uparrow\ \Longrightarrow e^{\Delta A}\downarrow`,
      '比较旧状态保留系数。原书 §32.3.4 的“小 Δ 快速遗忘”与公式相反；这里以公式为准，小 Δ 更保留旧状态。',
      '更快衰减旧状态，是否意味着当前输出一定更小？',
      '保留系数只描述旧状态项，输入注入也会同时变化。',
    ),
    lesson(
      '历史访问对照',
      String.raw`[a,c]\ \text{与}\ [a+\delta,c-\lambda\delta]\ \Rightarrow h_2\text{相同}`,
      '固定参数下给出两段不同历史的相同最终状态。并列显示另一个算子——因果 Attention——的输出，先预测改变基础步长是否保持保留系数。',
      '相同最终状态能区分所有历史细节吗？',
      'Mamba 的选择性 SSM 更灵活，但固定大小状态不代表无损记住任意长历史。',
    ),
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
    { title: 'Mamba Eq.1–4、Algorithm2', url: 'https://arxiv.org/html/2312.00752v2' },
    {
      title: '原书第32章（修正 §32.3.4）',
      url: 'https://waylandz.com/llm-transformer-book/第32章-后Transformer架构-Mamba与混合模型/',
    },
  ],
  limits: [
    '标量选择性 SSM 机制示例；省略 Mamba 的卷积、门控、扩展投影与多维状态，不能称完整模型复现。',
    '固定参数历史碰撞只证明本示例映射的边界；不作为所有任务的架构优劣排序。',
    '序列扫描、额外状态空间和 Attention 历史访问分别讨论；没有真实硬件性能测量。',
  ],
} satisfies TopicDefinition;
