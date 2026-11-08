import { normalization } from '../../math/p1/normalization';
import { choice, field, lesson, type TopicDefinition } from './types';
const ln = 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html',
  rms = 'https://arxiv.org/html/1910.07467v1';
export default {
  id: 'normalization',
  path: '/normalization',
  title: '归一化：缩放和概率是一回事吗？',
  chapter: '第 6 章',
  intro:
    '手算四维向量的均值、方差、RMS 与概率。对照常量和接近零的输入，再验证整行平移会改变哪些结果。',
  input: [1, 2, 3, 4],
  fields: [
    choice('eps', 'ε', 1e-5, [
      { value: 1e-5, label: '0.00001' },
      { value: 0.01, label: '0.01' },
      { value: 1, label: '1（放大边界）' },
    ]),
    field('gain', 'γ₀', 1, -2, 2, 0.25, [3]),
    field('bias', 'β₀', 0, -2, 2, 0.25, [3]),
    field('temperature', '温度 τ', 1, 0.25, 4, 0.25, [4]),
    choice(
      'mask',
      '候选屏蔽',
      0,
      [
        { value: 0, label: '不屏蔽' },
        { value: 1, label: '屏蔽最后一项' },
        { value: 2, label: '全部屏蔽' },
      ],
      [4, 5],
    ),
    field('shift', '已执行平移 c', 0, -3, 3, 0.25, [5]),
  ],
  steps: [
    lesson(
      '统计轴',
      String.raw`\mu=\frac1D\sum_i x_i,\quad\sigma^2=\frac1D\sum_i(x_i-\mu)^2`,
      '一行表示一个 token 的隐藏向量，沿特征轴计算统计量，不跨 token 求平均。方差除以 D，不是统计样本估计中的 D−1。',
      '改变另一个 token，需要重算这一行的统计量吗？',
      '每个 token 独立统计特征，同层共享归一化参数。',
    ),
    lesson(
      'LayerNorm',
      String.raw`z_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}`,
      '先减均值，再除以共同分母。ε 放在根号内，保证常量输入也可计算；ε 大时，归一化后的方差明显小于 1。',
      '常量输入减去均值以后是什么？为什么方差不一定等于 1？',
      '均值为 0、方差近似 1，描述的是仿射之前且非退化的核心结果。',
    ),
    lesson(
      'RMSNorm',
      String.raw`r_i=\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}`,
      'RMSNorm 不做中心化。负坐标可以保留为负，均值通常不为零；同样不把向量变成概率。',
      '对所有坐标加相同常数，RMS 分母会怎样变化？',
      'RMSNorm 使用均方根控制尺度，省去减均值。',
    ),
    lesson(
      '仿射参数',
      String.raw`y_i^{LN}=\gamma_i z_i+\beta_i,\qquad y_i^{RMS}=\gamma_i r_i`,
      '真实层有逐特征参数。本例只编辑第 0 维 γ、β，其他维度用 1、0；RMS 分支按本例定义不加 β。参数改变后，输出均值和方差不再受核心归一化的数值限制。',
      '把 γ₀ 设为 0，哪个坐标还会依赖输入？',
      '归一化统计量来自输入，γ、β 是参数；前向不会训练参数。',
    ),
    lesson(
      'Softmax',
      String.raw`p_i=\frac{\exp(s_i-m)}{\sum_j\exp(s_j-m)},\quad s_i=x_i/\tau+M_i`,
      '这一步把四个数字重新解释为候选分数。沿候选轴做指数与归一化，合法项非负且和为 1。Softmax 的轴语义与隐藏特征归一化不同。',
      '减最大值为什么不改概率？屏蔽应在归一化前还是之后？',
      'Softmax 构造概率，LayerNorm 与 RMSNorm 调整隐藏激活的尺度。',
    ),
    lesson(
      '平移对照',
      String.raw`LN(x+c)=LN(x),\quad softmax(x+c)=softmax(x)`,
      '先预测 RMS 核心结果会不会变化，再执行共同平移。并列查看 LN、RMS 与 Softmax，并用数值容差核对不变性质。',
      '为什么共同平移会被 LN 消去，却通常留在 RMS 中？',
      '固定 ε 时 LN 平移不变；正倍缩放则会受 ε 影响，不能泛称任意缩放完全不变。',
    ),
  ],
  examples: [
    { name: '递增向量', input: [1, 2, 3, 4] },
    { name: '常量边界', input: [2, 2, 2, 2] },
    { name: '接近零', input: [-0.001, 0, 0.001, 0.002] },
  ],
  compute: normalization,
  prediction: {
    key: 'shift',
    target: 2,
    question: '只做共同平移，RMS 核心结果会保持不变吗？',
    label: 'RMS 核心输出',
  },
  sources: [
    { title: 'LayerNorm 原论文 §3', url: 'https://arxiv.org/html/1607.06450v1' },
    { title: 'PyTorch LayerNorm：方差与 ε 约定', url: ln },
    { title: 'RMSNorm 原论文 §4', url: rms },
    {
      title: 'PyTorch Softmax',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softmax.html',
    },
    {
      title: '原书第6章',
      url: 'https://waylandz.com/llm-transformer-book/第06章-LayerNorm与Softmax-数字的缩放与概率化/',
    },
  ],
  limits: [
    'ε 使用明确的教学选项，不声称是所有模型的默认值。',
    '全部屏蔽时不存在合法概率分布；本工具返回零向量，避免 NaN。',
    '没有 BatchNorm、运行均值或训练更新；小向量不表示真实语义。',
  ],
} satisfies TopicDefinition;
