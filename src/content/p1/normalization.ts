import { formatNumber as fmt } from '../../math/attention';
import { normalization } from '../../math/p1/normalization';
import { BOOK } from '../book';
import { choice, field, lesson, type TopicDefinition } from './types';
import { T } from '../../i18n';
const ln = 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html',
  rms = 'https://arxiv.org/html/1910.07467v1';
export default {
  id: 'normalization',
  path: '/normalization',
  title: T('归一化：缩放和概率是一回事吗？'),
  chapter: T('第 6 章'),
  intro: T(
    '从原书 6.2.4 的 [22, 5, 6, 8] 出发，手算均值、方差、LayerNorm、RMSNorm 与 Softmax，再验证整行平移会改变哪些结果。',
  ),
  question: T('同一个问题：一行数字怎样被整理到合适的尺度，或变成概率'),
  headline: T('统计 → LayerNorm → RMSNorm → γ、β → Softmax → 平移对照。'),
  why: {
    question: T('为什么需要 LayerNorm 和 Softmax？'),
    points: [
      T(
        '数字经过一层又一层计算，可能越来越大或越来越小：太大会溢出，太小会丢精度。LayerNorm 在每个 Block 里把一个 token 的向量拉回统一的尺度，后面的层更容易处理。',
      ),
      T(
        'Softmax 解决另一件事：把一组可正可负的分数，变成“每项在 0 到 1 之间、加起来等于 1”的概率。Attention 用它分配取用比例，模型最后也用它给下一个 token 分配概率。',
      ),
      T('默认数字就是原书 6.2.4 的手算例子，可以直接与书中结果 [1.71, −0.76, −0.62, −0.33] 对照。'),
    ],
    analogy: {
      title: T('换一把合适的标尺（6.2.1）'),
      text: T(
        '同样是 70 公斤，用克记是 70000，用吨记是 0.07。数没有变，只是换了标尺。LayerNorm 给每个 token 的向量换一把合适的标尺。',
      ),
    },
    book: [T('第 6.1、6.2.1 节 · 为什么需要这两个工具'), BOOK.ch06],
  },
  input: [22, 5, 6, 8],
  inputRange: [-32, 32],
  inputNames: ['d0', 'd1', 'd2', 'd3'],
  fields: [
    choice('eps', 'ε', 1e-5, [
      { value: 1e-5, label: '0.00001' },
      { value: 0.01, label: '0.01' },
      { value: 1, label: T('1（放大边界）') },
    ]),
    field('gain', 'γ₀', 1, -2, 2, 0.25, [3]),
    field('bias', 'β₀', 0, -2, 2, 0.25, [3]),
    field('temperature', T('温度 τ'), 1, 0.25, 4, 0.25, [4]),
    choice(
      'mask',
      T('候选屏蔽'),
      0,
      [
        { value: 0, label: T('不屏蔽') },
        { value: 1, label: T('屏蔽最后一项') },
        { value: 2, label: T('全部屏蔽') },
      ],
      [4, 5],
    ),
    field('shift', T('已执行平移 c'), 0, -3, 3, 0.25, [5]),
  ],
  steps: [
    lesson({
      name: T('统计轴'),
      title: T('先在这一行里算平均值和分散程度'),
      formula: String.raw`\mu=\frac1D\sum_i x_i,\quad\sigma^2=\frac1D\sum_i(x_i-\mu)^2`,
      explanation: (c) =>
        T(
          '这四个数是同一个 token 的四个特征维 d0–d3。LayerNorm 只在这一行里统计，不和别的 token 混算。平均值 μ = 四个数之和 ÷ {#0} = {#1}；每个数减去 μ、平方、再平均，得到方差 σ² = {#2}。这里除以 D = {#3}，不是统计学估计样本方差时用的 D − 1。',
          c.input.length,
          fmt(c.metric('μ')),
          fmt(c.metric(T('方差（除以 D）'))),
          c.input.length,
        ),
      prompt: T('改动另一个 token 的向量，需要重算这一行的 μ 和 σ² 吗？'),
      takeaway: T('每个 token 只用自己的特征算统计量；同一层的所有 token 共用同一组 γ、β。'),
      book: [T('第 6.2.4 节 · 手算示例：均值与方差'), BOOK.ch06],
    }),
    lesson({
      name: 'LayerNorm',
      title: T('减去平均值，再除以标准差'),
      formula: String.raw`z_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}`,
      explanation: (c) =>
        T(
          '每个数先减去 μ，再除以同一个分母 $\\sqrt{\\sigma^2+\\epsilon}$ = {#0}。结果 z 的均值是 {#1}、方差是 {#2}：数被拉回 0 附近，但谁大谁小的顺序没有变。ε 是防止除以 0 的小数；把 ε 调到 1，方差会明显小于 1。',
          fmt(c.metric(T('分母'))),
          fmt(c.metric(T('z 均值'))),
          fmt(c.metric(T('z 方差'))),
        ),
      prompt: T('常量输入减去均值后是什么？为什么 z 的方差不一定正好是 1？'),
      takeaway: T('LayerNorm 让一行数的均值为 0、方差约为 1；数之间的相对位置不变。'),
      book: [T('第 6.2.3 节 · 缩放规则'), BOOK.ch06],
    }),
    lesson({
      name: 'RMSNorm',
      title: T('RMSNorm：不减平均值，只除以均方根'),
      formula: String.raw`r_i=\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}`,
      explanation: (c) =>
        T(
          'RMSNorm 省掉“减平均值”这一步，直接把每个数除以均方根 $\\sqrt{\\overline{x^2}+\\epsilon}$ = {#0}。所以 0 仍在 0、负数仍是负数，结果的均值是 {#1}，一般不为 0。不少开源模型（例如 LLaMA）用它代替 LayerNorm；原书第 6 章只讲 LayerNorm，这一步是补充对照。',
          fmt(c.metric(T('RMS 分母'))),
          fmt(c.metric(T('r 均值'))),
        ),
      prompt: T('给所有坐标加同一个常数，RMS 分母会怎样变化？'),
      takeaway: T('RMSNorm 只控制尺度、不做中心化，比 LayerNorm 少一步。'),
      book: [T('第 6.2 节 · LayerNorm（RMSNorm 为补充对照）'), BOOK.ch06],
      source: [T('RMSNorm 原论文 §4'), rms],
    }),
    lesson({
      name: T('仿射参数'),
      title: T('再乘 γ、加 β：让模型自己决定最终范围'),
      formula: String.raw`y_i^{LN}=\gamma_i z_i+\beta_i,\qquad y_i^{RMS}=\gamma_i r_i`,
      explanation: T(
        '归一化之后，每个特征维还会乘一个 γ、加一个 β，也就是原书公式末尾的“× γ + β”。它们是训练出来的参数，让模型可以把某些维度重新放大或移动。本例只开放第 0 维的 γ₀、β₀，其余维度固定为 1 和 0；RMSNorm 按常见定义只乘 γ、不加 β。',
      ),
      prompt: T('把 γ₀ 设为 0，LN 输出的第 0 维还和输入有关吗？'),
      takeaway: T('统计量 μ、σ 来自输入；γ、β 是参数，前向计算时不会被更新。'),
      book: [T('第 6.2.3 节 · 缩放规则（γ 与 β）'), BOOK.ch06],
    }),
    lesson({
      name: 'Softmax',
      title: T('把同一组数当作分数，变成加起来为 1 的概率'),
      formula: String.raw`p_i=\frac{\exp(s_i-m)}{\sum_j\exp(s_j-m)},\quad s_i=x_i/\tau+M_i`,
      explanation: (c) => {
        const m = c.metric(T('最大值 m'));
        const head = T(
          '现在换一个用途：把这几个数当作候选项的分数。Softmax 对每个分数取 e 的指数，再除以总和；温度 τ 先除分数（原书 6.5 节）。',
        );
        if (!Number.isFinite(m))
          return T('{#0}当前所有候选都被屏蔽，没有合法分布，本页返回全零。', head);
        return T(
          '{#0}为了不溢出，先减去最大值 m = {#1}，概率不变；当前概率之和为 {#2}。点“书中 6.3.4”预设，[3.01, 0.09, 2.48, 1.95] 会得到书中的 50.28%、2.71%、29.59%、17.42%。',
          head,
          fmt(m),
          fmt(c.metric(T('概率和')), 6),
        );
      },
      prompt: T('减去最大值为什么不改变概率？屏蔽应该在归一化之前还是之后加？'),
      takeaway: T('Softmax 构造概率；LayerNorm、RMSNorm 只调整向量尺度，不产生概率。'),
      book: [T('第 6.3.4 节 · Softmax 手算示例'), BOOK.ch06],
    }),
    lesson({
      name: T('平移对照'),
      title: T('整行加同一个数：谁变，谁不变？'),
      formula: String.raw`LN(x+c)=LN(x),\quad softmax(x+c)=softmax(x)`,
      explanation: (c) =>
        T(
          '给这一行每个数都加上 c = {#0}。LayerNorm 第一步就减去平均值，平移被抵消；Softmax 的分子、分母同乘 $e^{c}$，也被抵消。RMSNorm 没有减平均值，结果通常会变。先预测，再执行，用数值容差核对。',
          fmt(c.config.shift),
        ),
      prompt: T('为什么共同平移会被 LayerNorm 消去，却通常留在 RMSNorm 中？'),
      takeaway: T(
        '固定 ε 时，LayerNorm 和 Softmax 对整行平移不变；正倍缩放会受 ε 影响，不能说任意缩放都完全不变。',
      ),
      book: [T('第 6.6.1 节 · 核心概念对比（平移性质为本页补充）'), BOOK.ch06],
    }),
  ],
  examples: [
    { name: T('书中 6.2.4：[22, 5, 6, 8]'), input: [22, 5, 6, 8], book: true },
    {
      name: T('书中 6.3.4：[3.01, 0.09, 2.48, 1.95]'),
      input: [3.01, 0.09, 2.48, 1.95],
      book: true,
    },
    { name: T('常量边界'), input: [2, 2, 2, 2] },
    { name: T('接近零'), input: [-0.001, 0, 0.001, 0.002] },
  ],
  compute: normalization,
  prediction: {
    key: 'shift',
    target: 2,
    question: T('只做共同平移，RMS 核心结果会保持不变吗？'),
    label: T('RMS 核心输出'),
  },
  sources: [
    { title: T('原书第 6 章 · LayerNorm 与 Softmax'), url: BOOK.ch06 },
    { title: T('LayerNorm 原论文 §3'), url: 'https://arxiv.org/html/1607.06450v1' },
    { title: T('PyTorch LayerNorm：方差与 ε 约定'), url: ln },
    { title: T('RMSNorm 原论文 §4'), url: rms },
    {
      title: 'PyTorch Softmax',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softmax.html',
    },
  ],
  limits: [
    T('ε 使用明确的教学选项，不声称是所有模型的默认值。'),
    T('全部屏蔽时不存在合法概率分布；本工具返回零向量，避免 NaN。'),
    T('没有 BatchNorm、运行均值或训练更新；小向量不表示真实语义。'),
  ],
  bookNotes: [
    T(
      '第 6.2.2 节把 22、5、6、8 分别标成“小、沈、阳、邀”四个字，读起来像在四个 token 之间做归一化；这与 6.2.6 节“每个 token 的向量单独归一化”不一致。本页把这四个数当作同一个 token 的 4 个特征维 d0–d3。',
    ),
    T(
      '书中 6.2.4 的手算保留两位小数（σ² ≈ 47.19，结果 [1.71, −0.76, −0.62, −0.33]）；本页 ε = 0.00001 时显示 [1.711, −0.764, −0.619, −0.328]，与书中配图的 4 位小数一致。',
    ),
    T('RMSNorm 不在原书第 6 章内；本页把它作为 LayerNorm 的对照补充，公式按 RMSNorm 原论文。'),
  ],
} satisfies TopicDefinition;
