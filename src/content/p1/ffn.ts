import { formatNumber as fmt } from '../../math/attention';
import { ffn } from '../../math/p1/ffn';
import { BOOK } from '../book';
import { choice, field, lesson, vec, type TopicDefinition } from './types';
const row = (values: number[], width: number, i: number) =>
  values.slice(i * width, (i + 1) * width);
export default {
  id: 'ffn',
  path: '/ffn-residual',
  title: 'FFN 与残差：一行向量怎样被变换？',
  chapter: '第 7、13 章',
  intro:
    '四个 token、每个两维：经过升维、ReLU 和降维，再加回原输入。切换训练与推理，查看固定种子的 Dropout 掩码。',
  question: '同一个问题：Attention 读完上下文之后，每个位置还要怎样加工',
  headline: '升维 → ReLU → 降维 → 残差相加 → Dropout → 运行对照。',
  why: {
    question: '为什么每个 Block 里还要有 FFN、残差和 Dropout？',
    points: [
      'Attention 让各位置互相读取信息，本质上是一次加权汇总。FFN（前馈网络）接着对每个位置单独加工：先升维，经过非线性，再降回原维度。原书第 7 章的说法是：神经网络从输入里提取特征，再据此做判断。',
      '网络一深，信息和梯度都要穿过很多层。残差连接让输入“跳过”这一层，直接加到输出上；Dropout 在训练时随机关掉一部分激活，防止模型过度依赖某几个特征。',
    ],
    analogy: {
      title: '高速公路与轮流休假（13.2、13.3.3）',
      text: '残差像一条绕开收费站的直道：这一层还没学好，原信息也能原样通过。Dropout 像公司每天随机安排一些员工休假：没人能独揽全部工作，团队整体更稳。',
    },
    book: ['第 13.1 节 · 回顾 Transformer Block 的结构', BOOK.ch13],
  },
  input: [1, 2, -1, 1, 2, -1, -2, -1],
  inputNames: ['t0·d0', 't0·d1', 't1·d0', 't1·d1', 't2·d0', 't2·d1', 't3·d0', 't3·d1'],
  fields: [
    choice(
      'nonlinear',
      '非线性',
      1,
      [
        { value: 1, label: 'ReLU' },
        { value: 0, label: '旁路线性' },
      ],
      [1, 2, 5],
    ),
    field('dropout', '丢弃率 p', 0.5, 0, 0.75, 0.25, [4, 5]),
    choice(
      'training',
      '运行模式',
      1,
      [
        { value: 1, label: '训练' },
        { value: 0, label: '推理' },
      ],
      [4, 5],
    ),
  ],
  steps: [
    lesson({
      name: '升维',
      title: '每个位置单独升维：2 维变成 3 维',
      formula: String.raw`H=XW_1,\quad[4,2][2,3]\to[4,3]`,
      explanation: (c) => {
        const x = c.values('X [4, 2]'),
          h = c.values('H = XW₁ [4, 3]');
        return `X 的每一行是一个 token（本例 4 个，每个 2 维）。同一个 $W_1$ 乘到每一行上，把 2 维变成 3 维的中间特征 H。以 t0 为例：${vec(row(x, 2, 0))} 乘 $W_1$ 得到 ${vec(row(h, 3, 0))}。各行之间互不读取，这是 FFN 与 Attention 的分工。真实模型的中间维度远大于 3，本例只取最小可手算的尺寸，并省略偏置。`;
      },
      prompt: '修改 t1 的输入，会不会直接改变 t0 的 FFN 输出？',
      takeaway: 'FFN 对每个位置单独加工，参数在各位置共享；跨位置交流由 Attention 负责。',
      book: ['第 7.7.1 节 · Feed Forward Network', BOOK.ch07],
    }),
    lesson({
      name: '非线性',
      title: 'ReLU：负数清零，正数照常通过',
      formula: String.raw`U=\operatorname{ReLU}(H)=\max(0,H)`,
      explanation:
        '对 H 的每个数做 ReLU：负数变成 0，正数保持不变。有了这一步，FFN 才能表示“折线”，而不只是直线。关掉 ReLU 时，两次矩阵乘法可以合成一次（最后一步会核对），等于白加一层。坐标 d0–d2 没有预先分配的含义。',
      prompt: '为什么两次矩阵乘法之间要加入非线性？',
      takeaway: '没有非线性，多次线性变换仍然等于一次线性变换。',
      book: ['第 7.6.3 节 · 激活函数', BOOK.ch07],
    }),
    lesson({
      name: '降维',
      title: '降回原来的维度，才能和输入相加',
      formula: String.raw`F=UW_2,\quad[4,3][3,2]\to[4,2]`,
      explanation:
        '再乘 $W_2$，把 3 维中间特征投影回 2 维。左图的圆点是原输入 X，方块是 FFN 输出 F(X)，同一个 token 用线连起来。形状回到 [4, 2]，下一步才能和输入逐项相加。二维图只是教学数值，不代表真实语义空间。',
      prompt: '残差相加前，两条支路必须满足什么形状条件？',
      takeaway: 'FFN 的中间维度可以更大，但输出必须回到残差流的维度。',
      book: ['第 7.6.2 节 · 矩阵维度变化', BOOK.ch07],
    }),
    lesson({
      name: '残差与梯度',
      title: '残差：把原输入直接加回来',
      formula: String.raw`Y=X+F(X),\qquad J_Y=I+J_F`,
      explanation: (c) => {
        const x = c.values('原 X'),
          f = c.values('分支 F(X)'),
          y = c.values('X + F(X)');
        return `输出 = 原输入 + FFN 结果。以 t0 为例：${vec(row(x, 2, 0))} + ${vec(row(f, 2, 0))} = ${vec(row(y, 2, 0))}，左图的方块就是相加后的位置。求导时，这条直道贡献一个单位矩阵 I（表中 t0 的 Jacobian = I + $J_F$），梯度因此至少有一条不经过 FFN 的路。其他分支仍可能把它抵消，所以残差不保证梯度永不消失。`;
      },
      prompt: '为什么有一条恒等路径，仍不能保证总梯度永远非零？',
      takeaway: '残差保留一条直达的信息与梯度通路，不是把 FFN 再算一遍。',
      book: ['第 13.2 节 · 残差连接：信息的“高速公路”', BOOK.ch13],
    }),
    lesson({
      name: 'Dropout',
      title: '训练时随机关掉一部分，推理时全部保留',
      formula: String.raw`D(F)_i=\frac{m_iF_i}{1-p},\quad m_i\sim Bernoulli(1-p)`,
      explanation: (c) =>
        c.config.training
          ? `当前是训练模式，丢弃率 p = ${fmt(c.config.dropout)}：被抽中的 FFN 输出置 0，留下的除以 1 − p = ${fmt(1 - c.config.dropout)}，让平均大小保持不变。被丢弃的是激活，不是参数；只有 FFN 分支会被丢弃，残差直道不受影响。掩码由固定种子生成，真实训练每次前向都会重新抽。`
          : '当前是推理模式：Dropout 不起作用，FFN 输出原样加回。训练时留下的值已经除过 1 − p，所以推理时不用再乘任何系数（inverted Dropout）。被丢弃的从来是激活，不是参数。',
      prompt: '为什么推理时不用再乘 1 − p？被屏蔽的是参数还是激活？',
      takeaway: '本例用 inverted Dropout：训练时补偿，推理时直接通过。',
      book: ['第 13.3.4 节 · Dropout 的数学', BOOK.ch13],
    }),
    lesson({
      name: '运行对照',
      title: '对照：线性能合并，非线性不能；训练与推理不同',
      formula: String.raw`XW_1W_2=X(W_1W_2)`,
      explanation:
        '表中并列 $X(W_1W_2)$ 与当前 F(X)：关掉 ReLU 时两者相等。再先预测：切到推理模式，残差输出会不会变？关掉 Dropout 后通常会变；若 p = 0，或 FFN 分支恰好为 0，则可能不变。',
      prompt: '非线性、Dropout 和残差各自解决哪个问题？',
      takeaway: '比较输出前先写明运行模式；训练时的随机前向与推理输出不是同一个条件。',
      book: ['第 13.5 节 · 残差连接 + Dropout 的协同作用', BOOK.ch13],
    }),
  ],
  examples: [
    { name: '四个点', input: [1, 2, -1, 1, 2, -1, -2, -1] },
    { name: '全零', input: Array(8).fill(0) },
  ],
  compute: ffn,
  prediction: {
    key: 'training',
    target: 0,
    question: '切换到推理，当前残差输出会保持不变吗？',
    label: '残差输出',
  },
  sources: [
    { title: '原书第 7 章 · 神经网络层', url: BOOK.ch07 },
    { title: '原书第 13 章 · 残差连接与 Dropout', url: BOOK.ch13 },
    { title: 'Transformer §3.3、§5.4', url: 'https://arxiv.org/html/1706.03762v7' },
    {
      title: 'PyTorch Dropout',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html',
    },
  ],
  limits: [
    '只演示位置独立的两层 ReLU FFN，省略偏置、归一化和完整 Transformer。',
    'Jacobian 以行输入坐标、列输出坐标展示；ReLU 在零点的导数约定为 0。',
    '没有执行反向传播或优化；Dropout 掩码只作用于 FFN 分支。',
  ],
  bookNotes: [
    '原书 7.7.1 节的图里出现“春 = 30%、江 = 70%”的概率，书中注明那来自最后的 LM Head，不是 FFN；本页的 FFN 只输出向量，不产生概率。',
    '原书 13.4 节比较 Pre-Norm 与 Post-Norm；本页只演示 FFN 子层本身，省略 LayerNorm 的位置。',
  ],
} satisfies TopicDefinition;
