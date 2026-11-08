import { ffn } from '../../math/p1/ffn';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'ffn',
  path: '/ffn-residual',
  title: 'FFN 与残差：一行向量怎样被变换？',
  chapter: '第 7、13 章',
  intro:
    '四个二维点经过升维、ReLU 和降维，再加回原输入。切换训练与推理，查看固定种子的 Dropout 掩码。',
  input: [1, 2, -1, 1, 2, -1, -2, -1],
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
    lesson(
      '升维',
      String.raw`H=XW_1,\quad[4,2][2,3]\to[4,3]`,
      '同一个 FFN 对每个 token 独立作用，参数在各行间共享。本例省略偏置，先从二维投影到三维。',
      '修改一行输入，会不会直接改变其他行的 FFN 输出？',
      'FFN 沿特征做变换，Attention 才承担跨 token 的信息汇合。',
    ),
    lesson(
      '非线性',
      String.raw`U=\operatorname{ReLU}(H)=\max(0,H)`,
      '逐坐标保留正值、把负值置零。开关可以直接比较带非线性与纯线性支路；它不把每个坐标预先分配成语义标签。',
      '为什么两次矩阵乘法之间要加入非线性？',
      '没有非线性，多次线性投影仍可合成一个线性投影。',
    ),
    lesson(
      '降维',
      String.raw`F=UW_2,\quad[4,3][3,2]\to[4,2]`,
      '把三维中间特征重新投影到输入维度，让残差能够逐项相加。二维图只是教学数值，不是高维语义的证明。',
      '残差相加前，两条支路必须满足什么形状条件？',
      'FFN 的中间维度与残差流维度是不同的量。',
    ),
    lesson(
      '残差与梯度',
      String.raw`Y=X+F(X),\qquad J_Y=I+J_F`,
      '恒等支路让原输入直接进入输出，局部 Jacobian 中出现 I。残差不保证所有层的梯度都不消失；其他分支可能抵消。',
      '为什么有一条恒等路径，仍不能保证总梯度永远非零？',
      '残差保留直接路径，不能代替对完整梯度的分析。',
    ),
    lesson(
      'Dropout',
      String.raw`D(F)_i=\frac{m_iF_i}{1-p},\quad m_i\sim Bernoulli(1-p)`,
      '训练时屏蔽分支激活并补偿保留项；推理时直接使用原分支。这里固定种子，真实训练通常每次前向重新采样。',
      '为什么推理不用再乘 1−p？被屏蔽的是参数还是激活？',
      '本例采用 inverted Dropout，推理时无需额外缩放。',
    ),
    lesson(
      '运行对照',
      String.raw`XW_1W_2=X(W_1W_2)`,
      '并列核对线性两种计算次序，先预测切到推理后残差输出是否不变，再实际执行。零分支或 p=0 是可能不变的边界。',
      '非线性、Dropout 和恒等支路各自解决哪个计算问题？',
      '不同运行模式需要明确；不能把随机训练前向与推理输出直接当作同一条件。',
    ),
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
} satisfies TopicDefinition;
