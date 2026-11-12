import { attentionSymbols } from './notation/attention';
import { chapterUrl } from './book';
import { T } from '../i18n';
export const steps = [
  {
    title: T('投影'),
    symbol: 'Q, K, V',
    symbols: attentionSymbols[0],
    formula: 'Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V',
    shape: '[3, 2] × [2, 2] → [3, 2]',
    question: T('同一份输入，为什么要变成三个矩阵？'),
    explanation: T(
      '每个 token 的隐藏状态 X 分别乘上三个参数矩阵，得到 Query、Key 和 Value。Q 与 K 用来算分数，V 提供最后要汇总的内容。',
    ),
    caution: T(
      'Q/K/V 是本次前向的激活。$W_Q$/$W_K$/$W_V$ 是固定参数；播放计算不会训练或更新参数。',
    ),
  },
  {
    title: T('打分'),
    symbol: 'S',
    symbols: attentionSymbols[1],
    formula: 'S=\\frac{QK^\\top}{\\sqrt{d_k}}',
    shape: T('[3, 2] × [2, 3] → [3, 3]，$d_k = 2$'),
    question: T('矩阵里的一个分数是怎样算出来的？'),
    explanation: T(
      '取 Q 的一行和 K 的一行，逐维相乘再相加，最后除以 $\\sqrt{2}$。S 的行对应 Query，列对应 Key。点击一个格子核对这一次点积。',
    ),
    caution: T(
      '点积同时受方向和模长影响，不等于余弦相似度。缩放对照只是教学开关，标准公式使用 $1/\\sqrt{d_k}$。',
    ),
  },
  {
    title: T('屏蔽'),
    symbol: 'S + M',
    symbols: attentionSymbols[2],
    formula: T(
      'M_{ij}=\\begin{cases}-\\infty & j>i\\text{ 且开启因果 Mask}\\\\0&\\text{其他}\\end{cases}',
    ),
    shape: '[3, 3] + [3, 3] → [3, 3]',
    question: T('怎样让当前位置看不到未来？'),
    explanation: T(
      '因果 Mask 将未来 Key 的分数设为 −∞，再交给 Softmax。关闭时所有位置都可见；开启时每行只允许访问自身与更早的位置。',
    ),
    caution: T(
      'Mask 必须在 Softmax 之前应用。若一行没有任何有效 Key，此实现约定权重与输出为零，避免 NaN。',
    ),
  },
  {
    title: T('归一化'),
    symbol: 'P',
    symbols: attentionSymbols[3],
    formula:
      '\\begin{gathered}P_{ij}=\\frac{\\exp\\big((S+M)_{ij}-m_i\\big)}{\\sum_k \\exp\\big((S+M)_{ik}-m_i\\big)}\\\\[0.5em] m_i=\\max_j\\,(S+M)_{ij}\\end{gathered}',
    shape: T('[3, 3] → [3, 3]，沿 Key 轴逐行计算'),
    question: T('为什么改一个分数，整行权重都可能变化？'),
    explanation: T(
      '每个 Query 都有自己的归一化分母。可见 Key 的权重之和为 1；被屏蔽的位置权重为 0。减去行最大值保持结果不变，同时避免指数溢出。',
    ),
    caution: T(
      '这里是“取用各 Value 的比例”，不是预测下一个 token 的词表概率，也不是沿列做 Softmax。',
    ),
  },
  {
    title: T('加权求和'),
    symbol: 'O',
    symbols: attentionSymbols[4],
    formula: 'O=PV,\\qquad O_i=\\sum_j P_{ij}V_j',
    shape: '[3, 3] × [3, 2] → [3, 2]',
    question: T('权重不变，输出也一定不变吗？'),
    explanation: T(
      '对每个 Query，用这一行的 P 分别乘各个 Value，再逐维相加得到输出。只干预 V 时，Q、K、S 和 P 保持不变，O 却可能改变。',
    ),
    caution: T('如果被修改的 Value 在当前行被完全屏蔽，其权重为 0，那么当前输出也不受影响。'),
  },
] as const;

export const sources = [
  {
    title: T('主地图：原书第 3 章 · Transformer 全景图'),
    url: chapterUrl('第03章-Transformer全景图/'),
  },
  {
    title: T('原书第 10 章 · QKV 的三个角色'),
    url: chapterUrl('第10章-QKV到底是什么-Attention的三个主角/'),
  },
  {
    title: T('原书第 11 章 · 多头注意力'),
    url: chapterUrl('第11章-Multi-Head-Attention-多视角理解/'),
  },
  {
    title: T('原书第 12 章 · 输出的本质'),
    url: chapterUrl('第12章-QKV输出的本质/'),
  },
  { title: 'Attention Is All You Need · §3.2', url: 'https://arxiv.org/html/1706.03762v7#S3.SS2' },
];
