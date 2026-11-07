import { BOOK_URL } from './topics';

export type Insertion = 'add' | 'rope' | 'alibi';

export const positionSteps: {
  name: string;
  title: string;
  insertion: Insertion[];
  takeaway: string;
  prompt: string;
  formula: string;
  chapter: [string, string];
}[] = [
  {
    name: '位置盲',
    title: '没有位置信息，Attention 分不清“我爱你”和“你爱我”',
    insertion: [],
    takeaway: 'Attention 只看“有哪些向量”，不看“谁排第几”；顺序必须另外加进去。',
    prompt: '为什么两句话里“我”的输出完全一样？因果 Mask 能不能单独解决这个问题？',
    formula:
      'O_i=\\sum_j \\operatorname{softmax}_j\\!\\left(\\frac{x_i\\cdot x_j}{\\sqrt d}\\right)x_j,\\qquad x_t=E[\\text{token}_t]',
    chapter: ['第 5.1 节 · 为什么需要位置信息', '第05章-Positional-Encoding-给文字加位置/'],
  },
  {
    name: '正弦表',
    title: '正弦位置编码：每个位置一组不同快慢的“指针”',
    insertion: ['add'],
    takeaway:
      '每对维度是一根转速不同的指针：低维转得快，高维转得慢，合起来给每个位置一组独特的“门牌号”。',
    prompt: '第 $i$ 对维度每前进一个位置转多少？为什么同一对里的 $\\sin$ 和 $\\cos$ 频率相同？',
    formula:
      'PE_{(\\mathrm{pos},2i)}=\\sin\\frac{\\mathrm{pos}}{10000^{2i/d_{\\mathrm{model}}}},\\qquad PE_{(\\mathrm{pos},2i+1)}=\\cos\\frac{\\mathrm{pos}}{10000^{2i/d_{\\mathrm{model}}}}',
    chapter: ['第 5.3 节 · 正弦位置编码', '第05章-Positional-Encoding-给文字加位置/'],
  },
  {
    name: '加到输入',
    title: '相加：位置在进入第一个 Block 之前加到 X 上',
    insertion: ['add'],
    takeaway: '加法发生在投影之前：X 变了，Q、K、V 三条支路都带上了位置。',
    prompt: '两个“L”的嵌入相同，为什么相加后不同？位置信息会不会进入 V？',
    formula:
      'X_{\\mathrm{pos}}=E[\\text{token}]+PE_{\\mathrm{pos}},\\qquad (e_m+p_m)\\cdot(e_n+p_n)=e_m\\cdot e_n+e_m\\cdot p_n+p_m\\cdot e_n+p_m\\cdot p_n',
    chapter: [
      '第 5.4 节、第 14 章 · 为什么相加',
      '第14章-词嵌入与位置信息的深层逻辑-为什么相加而不是拼接/',
    ],
  },
  {
    name: 'RoPE 旋转',
    title: 'RoPE：不加到 X 上，而是把 Q、K 按位置旋转',
    insertion: ['rope'],
    takeaway: 'RoPE 在投影之后旋转 Q 和 K；长度不变，V 不旋转。',
    prompt: 'RoPE 作用在哪两个量上？旋转会改变向量长度吗？V 为什么不用转？',
    formula:
      'q^{\\prime}_m=R_m\\,q,\\quad k^{\\prime}_n=R_n\\,k,\\quad R(\\alpha)=\\begin{bmatrix}\\cos\\alpha&-\\sin\\alpha\\\\\\sin\\alpha&\\cos\\alpha\\end{bmatrix},\\quad \\alpha=m\\theta_i,\\ \\theta_i=10000^{-2i/d}',
    chapter: ['第 25.3 节 · RoPE 旋转位置编码', '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/'],
  },
  {
    name: '相对距离',
    title: '同时平移两个位置，RoPE 分数不变',
    insertion: ['rope'],
    takeaway: '内容固定时，$(R_m q)\\cdot(R_n k)$ 只取决于 $n-m$；但它会随距离起伏，并不单调下降。',
    prompt: '为什么 $R_m^{\\top}R_n = R_{n-m}$？“分数只看相对距离”需要哪些条件？',
    formula: '(R_m q)^{\\top}(R_n k)=q^{\\top}R_m^{\\top}R_n\\,k=q^{\\top}R_{n-m}\\,k',
    chapter: [
      '第 25.3.7 节 · 为什么能编码相对位置',
      '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/',
    ],
  },
  {
    name: 'ALiBi',
    title: 'ALiBi：不改 Q、K、V，只在分数上减去距离惩罚',
    insertion: ['alibi'],
    takeaway:
      'ALiBi 把 $-m\\cdot\\text{距离}$ 加到 Softmax 之前的分数上；斜率越大，注意力越集中在附近。',
    prompt: 'ALiBi 改的是哪个矩阵？为什么每个头的斜率不同？能算更长序列意味着什么、不意味着什么？',
    formula:
      'P=\\operatorname{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt d}+m\\cdot B+M\\right),\\qquad B_{ij}=-(i-j)\\quad(j\\le i)',
    chapter: ['第 25.4 节 · ALiBi 线性偏置', '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/'],
  },
  {
    name: '三处对照',
    title: '三种方法，分别在计算的哪一步加入顺序',
    insertion: ['add', 'rope', 'alibi'],
    takeaway: '加法改 X（影响 Q、K、V），RoPE 改 Q、K，ALiBi 改分数。',
    prompt: '不看图，说出三种方法各自加在哪一步、改变了哪些量、V 是否带有位置。',
    formula:
      '\\underbrace{X=E+PE}_{\\text{加法}}\\ \\to\\ \\underbrace{R_mq,\\ R_nk}_{\\text{RoPE}}\\ \\to\\ \\underbrace{\\tfrac{QK^\\top}{\\sqrt d}+mB}_{\\text{ALiBi}}\\ \\to\\ \\operatorname{softmax}\\ \\to\\ PV',
    chapter: ['第 25.6 节 · 各方案对比', '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/'],
  },
];

export const positionSources = [
  {
    title: '原书第 5 章 · Positional Encoding：给文字加位置',
    url: BOOK_URL + '第05章-Positional-Encoding-给文字加位置/',
  },
  {
    title: '原书第 14 章 · 为什么相加而不是拼接',
    url: BOOK_URL + '第14章-词嵌入与位置信息的深层逻辑-为什么相加而不是拼接/',
  },
  {
    title: '原书第 25 章 · 位置编码演进：Sinusoidal 到 RoPE 到 ALiBi',
    url: BOOK_URL + '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/',
  },
  {
    title: '原书源文件（GitHub · waylandzhang/llm-transformer-book，MIT）',
    url: 'https://github.com/waylandzhang/llm-transformer-book',
  },
  { title: 'Attention Is All You Need · §3.5 位置编码', url: 'https://arxiv.org/abs/1706.03762' },
  { title: 'RoFormer（RoPE）· Su et al., 2021', url: 'https://arxiv.org/abs/2104.09864' },
  {
    title: 'ALiBi · Train Short, Test Long · Press et al., 2021',
    url: 'https://arxiv.org/abs/2108.12409',
  },
];

/** Places where this page deliberately differs from the book's wording, with the reason. */
export const bookNotes = [
  '第 5 章的位置表是 $d_{\\mathrm{model}} = 64$ 时的前 4 维，并且截断（不是四舍五入）到两位小数：例如 $\\sin 2 = 0.909\\ldots$ 印成 0.90。本页保留 3 位小数并四舍五入，所以个别格子会差 0.01。',
  '第 5.3.2 节说 Col 6 频率最低、Col 9 最高。按公式，同一对 $\\sin$/$\\cos$ 频率相同，维度序号越大频率越低，书中的波形图本身也是 Col 6、7 比 Col 8、9 转得快。',
  '第 25.3.6 节的 $\\theta = 0.1$、0.2、0.4 是示意数字。按 $\\theta_i = 10000^{-2i/d}$，$\\theta$ 随 $i$ 变小；本页取 $d = 8$，$\\theta$ 恰为 1、0.1、0.01、0.001。',
  '第 25.3.8 节说相邻 token 能获得更高注意力。RoFormer 论文证明的是一个上界随距离衰减；单个分数可能随距离上下起伏，见“相对距离”一步的曲线。',
];
