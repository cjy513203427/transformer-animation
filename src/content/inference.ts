import type { FormulaSymbol } from './notation/shared';
import { inferenceSymbols } from './notation/inference';
import { chapterUrl } from './book';
import { T } from '../i18n';

const CH16 = '第16章-训练与推理的异同-为什么推理要一个字一个字生成/';
const CH22 = '第22章-KV-Cache-推理加速/';

export const inferenceSteps: {
  name: string;
  title: string;
  explanation: string;
  takeaway: string;
  prompt: string;
  formula: string;
  symbols: FormulaSymbol[];
  chapter: [string, string];
}[] = [
  {
    name: T('训练并行'),
    title: T('训练：一次前向，同时算出每个位置的预测'),
    explanation: T(
      '训练样本 a b c d e 是完整已知的。把它错开一位：输入 x = a b c d，目标 y = b c d e。因果 Mask 让第 $t$ 行只读得到 $x_0,\\dots,x_t$，所以一次矩阵运算就能同时得到 4 个位置的预测，每个位置各自和目标比出一个 loss。目标已知，不等于前向偷看了答案：第 $t$ 行的输出根本没有用到 $x_{t+1}$ 及之后的输入。',
    ),
    takeaway: T('训练能并行，是因为整条序列事先已知；不会偷看，是因为因果 Mask。两件事同时成立。'),
    prompt: T('为什么训练能一次算出所有位置？把第 2 个输入换掉，第 0、1 行的输出会不会变？'),
    symbols: inferenceSymbols[0],
    formula:
      '\\mathcal{L}=\\frac{1}{T}\\sum_{t=0}^{T-1}-\\log p\\left(y_t\\mid x_0,\\dots,x_t\\right),\\qquad y_t=x_{t+1}',
    chapter: [T('第 16.2 节 · Teacher Forcing 与 Causal Mask'), CH16],
  },
  {
    name: T('无缓存基线'),
    title: T('生成：下一个输入，要等这一步的输出'),
    explanation: T(
      '推理时只有提示 a b c。模型算出最后一个位置的分布，选出下一个 token，接到序列末尾，再把整条序列重新算一遍。每一步只用最后一行的输出，可前面那些行在上一步已经算过，而且结果完全相同。重复计算就藏在这里。',
    ),
    takeaway: T(
      '生成必须一步一步来，因为下一个输入要等这一步的输出。不用缓存时，每一步都在重算前面早已算完的行。',
    ),
    prompt: T('不用缓存时，第 3 步重算的那些行里，哪些在上一步已经算过？它们的值会不会变？'),
    symbols: inferenceSymbols[1],
    formula: T(
      'x_n=\\arg\\max_{v}\\ p\\left(v\\mid x_0,\\dots,x_{n-1}\\right),\\qquad \\text{只用最后一行 } h_{n-1}',
    ),
    chapter: [T('第 16.4 节 · 一个不带 KV Cache 的基线循环'), CH16],
  },
  {
    name: 'Prefill',
    title: T('Prefill：一次处理整段提示，写好每层的 K、V'),
    explanation: T(
      'Prefill 和训练时的前向一样：提示的 3 个位置在一次因果前向里并行计算。不同的是，它把每一层、每个位置的 K 和 V 留下来，作为后面每一步都要读的历史。Prefill 也算出了每个位置的输出，但生成只用最后一个位置的分布，用它选出第一个新 token。',
    ),
    takeaway: T('Prefill 并行处理已有上下文，写好每一层的 K、V 缓存，并给出第一个新 token。'),
    prompt: T('Prefill 之后，缓存里一共有几组 K、V？为什么每一层都要各存一份？'),
    symbols: inferenceSymbols[2],
    formula:
      'K^{(\\ell)}=H^{(\\ell-1)}W_K^{(\\ell)},\\quad V^{(\\ell)}=H^{(\\ell-1)}W_V^{(\\ell)},\\qquad \\ell=1,\\dots,L',
    chapter: [T('第 16.3.1 节 · Prefill 和 Decode'), CH16],
  },
  {
    name: T('Decode 一步'),
    title: T('Decode：只为新 token 算 Q、K、V，再读全部历史'),
    explanation: T(
      '每一步只喂入一个新 token。每一层只为它算一组 $q$、$k$、$v$，把 $k$、$v$ 追加进缓存，再用新的 $q$ 去读缓存里全部可见的 K、V。旧位置的 Q 不用保存：它们对应的输出早在之前的步骤里算完了，后面的位置只会用自己的新 Q 去查询历史。',
    ),
    takeaway: T('缓存存的是每层每个位置的 K、V，不存 Q。新 Query 仍要读完全部可见历史。'),
    prompt: T('一次 Decode 里，每层新算了几组向量？读了几组？为什么不缓存旧的 Q？'),
    symbols: inferenceSymbols[3],
    formula:
      'q_n=h_nW_Q,\\ \\ k_n=h_nW_K,\\ \\ v_n=h_nW_V,\\qquad K\\leftarrow\\begin{bmatrix}K\\\\ k_n\\end{bmatrix},\\ \\ V\\leftarrow\\begin{bmatrix}V\\\\ v_n\\end{bmatrix},\\qquad o_n=\\operatorname{softmax}\\!\\left(\\frac{q_nK^{\\top}}{\\sqrt d}\\right)V',
    chapter: [T('第 22.2 节 · Cache 里存什么，每一步又算什么'), CH22],
  },
  {
    name: T('缓存对照重算'),
    title: T('对照：结果相同，省掉的是重复工作'),
    explanation: T(
      '同一个提示、同一组权重、同样的位置编号和 Mask，分别用“缓存 + 只算新 token”和“每步整段重算”生成。两条路线每一步的 logits 一致，选出的 token 也一样。区别在工作量：序列长为 $n$ 时，重算每层要做 $n$ 次 K/V 投影和 $n(n+1)/2$ 次打分；用缓存只做 1 次投影和 $n$ 次打分。省下的是重复的投影和打分，读取全部历史这件事省不掉。',
    ),
    takeaway: T(
      '缓存不改变结果，只省掉重复计算。新 Query 仍要访问全部可见历史，所以不能承诺固定的加速倍数。',
    ),
    prompt: T('为什么两条路线的 logits 一样？用缓存时，每一步的打分次数随什么增长？'),
    symbols: inferenceSymbols[4],
    formula: T(
      '\\text{重算：}\\ L\\cdot\\frac{n(n+1)}{2}\\ \\text{次打分}\\qquad\\text{缓存：}\\ L\\cdot n\\ \\text{次打分}',
    ),
    chapter: [T('第 22.3 节 · 计算量：先说清楚在数哪一部分'), CH22],
  },
  {
    name: T('缓存失效'),
    title: T('改了前缀，哪些缓存不能再用？'),
    explanation: T(
      '第 1 层的 K、V 只取决于本位置的 token 和位置编号。从第 2 层开始，每个位置的输入已经混入了它之前所有 token 的信息。所以改动第 $j$ 个 token，第 1 层只有位置 $j$ 失效，第 2 层从位置 $j$ 起全部失效。滑动窗口丢掉最早的 token 时，剩下的位置编号整体减 1，在加法位置编码下，每个旧条目都不能直接复用。',
    ),
    takeaway: T(
      '缓存只对完全相同的前缀、位置编号、权重和运行模式有效。改动一个 token，它之后的高层缓存都要重算。',
    ),
    prompt: T(
      '改动第 1 个 token，第 1 层和第 2 层分别有哪些位置的缓存失效？多轮对话为什么不会自动复用缓存？',
    ),
    symbols: inferenceSymbols[5],
    formula: T(
      'h^{(1)}_t=f\\left(x_0,\\dots,x_t\\right)\\ \\Rightarrow\\ K^{(2)}_t,\\ V^{(2)}_t\\ \\text{依赖 } x_0,\\dots,x_t',
    ),
    chapter: [T('第 22.6 节 · 多轮对话不是自动白送'), CH22],
  },
];

export const inferenceSources = [
  {
    title: T('原书第 2 章 · 大模型的本质（预测下一个 token）'),
    url: chapterUrl('第02章-大模型的本质-就是两个文件/'),
  },
  { title: T('原书第 16 章 · 训练与推理的异同'), url: chapterUrl(CH16) },
  { title: T('原书第 22 章 · KV Cache'), url: chapterUrl(CH22) },
  {
    title: T('Attention Is All You Need · §3.2.3 解码器中的 Mask'),
    url: 'https://arxiv.org/abs/1706.03762',
  },
  {
    title: T(
      'Efficiently Scaling Transformer Inference · Pope et al., 2022（Prefill 与 Decode 两个阶段）',
    ),
    url: 'https://arxiv.org/abs/2211.05102',
  },
];

/** Common misconceptions, each answered by something the page computes. */
export const inferenceMyths = [
  T('缓存存的不是“上一轮的输出”，而是每一层、每个位置的 K 和 V。'),
  T('不缓存 Q：旧位置的输出已经算完，新位置只用自己的新 Q 去查询历史。'),
  T('缓存不改变结果，只省掉重复的 K/V 投影和打分；新 Query 仍要读全部可见的 K、V。'),
  T('训练时标签已知，但前向仍受因果 Mask 约束：第 $t$ 行的输出不使用 $x_{t+1}$ 之后的输入。'),
  T('不承诺固定的加速倍数：实际耗时还取决于内存带宽、批大小和实现。'),
  T('多轮对话界面本身不证明缓存被复用：需要完全相同的 token 前缀、位置编号、权重和运行模式。'),
];
