import { T } from '../../i18n';

/** A symbol as typeset in the formula, followed by its local meaning. */
export type FormulaSymbol = readonly [tex: string, meaning: string];

export function createNotation<T extends Record<string, FormulaSymbol>>(
  terms: T,
  steps: readonly (readonly (keyof T)[])[],
): FormulaSymbol[][] {
  return steps.map((keys) => keys.map((key) => terms[key]));
}

export const commonSymbols = {
  input: ['X', T('输入隐藏状态矩阵；每行一个 token，每列一个特征。')],
  query: ['Q', T('查询（Query）矩阵：各位置用什么去匹配 Key。')],
  key: ['K', T('键（Key）矩阵：各位置提供给 Query 匹配的特征。')],
  value: ['V', T('值（Value）矩阵：按注意力权重汇总的内容。')],
  projection: ['W_Q, W_K, W_V', T('把 X 投影为 Q、K、V 的参数矩阵；本次前向中保持固定。')],
  score: ['S', T('注意力分数矩阵；行是 Query 位置，列是 Key 位置。')],
  probability: ['P', T('注意力权重矩阵；每行给出取用各个 Value 的比例。')],
  output: ['O', T('Attention 输出矩阵；每行是一个 Query 汇总得到的新向量。')],
  mask: ['M', T('加到分数上的掩码（Mask）；可见处为 0，屏蔽处为 −∞。')],
  keyDim: ['d_k', T('每个 Query / Key 向量的维数；除以其平方根来缩放分数。')],
  valueDim: ['d_v', T('每个 Value 和输出向量的维数。')],
  tokens: ['T', T('当前序列的 token 数；若矩阵每行对应一个 token，T 就是行数。')],
  modelDim: ['d_{\\mathrm{model}}', T('每个 token 的隐藏状态维数，也就是输入的特征列数。')],
  transpose: [
    '{}^{\\top},{}^{T}',
    T('右上角 ⊤ 或 T 表示转置：交换矩阵的行列，向量也随之切换行、列写法。'),
  ],
  sum: ['\\sum', T('求和：把下标遍历到的各项相加。')],
  exp: ['\\exp(x)=e^x', T('指数函数；Softmax 把分数转成非负的相对权重。')],
  softmax: ['\\operatorname{softmax}', T('沿指定轴把分数归一化；有效候选的权重和为 1。')],
  realShape: [
    '\\mathbb R^{a\\times b}',
    T('实数矩阵的形状：a 行、b 列；方括号 [a, b] 也表示同一形状。'),
  ],
  relu: ['\\operatorname{ReLU}(x)', T('逐元素取 max(0, x)：负数变 0，非负数保留。')],
  round: [
    '\\operatorname{round},\\operatorname{clip}',
    T('round 四舍五入到整数；clip 把数值限制到给定上下界。'),
  ],
} satisfies Record<string, FormulaSymbol>;
