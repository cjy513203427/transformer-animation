import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  logits: ['z_i,p_i,i,j', T('zᵢ 是词表候选 i 的 Logit，pᵢ 是其概率；i、j 遍历词表候选。')],
  temperature: ['\\tau', T('温度（Temperature），必须大于 0；越小通常越集中，越大通常越平坦。')],
  topk: [
    "A,K,p'_i,\\mathbf1[i\\in A]",
    T('A 是保留的 K 个候选集合；指示函数对保留项取 1，其余取 0；p′ 是过滤后重新归一化的概率。'),
  ],
  topp: [
    'P,p_{(i)},k,m',
    T(
      'P 是 Top-P 阈值；p_(i) 按概率从大到小排列；m 是累计概率首次达到 P 时的最少候选数，k 为计数变量。',
    ),
  ],
  sampling: [
    "u\\sim U[0,1),p'_j,i",
    T('u 是 [0,1) 内均匀随机数；沿过滤后的概率 p′ 累加，首次超过 u 的候选编号就是 i。'),
  ],
  sequence: [
    'y_1,y_2,p(y_2\\mid y_1)',
    T('y₁、y₂ 是先后两个输出 token；条件概率表示已选 y₁ 后生成 y₂ 的概率，log 是自然对数。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const decodingSymbols = createNotation(terms, [
  ['logits', 'exp', 'sum'],
  ['logits', 'temperature', 'softmax'],
  ['topk', 'logits'],
  ['topp', 'sum'],
  ['sampling', 'sum'],
  ['sequence'],
]);
