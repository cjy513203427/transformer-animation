import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  embedding: [
    'E[\\mathrm{token}_t]',
    T('E 是词嵌入表；按位置 t 上的 token 编号取出一行内容向量。'),
  ],
  blind: [
    'x_i,x_j,O_i',
    T('x 是未加位置的内容行向量；i 是当前 Query 位置，j 遍历各 Key，Oᵢ 是汇总结果。'),
  ],
  dimension: ['d', T('当前向量的维数；在 Attention 分数中用 √d 缩放。')],
  pe: ['PE', T('位置编码（Positional Encoding）；每个位置一行，与隐藏状态有相同的特征维数。')],
  sinusAxes: [
    '\\mathrm{pos},i,2i,2i+1',
    T('pos 是从 0 开始的位置；i 是维度对编号，2i 与 2i+1 是该对的正弦、余弦列。'),
  ],
  contentPosition: [
    'e_m,e_n,p_m,p_n',
    T('e 是内容嵌入，p 是位置向量；下标 m、n 分别指两个 token 的位置。'),
  ],
  positions: ['m,n', T('两个 token 的位置编号；n − m 是它们的相对位移。')],
  ropeVectors: [
    'q,k,q_m^{\\prime},k_n^{\\prime}',
    T('q、k 是固定内容向量；带撇号的是按位置旋转后的 Query、Key。本式用列向量写旋转。'),
  ],
  rotation: [
    'R_m,R_n,R(\\alpha)',
    T('R 是二维旋转矩阵，逐维度对作用；下标是位置，α 是旋转角，单位为弧度。'),
  ],
  frequency: [
    '\\theta_i,i,d',
    T('θᵢ 是第 i 个维度对每前进一个位置转过的角度；i 从 0 开始，d 是 Q/K 的总维数。'),
  ],
  relative: ['R_{n-m}', T('按相对位移 n − m 旋转；共同平移不变性要求内容 q、k 与频率规则固定。')],
  slope: ['m', T('此处是 ALiBi 的头斜率（slope），控制距离惩罚强度；不是 RoPE 的位置下标。')],
  bias: ['B_{ij}', T('距离偏置；i 为 Query 位置、j 为可见 Key 位置，取负距离 −(i − j)。')],
  overview: [
    'R_mq,R_nk,mB',
    T(
      'R 的下标 m、n 是 RoPE 的位置；后面 mB 中的 m 是 ALiBi 的斜率，B 是距离偏置。这里并列展示三种对照方案。',
    ),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const positionSymbols = createNotation(terms, [
  ['embedding', 'blind', 'dimension', 'softmax', 'sum'],
  ['pe', 'sinusAxes', 'modelDim'],
  ['input', 'embedding', 'pe', 'contentPosition'],
  ['ropeVectors', 'rotation', 'positions', 'frequency'],
  ['ropeVectors', 'rotation', 'positions', 'relative', 'transpose'],
  ['query', 'key', 'probability', 'slope', 'bias', 'mask', 'dimension', 'softmax', 'transpose'],
  ['input', 'embedding', 'pe', 'overview', 'query', 'key', 'probability', 'value', 'dimension'],
]);
