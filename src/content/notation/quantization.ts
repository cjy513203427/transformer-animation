import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  bits: ['b,q_{max}', T('b 是量化位数；q_max 是对称整数范围的正端点，本例使用 [−q_max, q_max]。')],
  scale: [
    'g,x_g,s_g,cap',
    T('g 是分组编号，x_g 是该组原值，s_g 是该组量化步长；cap 限制用于计算步长的最大幅值。'),
  ],
  values: [
    'x_i,q_i,\\hat x_i',
    T('i 索引原始元素；xᵢ 是原值，qᵢ 是整数编码，带帽的 x̂ᵢ 是反量化近似值。'),
  ],
  clipping: [
    'a_g,x_{clip}',
    T('a_g 是第 g 组的裁剪幅值上限；x_clip 是限制到 [−a_g, a_g] 后的值。'),
  ],
  size: ['N,G,\\lceil N/G\\rceil', T('N 是元素总数，G 是每组元素数；⌈N/G⌉ 向上取整得到分组数。')],
  metadata: ['B_{scale}', T('所有分组步长的存储字节数；本例每组保存一个 4 字节的步长。')],
  error: ['MSE', T('均方误差（Mean Squared Error）：逐元素反量化误差的平方，再取平均。')],
  total: ['B,b', T('B 是整数载荷与步长元数据的总字节数；b 是每个整数编码的位数。')],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const quantizationSymbols = createNotation(terms, [
  ['bits', 'scale'],
  ['values', 'scale', 'bits', 'round'],
  ['clipping'],
  ['metadata', 'size'],
  ['error', 'values', 'size', 'sum'],
  ['total', 'size'],
]);
