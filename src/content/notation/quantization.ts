import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  bits: [
    'b,q_{\\min},q_{\\max}',
    T('b 是量化位数；对称范围取 [−(2^(b−1)−1), 2^(b−1)−1]，非对称范围取 [0, 2^b−1]。'),
  ],
  scale: [
    'g,x_g,s_g,z_g,cap',
    T('g 是分组编号，x_g 是该组原值，s_g 是步长，z_g 是整数零点（对称时为 0）；cap 限制校准幅值。'),
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
  awqGptq: [
    's_i,H^{-1},\\hat w_j',
    T('sᵢ 是 AWQ 按激活幅值计算的通道缩放；H⁻¹ 是 GPTQ 校准激活的二阶逆矩阵，用于把第 j 列误差补偿给后续列 k > j。'),
  ],
  total: [
    'B,b,b_z',
    T('B 是总字节数，b 是位数，b_z 是每组零点字节数（对称时为 0，非对称本例计 1 字节）。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const quantizationSymbols = createNotation(terms, [
  ['bits', 'scale', 'round'],
  ['values', 'scale', 'bits', 'round'],
  ['clipping'],
  ['metadata', 'size'],
  ['awqGptq'],
  ['total', 'size'],
]);
