import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  base: ['W_0,Y_0', T('W₀ 是冻结的基础权重，形状 [D, K]；Y₀ 是只走基础分支的输出。')],
  dimensions: [
    'D,K,r',
    T('D 是输入维数，K 是输出维数，r 是适配器的中间秩（rank）；这里 K 不表示 Key。'),
  ],
  adapter: ['A,B,\\Delta W', T('A、B 是可训练的低秩矩阵；ΔW 是它们相乘并缩放后形成的权重增量。')],
  scaling: ['\\alpha/r', T('α 是适配器缩放超参数；除以秩 r 得到增量分支的倍率。')],
  result: ['Y', T('基础分支输出加适配器分支输出；两条路径的输出形状相同。')],
  quantized: [
    'W_{0,q},\\operatorname{dequant}',
    T('W₀,q 是低比特存储的基础权重；dequant 将其恢复为本次乘法使用的数值。'),
  ],
  count: ['N_{adapter}', T('适配器参数个数；A 有 D×r 个、B 有 r×K 个，只统计这两组。')],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const loraSymbols = createNotation(terms, [
  ['input', 'base', 'dimensions'],
  ['adapter', 'dimensions', 'scaling', 'realShape'],
  ['input', 'base', 'adapter', 'scaling', 'result'],
  ['input', 'base', 'adapter'],
  ['input', 'quantized', 'adapter', 'scaling', 'result'],
  ['count', 'dimensions'],
]);
