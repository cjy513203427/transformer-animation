import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  heads: ['\\mathrm{num\\_heads}', T('注意力头数；每个头保留全部 token，只分取投影后的特征列。')],
  headDim: ['d_{\\mathrm{key}}', T('本例每个头的 Q、K、V 维数，等于隐藏维数除以头数。')],
  headQKV: ['Q_h,K_h,V_h', T('下标 h 指第 h 个头；三者形状均为 [token 数, 每头维数]。')],
  headOutput: ['A_h', T('第 h 个头的 Attention 输出；每个头独立归一化并汇总自己的 Value。')],
  concat: [
    'A,\\operatorname{Concat}',
    T('A 是各头输出沿特征列拼接后的矩阵；token 的行顺序保持不变。'),
  ],
  outProjection: ['W_O,W_O^{(h)}', T('W_O 是输出投影参数；右上角 (h) 表示与第 h 个头相乘的行块。')],
  final: ['\\mathrm{Output}', T('拼接结果 A 再乘 W_O 后的最终输出；形状为 [T, d_model]。')],
  reshape: [
    '\\mathrm{transpose}',
    T('这里 transpose 只交换“token”和“头”两个轴；每个 token 的特征与数值保持对应。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const multiheadSymbols = createNotation(terms, [
  ['input', 'query', 'key', 'value', 'projection'],
  ['modelDim', 'heads', 'headDim', 'tokens'],
  ['tokens', 'heads', 'headDim', 'reshape'],
  ['headQKV', 'headOutput', 'headDim', 'mask', 'softmax', 'transpose'],
  ['concat', 'headOutput', 'heads', 'tokens', 'modelDim', 'realShape'],
  ['final', 'concat', 'headOutput', 'outProjection', 'sum'],
]);
