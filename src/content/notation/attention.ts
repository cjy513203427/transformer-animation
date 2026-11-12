import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  axes: ['i,j,k', T('i 是 Query 行，j 是 Key 列；k 在分母中遍历同一行的所有 Key。位置从 0 开始。')],
  maximum: ['m_i', T('第 i 行加上 Mask 后的最大分数；减去它可避免指数溢出。')],
  indexedOutput: [
    'O_i, P_{ij}, V_j',
    T('Oᵢ 是第 i 行输出；Pᵢⱼ 是它分给位置 j 的权重；Vⱼ 是该位置的 Value 行向量。'),
  ],
  projectionCell: [
    'Q_{i,a}',
    T('Query 矩阵第 i 个 token、第 a 个特征的数值；下标中的数字是从 0 开始的行、列编号。'),
  ],
  scoreCell: [
    'S_{ij}',
    T('第 i 个 Query 与第 j 个 Key 的分数；展开式中每对数按同一个特征维相乘。'),
  ],
  outputCell: [
    'O_{i,a}',
    T('第 i 个 Query 的第 a 个输出特征；展开式依次累加“Key j 的权重 × Value j 的该特征”。'),
  ],
  delta: [
    '\\Delta O_{i,0},\\Delta V',
    T('Δ 表示改动前后的差值；这里只改变 V[1,0]，观察第 i 行输出的第 0 维。'),
  ],
  deltaWeight: [
    'P_{i,1}',
    T('第 i 个 Query 分给 token 1 的权重；固定 Q、K 和 Mask 时，它不随 ΔV 改变。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const attentionSymbols = createNotation(terms, [
  ['input', 'query', 'key', 'value', 'projection'],
  ['score', 'query', 'key', 'keyDim', 'transpose'],
  ['mask', 'axes'],
  ['probability', 'score', 'mask', 'maximum', 'axes', 'sum', 'exp'],
  ['output', 'probability', 'value', 'indexedOutput', 'sum'],
]);
export const projectionCellSymbols = createNotation(terms, [['projectionCell']])[0];
export const scoreCellSymbols = createNotation(terms, [['scoreCell']])[0];
export const outputCellSymbols = createNotation(terms, [['outputCell']])[0];
export const valueChangeSymbols = createNotation(terms, [['delta', 'deltaWeight']])[0];
