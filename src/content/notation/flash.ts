import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  tile: [
    't,b,B,j',
    T('t 是当前 Query 位置，b 是当前 Key/V 块编号，B 是块最多包含的行数；j 遍历本块 Key。'),
  ],
  tileVectors: [
    'q_t,K_b,V_b,s_b,M_{t,b}',
    T('依次是当前 Query 行、Key 块、Value 块、该块分数和对应 Mask；块内 K/V 来自相同 token 行。'),
  ],
  maximum: ["m,m'", T('m 是处理旧块后的最大分数；m′ 是合入新块后的最大分数。撇号表示更新后。')],
  accumulators: [
    "l,l',a,a'",
    T('l 是当前尺度下的指数和（标量）；a 是未归一化的 Value 加权和（d_v 维向量）；撇号为更新后。'),
  ],
  alpha: ["\\alpha=e^{m-m'}", T('重缩放系数：把旧分母 l 和旧分子 a 一起换到新最大值的指数尺度。')],
  exponent: [
    's_{b,j},r_j,v_j',
    T('s 是本块第 j 个分数，r 是减去新最大值后的指数，v 是对应的 Value 行向量。'),
  ],
  partialOutput: [
    "o'=a'/l'",
    T('当前已处理 Key 的暂时输出；只有读完所有可见块，才是这一行的最终结果。'),
  ],
  equality: [
    'O_{t,:},O^{\\mathrm{online}}_{t,d},O^{\\mathrm{full}}_{t,d}',
    T('t 索引 Query 行，d 索引输出特征，冒号表示整行；online 与 full 分别是分块和全量路径的结果。'),
  ],
  finalState: [
    'a_t,l_t',
    T('第 t 行读完所有 Key 后的加权和与指数和；同一行的向量 aₜ 除以标量 lₜ。'),
  ],
  tolerance: [
    '\\max|\\cdot|\\le10^{-12}',
    T('逐行、逐特征检查最大绝对误差；本例 float64 核对容差为 10⁻¹²。'),
  ],
  storage: [
    'N_{S,P},N_{\\mathrm{transfer},S,P}',
    T('前者是完整 S/P 的中间元素数，后者是它们写回、读回的元素次数；0 只针对这两张矩阵。'),
  ],
  work: [
    '\\Theta(T^2(d_k+d_v))',
    T('完整 Attention 的算术增长量级：每对 Query–Key 仍需打分并参与 Value 汇总。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const flashSymbols = createNotation(terms, [
  [
    'query',
    'key',
    'value',
    'score',
    'probability',
    'output',
    'mask',
    'tokens',
    'keyDim',
    'valueDim',
    'transpose',
    'softmax',
    'realShape',
  ],
  ['tile', 'tileVectors', 'keyDim', 'valueDim', 'transpose', 'realShape'],
  ['maximum', 'accumulators', 'alpha', 'exponent', 'partialOutput', 'sum'],
  ['maximum', 'alpha', 'accumulators'],
  ['equality', 'finalState', 'tolerance'],
  ['storage', 'tokens', 'keyDim', 'valueDim', 'work'],
]);
