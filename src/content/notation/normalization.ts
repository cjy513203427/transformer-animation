import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  features: ['x_i,D,i,j', T('x 是一个 token 的特征向量，D 是特征数；i、j 都只遍历同一行的特征。')],
  statistics: ['\\mu,\\sigma^2', T('μ 是该行均值，σ² 是该行方差；方差在本例中除以 D。')],
  normalized: [
    'z_i,r_i',
    T('zᵢ 是 LayerNorm 的归一化特征；rᵢ 是 RMSNorm 的归一化特征，后者不减均值。'),
  ],
  epsilon: ['\\epsilon', T('防止分母为 0 的小正数；它也会让归一化后的方差略有变化。')],
  affine: [
    '\\gamma_i,\\beta_i,y_i^{LN},y_i^{RMS}',
    T('γᵢ 是第 i 维的缩放参数，βᵢ 是平移参数；y 是最终输出，LN/RMS 标明采用哪种归一化。'),
  ],
  softmaxScores: [
    's_i,p_i,m,\\tau,M_i',
    T('s 是温度 τ 缩放并加 Mask 后的分数，p 是概率，m 是最大有效分数；Mᵢ 为 0 或 −∞。'),
  ],
  shift: ['x,c,LN', T('x 是整行输入，c 是加到每个特征的同一个常数；LN 表示 LayerNorm。')],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const normalizationSymbols = createNotation(terms, [
  ['features', 'statistics', 'sum'],
  ['features', 'statistics', 'normalized', 'epsilon'],
  ['features', 'normalized', 'epsilon', 'sum'],
  ['normalized', 'affine'],
  ['features', 'softmaxScores', 'exp', 'sum'],
  ['shift', 'softmax'],
]);
