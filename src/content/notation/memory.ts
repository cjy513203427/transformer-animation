import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  window: [
    't,j,w',
    T('t 是当前 Query 位置，j 是可见 Key 位置，w 是窗口宽度；保留当前及最多 w−1 个历史位置。'),
  ],
  kernel: [
    'w_{tj},q_t,k_j,\\phi',
    T('wₜⱼ 是未归一化的核权重，qₜ/kⱼ 是 Query/Key 行向量；φ 是逐元素特征映射 ELU(x)+1。'),
  ],
  state: [
    'S_t,z_t',
    T(
      'Sₜ 累积 Key 特征与 Value 的外积，zₜ 累积 Key 特征；状态覆盖截至 t 的历史。这里 S 不是分数矩阵。',
    ),
  ],
  vectors: [
    'q_t,k_t,v_t,o_t',
    T('位置 t 的 Query、Key、Value 和输出行向量；φ(qₜ) 查询累积状态，再用分母归一化。'),
  ],
  mixing: [
    'A,A_{mem},A_{local},\\beta,\\sigma',
    T(
      'A 是混合输出，A_mem 是历史记忆输出，A_local 是局部 Attention 输出；σ(β) 是 0–1 之间的门值。',
    ),
  ],
  update: [
    'M_{old},M_{new},\\widehat V_{old}',
    T(
      'M 是压缩记忆矩阵；old/new 表示更新前后，V̂_old 是旧记忆预测的 Value，残差 V−V̂_old 用于修正。',
    ),
  ],
  featureMap: ['\\phi(K),V', T('φ(K) 是对 Key 矩阵逐元素做特征映射，V 是对应的 Value 矩阵。')],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const memorySymbols = createNotation(terms, [
  ['query', 'key', 'value', 'probability', 'output', 'keyDim', 'mask', 'transpose', 'softmax'],
  ['window'],
  ['kernel'],
  ['state', 'vectors', 'kernel', 'transpose'],
  ['mixing'],
  ['update', 'featureMap', 'transpose'],
]);
