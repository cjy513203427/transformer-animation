import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  projection: [
    'W_Q,W_K,W_V',
    T('W_Q、W_K、W_V 是当前层从隐藏向量生成 Query、Key、Value 的投影参数，推理时固定。'),
  ],
  loss: ['\\mathcal L', T('训练损失：每个位置的目标 token 负对数概率，再对 T 个位置取平均。')],
  training: [
    'x_t,y_t,t,T',
    T('xₜ 是位置 t 的输入 token，yₜ 是下一位置的目标 token；T 是本次输入长度，t 从 0 开始。'),
  ],
  conditional: [
    'p(y\\mid x),-\\log p',
    T('p 是给定可见前缀后预测目标的概率；−log p 在预测目标概率越低时越大。'),
  ],
  greedy: [
    'x_n,v,n,\\arg\\max_v',
    T('n 是已有 token 数，v 遍历词表；选择概率最大的 v，作为新 token xₙ。'),
  ],
  hiddenLast: ['h_{n-1}', T('最后一个已有位置的隐藏向量，用来预测接下来的 token。')],
  layers: ['\\ell,L', T('ℓ 是层编号，L 是总层数；右上角 (ℓ) 表示这一层，不是乘方。')],
  layerInput: [
    'H^{(\\ell-1)}',
    T('当前层的输入隐藏状态；来自上一层，形状为 [已有 token 数, 隐藏维数]。'),
  ],
  layerKV: [
    'K^{(\\ell)},V^{(\\ell)},W_K^{(\\ell)},W_V^{(\\ell)}',
    T('K、V 是当前层要缓存的激活；W 是该层固定的投影参数。'),
  ],
  newVectors: [
    'h_n,q_n,k_n,v_n,o_n',
    T('下标 n 指新位置：h 是本层输入，q/k/v 是新投影，o 是新 Query 的 Attention 输出。'),
  ],
  append: [
    'K\\leftarrow[K;k_n],V\\leftarrow[V;v_n]',
    T('把新位置的 k、v 追加为缓存的新行；同一层的历史行保持原顺序。'),
  ],
  dimension: ['d', T('本例每层 Query / Key 的维数；√d 是分数缩放因子。')],
  count: ['L,n', T('L 是层数，n 是包含当前位置的序列长度；这里只数 Query–Key 打分配对。')],
  invalidation: [
    'h_t^{(1)},K_t^{(2)},V_t^{(2)}',
    T('位置 t 的第 1 层输出，以及第 2 层据此投影的 K、V；它们依赖从 x₀ 到 xₜ 的前缀。'),
  ],
  function: ['f', T('本例第 1 层的因果前向计算；第 t 行只能使用截至 t 的输入。')],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const inferenceSymbols = createNotation(terms, [
  ['loss', 'training', 'conditional', 'sum'],
  ['greedy', 'hiddenLast', 'conditional'],
  ['layers', 'layerInput', 'layerKV'],
  ['newVectors', 'projection', 'append', 'dimension', 'softmax', 'transpose'],
  ['count'],
  ['invalidation', 'function'],
]);
