import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  first: ['H,W_1', T('W₁ 是从 2 维扩到 3 维的参数；H 是投影后的中间激活，形状 [4, 3]。')],
  activation: ['H,U', T('H 是非线性之前的激活，U 是逐元素 ReLU 后的结果；形状相同。')],
  second: ['U,W_2,F', T('U 是 3 维中间激活，W₂ 投回 2 维；F 是 FFN 分支输出，形状 [4, 2]。')],
  residual: ['Y,X,F(X)', T('Y 是残差相加后的输出；X 走直连路径，F(X) 是 FFN 分支的计算结果。')],
  jacobian: ['J_Y,J_F,I', T('J 是输出对输入的导数矩阵（Jacobian）；I 是直连路径贡献的单位矩阵。')],
  dropout: [
    'D(F)_i,m_i,p',
    T('D(F) 是 Dropout 后的分支；p 是丢弃率，mᵢ 以 1−p 的概率取 1，否则取 0；i 遍历元素。'),
  ],
  linear: [
    'W_1,W_2,W_1W_2',
    T('两层之间没有非线性时，可以先把两个参数矩阵相乘，合成一个线性投影。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const ffnSymbols = createNotation(terms, [
  ['input', 'first'],
  ['activation', 'relu'],
  ['second'],
  ['residual', 'jacobian'],
  ['dropout'],
  ['input', 'linear'],
]);
