import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  scan: [
    'x_t,h_t,h_{t-1},y_t,t',
    T('t 是时间/位置；xₜ 是当前输入，hₜ₋₁ 是旧状态，hₜ 是新状态，yₜ 是当前输出。本例均为标量。'),
  ],
  discrete: [
    '\\bar A,\\bar B,C',
    T('Ā 控制旧状态保留多少，B̄ 控制新输入写入多少，C 将状态读成输出；横线表示离散化后。'),
  ],
  selective: [
    '\\Delta,\\Delta_t,B_t,C_t',
    T('Δ 是基础步长；Δₜ、Bₜ、Cₜ 是本例按当前 xₜ 生成的步长、写入和读出系数。'),
  ],
  softplus: [
    'softplus(x)=\\log(1+e^x)',
    T('平滑的正值函数，保证本例输入相关的步长为正；0.5、0.25、0.1 是教学设定。'),
  ],
  continuous: [
    'A,\\bar A_t,\\bar B_t',
    T('A 是连续时间的状态系数；Āₜ、B̄ₜ 按步长 Δₜ 离散化。本例 A 为负以产生衰减。'),
  ],
  decay: [
    'A,\\Delta,e^{\\Delta A}',
    T('A < 0 时，增大正步长 Δ 会降低保留比例 exp(ΔA)，旧状态衰减更快。'),
  ],
  summary: [
    'a,c,\\delta,\\lambda,h_2',
    T(
      'a、c 是连续两步的输入，δ 是对第一步的改动，λ = exp(ΔA) 是固定保留系数；第二步补偿 −λδ 后，两段历史得到同一个最终状态 h₂。',
    ),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const ssmSymbols = createNotation(terms, [
  ['scan', 'discrete'],
  ['scan', 'selective', 'softplus'],
  ['selective', 'continuous'],
  ['scan', 'continuous'],
  ['decay'],
  ['summary'],
]);
