import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  router: ['Z,W_R', T('Z 是 Router 的打分矩阵；W_R 是路由参数，形状为 [输入维数, 专家数]。')],
  routerAxes: [
    'T,D,E,t,e,j',
    T('T 是 token 数，D 是输入维数，E 是专家数；t 索引 token，e、j 索引专家。'),
  ],
  routerProb: [
    'z_{t,e},p_{t,e},m_t',
    T('依次是 token t 对专家 e 的分数、Softmax 概率，以及该 token 的最高专家分数。'),
  ],
  selection: [
    'A_t,K,\\operatorname{TopK}',
    T('Aₜ 是 token t 选中的专家集合；K 是每个 token 选择的专家个数。'),
  ],
  gate: [
    'p_{t,e},g_{t,e}',
    T('p 是全体专家的路由概率；g 是只在选中专家内重新归一化后的门值，未选中的为 0。'),
  ],
  expert: [
    'x_t,f_e,W_{1,e},W_{2,e}',
    T('xₜ 是一个 token 的输入行；fₑ 是专家 e 的两层 FFN，W₁、W₂ 是它自己的参数。'),
  ],
  combine: [
    'y_t,Y,A_t,g_{t,e}',
    T('yₜ 是 token t 的合并输出，所有行组成 Y；在选中集合 Aₜ 内，按门值 g 加权专家输出。'),
  ],
  counts: [
    'D,E,F,K,T',
    T('D 是输入/输出维数，E 是专家总数，F 是专家中间维数，K 是每 token 选中数，T 是 token 数。'),
  ],
  parameters: [
    'N_{\\mathrm{all}},N_{\\mathrm{active/token}},n_e',
    T('依次是总参数量、每 token 激活参数量、专家 e 收到的任务数；DE 统计 Router 参数。'),
  ],
  communication: [
    'B_{\\mathrm{dispatch}},B_{\\mathrm{return}},B_{\\mathrm{total}}',
    T('依次是发往远端专家、返回结果，以及两者合计的字节数。'),
  ],
  payload: [
    'R,D,b',
    T('R 是跨设备的 token–专家分支数，D 是每个向量的维数，b 是每元素字节数；本例 b = 4。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const moeSymbols = createNotation(terms, [
  ['input', 'router', 'routerAxes', 'routerProb', 'realShape', 'exp', 'sum'],
  ['selection', 'gate'],
  ['expert', 'relu'],
  ['combine', 'expert', 'routerAxes', 'sum', 'realShape'],
  ['counts', 'parameters', 'sum'],
  ['communication', 'payload'],
]);
