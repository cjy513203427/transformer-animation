import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  reward: [
    'y_w,y_l,r_w,r_l',
    T('w 表示偏好的回答（winner），l 表示另一个回答（loser）；r 是各自的奖励分数。'),
  ],
  sigmoid: ['\\sigma(x)=1/(1+e^{-x})', T('Sigmoid 函数，把奖励差等实数映射到 0 与 1 之间。')],
  prefProb: ['P(y_w\\succ y_l)', T('模型认为回答 w 优于回答 l 的概率；≻ 表示“更受偏好”。')],
  policies: [
    '\\pi_\\theta,\\pi_{ref},\\theta',
    T('π_θ 是当前策略对回答的概率，π_ref 是固定参考策略；θ 表示当前策略参数。'),
  ],
  dpo: ['L,\\beta', T('L 是要最小化的 DPO 损失；β 控制相对参考策略的对数概率差的缩放。')],
  ppo: [
    'J,\\rho,A,\\epsilon,clip',
    T(
      'J 是要增大的裁剪目标；ρ 是新/旧策略概率比，A 是优势；clip 将比例限制在 [1−ε,1+ε]，ε 是裁剪幅度。',
    ),
  ],
  group: [
    'G,r_i,\\bar r,A_i,i,j',
    T(
      'G 是同一 prompt 的回答数，rᵢ 是各回答奖励，r̄ 是组均值；Aᵢ 是除以组标准差后的优势，i/j 遍历回答。',
    ),
  ],
  grpo: [
    'J,\\rho_i,A_i,\\beta,k_i',
    T(
      'J 是组平均目标；ρᵢ 是新/旧概率比，Aᵢ 是组内优势，β 缩放相对参考策略的 KL 惩罚 kᵢ。clip 使用当前 ε 范围。',
    ),
  ],
  kl: [
    'k_i,\\pi_{ref,i},\\pi_{\\theta,i}',
    T('kᵢ 是当前回答的 KL 样本惩罚；分子是参考概率，分母是当前策略概率，两者都需为正。'),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const preferenceSymbols = createNotation(terms, [
  ['reward', 'sigmoid', 'prefProb'],
  ['reward', 'policies', 'dpo', 'sigmoid'],
  ['ppo'],
  ['group', 'sum'],
  ['group', 'grpo', 'sum'],
  ['kl'],
]);
