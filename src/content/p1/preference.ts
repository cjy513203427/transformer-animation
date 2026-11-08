import { preference } from '../../math/p1/preference';
import { field, lesson, type TopicDefinition } from './types';
export default {
  id: 'preference',
  path: '/preference',
  title: '偏好学习：奖励怎样进入训练目标？',
  chapter: '第 29、31 章',
  intro:
    '给同一 prompt 的四个回答打分，拆开偏好概率、DPO、PPO-Clip 与 GRPO 的组内优势。这里只核对公开目标，不运行训练。',
  input: [0, 1, 2, 3],
  fields: [
    field('policy', '当前回答0的 logit', 0.5, -2, 2, 0.25, [1, 4, 5]),
    field('beta', 'DPO β', 0.1, 0.05, 1, 0.05, [1]),
    field('ratio', '单项 PPO 比率 ρ', 1.5, 0.25, 2, 0.25, [2]),
    field('advantage', '单项 PPO 优势 A', 1, -2, 2, 0.25, [2]),
    field('clip', '裁剪宽度 ε', 0.2, 0.1, 0.4, 0.1, [2, 4]),
    field('kl', 'GRPO 参考约束系数', 0.02, 0, 0.2, 0.01, [4]),
    field('shift', '奖励共同平移 c', 0, -3, 3, 0.25, [3, 5]),
  ],
  steps: [
    lesson(
      '偏好概率',
      String.raw`P(y_w\succ y_l)=\sigma(r_w-r_l)`,
      '同一 prompt 的候选回答组成比较。Bradley–Terry 依据奖励差构造偏好概率，奖励整体平移不会改变差。',
      '同一对奖励都加 2，偏好概率会变吗？',
      '偏好概率依赖差值，不依赖奖励绝对零点。',
    ),
    lesson(
      'DPO',
      String.raw`L=-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right]\right)`,
      '固定回答0为偏好项、回答1为拒绝项，比较当前与参考策略的 log 概率比。用稳定 softplus 计算损失；偏好标签不会由本页奖励自动改写。',
      '提高被偏好回答相对于参考策略的概率，损失如何变化？',
      'DPO 从偏好对直接构造策略损失；这里没有另训奖励模型。',
    ),
    lesson(
      'PPO-Clip',
      String.raw`J=\min(\rho A,clip(\rho,1-\epsilon,1+\epsilon)A)`,
      '单独编辑比率和优势，检查正负优势下 min 的方向。裁剪限制目标的激励，不保证策略比率实际永远落在区间内。',
      'A 为负且比率低于下限时，哪一项被 min 选中？',
      '不能把所有情况都简写成 clip(ρ)A。',
    ),
    lesson(
      '组内优势',
      String.raw`A_i=\frac{r_i-\bar r}{\sqrt{\frac1G\sum_j(r_j-\bar r)^2}}`,
      'GRPO 的组是同一个 prompt 的多个回答。本例明确用总体标准差，零方差时优势约定为0；不是对不同 prompt 混合标准化。',
      '奖励全部相同，为什么不能直接除以标准差？',
      '组内中心化和尺度一起构成相对优势。',
    ),
    lesson(
      'GRPO 约束',
      String.raw`J=\frac1G\sum_i\left[\min(\rho_iA_i,clip(\rho_i)A_i)-\beta k_i\right]`,
      '用旧策略构造比率，用参考策略构造 R1 形式的样本 KL 估计项。四个完整回答在本例各被简化成一个动作，省略逐 token 平均与优化过程。',
      '优势都为0时，参考约束是否一定也为0？',
      '相对奖励项与参考策略约束是两个独立贡献。',
    ),
    lesson(
      '平移与蒸馏',
      String.raw`k_i=\frac{\pi_{ref,i}}{\pi_{\theta,i}}-\log\frac{\pi_{ref,i}}{\pi_{\theta,i}}-1`,
      '先预测共同平移奖励是否保持组优势。再查看玩具教师概率与学生概率的交叉熵，区分蒸馏监督和组奖励目标。',
      '奖励平移、策略概率变化、教师目标变化分别影响哪条路径？',
      '公开目标可计算；不能推断闭源模型的内部训练或隐藏推理。',
    ),
  ],
  examples: [
    { name: '不同奖励', input: [0, 1, 2, 3] },
    { name: '零方差组', input: [2, 2, 2, 2] },
  ],
  compute: preference,
  prediction: {
    key: 'shift',
    target: 2,
    question: '奖励共同加常数，组内优势会保持不变吗？',
    label: 'GRPO 组内优势',
  },
  sources: [
    { title: 'DPO Eq.7', url: 'https://arxiv.org/html/2305.18290v3' },
    { title: 'PPO 作者文档', url: 'https://spinningup.openai.com/en/latest/algorithms/ppo.html' },
    { title: 'DeepSeek-R1 §2.2.1', url: 'https://arxiv.org/html/2501.12948v1' },
  ],
  limits: [
    '只计算给定样本目标，没有采样、critic、GAE、反向或优化器。',
    'R1 原文 std 未指定总体或样本分母；本页明确用总体标准差，不冒充精确训练代码。',
    'KL 样本估计项不等于单个回答上的完整分布 KL。蒸馏分布为教学构造，没有隐藏推理内容。',
  ],
} satisfies TopicDefinition;
