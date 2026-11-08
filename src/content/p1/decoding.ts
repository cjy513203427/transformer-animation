import { sampling } from '../../math/p1/sampling';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'decoding',
  path: '/decoding',
  title: '解码：从分数到下一个 token',
  chapter: '第 16 章、附录 B',
  intro:
    '在三项词表上改变温度、Top-K 和 Top-P。固定种子重放一次采样，再比较两步序列中的贪心与束搜索。',
  input: [Math.log(0.5), Math.log(0.3), Math.log(0.2)],
  fields: [
    field('temperature', '温度 τ', 1, 0.25, 4, 0.25),
    choice(
      'k',
      'Top-K',
      3,
      [
        { value: 1, label: '1' },
        { value: 2, label: '2' },
        { value: 3, label: '全部3' },
      ],
      [2, 3, 4, 5],
    ),
    field('topP', 'Top-P', 1, 0.1, 1, 0.05, [3, 4, 5]),
  ],
  steps: [
    lesson(
      '分数分布',
      String.raw`p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}`,
      '先从原始 logits 得到候选概率。词表只有 a、b、c，不带语义，排序或采样均不会改变模型权重。',
      '最高概率意味着一定采到它吗？',
      '概率分布是解码策略的输入。',
    ),
    lesson(
      '温度',
      String.raw`p_i(\tau)=softmax(z/\tau)_i,\quad\tau>0`,
      '先除温度再做 Softmax。非并列高分在低温度下更占优势；同分不会被温度自动打破。',
      '给所有 logits 加同一个常数，会改变温度后的概率吗？',
      '温度控制分布集中程度，不等于质量开关。',
    ),
    lesson(
      'Top-K',
      String.raw`A=TopK(p,K),\quad p'_i=\frac{p_i\mathbf1[i\in A]}{\sum_{j\in A}p_j}`,
      '按降序保留 K 项，本例并列按 token 编号，恰好保留 K 个。删除其余项后重新归一化。',
      '过滤后为什么不能继续使用原概率直接采样？',
      '选择集合与重新归一化是两个步骤。',
    ),
    lesson(
      'Top-P',
      String.raw`m=\min\{k:\sum_{i=1}^k p_{(i)}\ge P\}`,
      '在当前 Top-K 分布上降序累计，保留首次达到阈值的那一项。不能只留下累计值仍小于阈值的前缀。',
      '阈值 0.6，第一项概率 0.5，第二项 0.3，应该保留几项？',
      '跨过阈值的 token 必须保留。',
    ),
    lesson(
      '种子采样',
      String.raw`u\sim U[0,1),\quad i=\min\{k:\sum_{j\le k}p'_j>u\}`,
      '用固定种子得到 u，按 token 编号排列概率区间，选中覆盖它的区间。这里展示一次抽样，不拿一次结果证明分布频率。',
      '相同输入、参数与种子，为什么可以重放？',
      '可复现需要同时保存分布、策略和随机种子。',
    ),
    lesson(
      '短序列搜索',
      String.raw`\log p(y_1,y_2)=\log p(y_1)+\log p(y_2\mid y_1)`,
      '比较两步树。Greedy 每步局部最大，Beam-2 暂存两条前缀，本例可用穷举九条路径核对。先预测 Top-P 改动是否保持筛选分布。',
      '为什么 aa 的第一步更大，ba 的整条路径仍可能更大？',
      '束宽限制搜索，不能普遍保证找到全局最优。',
    ),
  ],
  examples: [
    { name: '0.5 / 0.3 / 0.2', input: [Math.log(0.5), Math.log(0.3), Math.log(0.2)] },
    { name: '全部同分', input: [0, 0, 0] },
  ],
  compute: sampling,
  prediction: {
    key: 'topP',
    target: 0.6,
    question: '只收紧 Top-P，当前筛选分布会保持不变吗？',
    label: '筛选后概率',
  },
  sources: [
    { title: 'Nucleus Sampling 原论文', url: 'https://arxiv.org/abs/1904.09751' },
    {
      title: 'Transformers 解码实现',
      url: 'https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py',
    },
    {
      title: '官方生成策略',
      url: 'https://huggingface.co/docs/transformers/main/en/generation_strategies',
    },
  ],
  limits: [
    '明确采用先温度、再 Top-K、再 Top-P 的顺序；库与模型可能提供其他组合。',
    '束搜索固定长度2，无 EOS、长度惩罚或重复惩罚；条件概率表固定，第一步分布可编辑。',
    '本例恰好 K 个的同分规则与部分库的阈值规则可能不同。',
  ],
} satisfies TopicDefinition;
