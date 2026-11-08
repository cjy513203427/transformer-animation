import { formatNumber as fmt } from '../../math/attention';
import { sampling } from '../../math/p1/sampling';
import { BOOK } from '../book';
import { choice, field, lesson, vec, type TopicDefinition } from './types';
export default {
  id: 'decoding',
  path: '/decoding',
  title: '解码：从分数到下一个 token',
  chapter: '第 16 章、附录 B',
  intro:
    '在三项词表上改变温度、Top-K 和 Top-P。固定种子重放一次采样，再比较两步序列中的贪心与束搜索。',
  question: '同一个问题：同一个概率分布，怎样选出下一个 token',
  headline: '概率 → 温度 → Top-K → Top-P → 抽样 → 两步搜索。',
  why: {
    question: '模型给出了分数，怎样选出下一个 token？',
    points: [
      '模型每一步输出一个 logits 向量，长度等于词表大小。第一步是确定的：用 Softmax 把 logits 变成概率。第二步有很多选法：总选最大的、按概率抽、只在前几名里抽……这就是解码策略。',
      '不同策略在“稳定”和“多样”之间取舍。本页用 a、b、c 三个候选手算；默认数字是附录 B.4.2 的 logits [2.0, 1.0, 0.5]，温度为 1 时概率约为 [0.63, 0.23, 0.14]，可与书中表格对照。',
    ],
    book: ['附录 B.1 · 为什么需要解码策略', BOOK.appB],
  },
  input: [2, 1, 0.5],
  inputNames: ['a 的 logit', 'b 的 logit', 'c 的 logit'],
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
    lesson({
      name: '分数分布',
      title: '先把 logits 变成概率',
      formula: String.raw`p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}`,
      explanation: (c) =>
        `三个候选 a、b、c 的 logits 是 ${vec(c.input)}，Softmax 后的概率为 ${vec(c.values('Softmax(logits)'))}，和为 1。a、b、c 是玩具词表，没有语义，也没有调用语言模型。解码只决定怎样从这个分布里选，不会改变模型权重。`,
      prompt: '概率最高的 token 一定会被选中吗？',
      takeaway: '概率分布是解码策略的输入；策略决定怎样从中选。',
      book: ['附录 B.1.1 · 从 Logits 到 Token', BOOK.appB],
    }),
    lesson({
      name: '温度',
      title: '温度：先除以 τ，分布变尖或变平',
      formula: String.raw`p_i(\tau)=softmax(z/\tau)_i,\quad\tau>0`,
      explanation: (c) =>
        `当前 τ = ${fmt(c.metric('温度 τ'))}，温度分布为 ${vec(c.values('温度分布'))}。τ < 1 时高分更占优，分布变尖；τ > 1 时更平均。书中 [2.0, 1.0, 0.5] 这组 logits 在 τ = 0.5 时约为 [0.84, 0.11, 0.04]，τ = 2 时约为 [0.48, 0.29, 0.23]。所有分数相同时，温度改变不了均匀分布。`,
      prompt: '给所有 logits 加同一个常数，温度后的概率会变吗？',
      takeaway: '温度控制分布的集中程度，不是质量开关。',
      book: ['附录 B.4.2 · Temperature 的效果', BOOK.appB],
    }),
    lesson({
      name: 'Top-K',
      title: 'Top-K：只留前 K 名，再重新分配概率',
      formula: String.raw`A=TopK(p,K),\quad p'_i=\frac{p_i\mathbf1[i\in A]}{\sum_{j\in A}p_j}`,
      explanation: (c) =>
        `按概率从大到小保留 K = ${fmt(c.config.k)} 个（当前保留 ${c.text('保留 ID')}），删掉其余项后，把留下的概率除以它们的和，让总和回到 1。附录 B.5.2 的例子：保留 [0.4, 0.3, 0.15] 后变成 [0.47, 0.35, 0.18]。并列时本例按 token 编号取，恰好留下 K 个。`,
      prompt: '过滤后为什么不能直接用原来的概率采样？',
      takeaway: '先选集合，再重新归一化，是两个步骤。',
      book: ['附录 B.5 · Top-K Sampling', BOOK.appB],
    }),
    lesson({
      name: 'Top-P',
      title: 'Top-P：累计概率跨过阈值的那一项也要留下',
      formula: String.raw`m=\min\{k:\sum_{i=1}^k p_{(i)}\ge P\}`,
      explanation: (c) =>
        `在 Top-K 之后的分布上，按概率从大到小累加，直到累计值第一次 ≥ P = ${fmt(c.metric('阈值 P'))}；跨过阈值的那一项也保留。当前保留 ${c.text('保留 ID')}，累计 ${fmt(c.metric('截止累计质量'))}。附录 B.6.2：[0.4, 0.3, 0.15, 0.08, …] 在 P = 0.9 时保留前 4 个，因为累计到第 4 个才是 0.93。`,
      prompt: '阈值 0.6，第一项 0.5，第二项 0.3，应该保留几项？',
      takeaway: '跨过阈值的 token 必须保留，然后重新归一化。',
      book: ['附录 B.6 · Top-P（Nucleus）Sampling', BOOK.appB],
    }),
    lesson({
      name: '种子采样',
      title: '抽一次：随机数落在哪一段，就选谁',
      formula: String.raw`u\sim U[0,1),\quad i=\min\{k:\sum_{j\le k}p'_j>u\}`,
      explanation: (c) =>
        `把筛选后的概率按 a、b、c 的顺序排成首尾相接的区间，用固定种子生成 u = ${fmt(c.metric('u ∈ [0,1)'), 4)}，它落在哪个区间就选哪个：这次选中 ${c.text('本次 token')}。同样的分布、策略和种子可以完全重放；一次抽样不能说明长期频率。`,
      prompt: '相同输入、参数与种子，为什么可以重放？',
      takeaway: '可复现需要同时记下分布、策略和随机种子。',
      book: ['附录 B.3 · Random Sampling', BOOK.appB],
    }),
    lesson({
      name: '短序列搜索',
      title: '多看一步：贪心与束搜索可能选出不同的序列',
      formula: String.raw`\log p(y_1,y_2)=\log p(y_1)+\log p(y_2\mid y_1)`,
      explanation: (c) => {
        const greedy = c.text('Greedy'),
          best = c.text('穷举最优');
        return `把两步的概率相乘，比较整条路径。贪心每步都取最大：${greedy}；Beam-2 每步保留两条最好的前缀：${c.text('Beam-2')}；穷举 9 条路径的最优是 ${best}。${greedy.startsWith(best + ' ') ? '这组数里贪心恰好也是最优；调高温度或换预设再看。' : '贪心第一步取了最大，整条路径却不是最优。'}最后先预测：收紧 Top-P，筛选后的分布会不会变？`;
      },
      prompt: '为什么第一步概率最大的前缀，整条路径不一定最大？',
      takeaway: '束宽限制了搜索范围，不能普遍保证找到全局最优。',
      book: ['附录 B.7 · Beam Search', BOOK.appB],
    }),
  ],
  examples: [
    { name: '书中 B.4.2：[2, 1, 0.5]', input: [2, 1, 0.5], book: true },
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
    { title: '原书附录 B · 解码策略详解', url: BOOK.appB },
    { title: '原书第 16 章 · 训练与推理的异同', url: BOOK.ch16 },
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
  bookNotes: [
    '附录 B.6.1 的代码以“累计概率 ≥ p 的第一个位置”为截止并保留该位置，与本页规则一致。',
    '附录 B 的 Top-K、Top-P 例子有 6 个候选；本页词表只有 3 个，说明里引用书中数字，交互用 3 个候选核对同一规则。',
  ],
} satisfies TopicDefinition;
