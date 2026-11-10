import { formatNumber as fmt } from '../../math/attention';
import { lora } from '../../math/p1/lora';
import { BOOK } from '../book';
import { choice, field, lesson, vec, type TopicDefinition } from './types';
import { T } from '../../i18n';
const pair = (values: number[], i: number) => values.slice(i * 2, i * 2 + 2);
const QLORA = 'https://arxiv.org/html/2305.14314v1';
export default {
  id: 'lora',
  path: '/lora',
  title: T('LoRA：冻结底座，更新从哪里进入？'),
  chapter: T('第 26 章'),
  intro: T(
    '用两个二维输入核对冻结的底座与低秩旁路。改变 rank 和更新强度 α，再比较合并权重与量化底座两条路径。',
  ),
  question: T('同一个问题：不改原权重，模型的输出怎样被调整'),
  headline: T('冻结底座 → 低秩旁路 → 输出相加 → 合并权重 → QLoRA → 账本。'),
  why: {
    question: T('为什么不直接改全部参数，而要加一条低秩旁路？'),
    points: [
      T(
        '全参数微调 7B 模型，光是参数、梯度和 Adam 状态就要约 56GB 显存；每做一个任务，还要另存一整份 14GB 的模型（原书 26.1.1）。',
      ),
      T(
        '研究者发现，微调时权重的变化量 ΔW 往往可以用两个小矩阵相乘来近似。LoRA 因此冻结原权重 W₀，只训练这两个小矩阵。原书 26.2.1 的例子：1024 × 512 的矩阵有 524,288 个参数；r = 32 时两个小矩阵共 32,768 + 16,384 = 49,152 个，约为原来的 9.4%。',
      ),
    ],
    book: [T('第 26.1、26.2.1 节 · 为什么需要高效微调'), BOOK.ch26],
  },
  input: [1, 2, -1, 1],
  inputNames: ['t0·d0', 't0·d1', 't1·d0', 't1·d1'],
  fields: [
    choice('rank', 'rank r', 1, [
      { value: 1, label: '1' },
      { value: 2, label: '2' },
    ]),
    field('alpha', T('更新强度 α'), 1, 0, 4, 0.25),
  ],
  steps: [
    lesson({
      name: T('冻结底座'),
      title: T('原来的权重 W₀ 不动，但照样参与计算'),
      formula: String.raw`Y_0=XW_0`,
      explanation: (c) => {
        const base = c.values('XW₀');
        return T(
          '输入 X 有两行，即两个 token，各 2 维。底座输出 $XW_0$：t0 = {#0}，t1 = {#1}。“冻结”只表示训练时不更新 $W_0$；前向时它照样要乘，也照样要存。',
          vec(pair(base, 0)),
          vec(pair(base, 1)),
        );
      },
      prompt: T('为什么冻结后还需要保存并读取 W₀？'),
      takeaway: T('冻结等于不更新，不等于不计算、不存储。'),
      book: [T('第 26.2.3 节 · LoRA 的关键设计'), BOOK.ch26],
    }),
    lesson({
      name: T('低秩旁路'),
      title: T('新增一条窄旁路：先压到 r 维，再还原'),
      formula: String.raw`A\in\mathbb R^{D\times r},\quad B\in\mathbb R^{r\times K},\quad\Delta W=\frac\alpha r AB`,
      explanation: (c) =>
        T(
          '输入先乘 A，从 2 维压到 r = {#0} 维（表中的 XA），再乘 B 回到输出维度；整条旁路再乘缩放 α/r = {#1}。原书写作 B @ A，是列向量写法，先作用的也是 A；本页按行向量写成 XAB，顺序相同。本例的 A、B 是非零教学数字，相当于已经训练过一段时间。',
          fmt(c.metric('rank r')),
          fmt(c.metric(T('缩放 α/r'))),
        ),
      prompt: T('低秩约束限制的是 W₀ 还是 ΔW？'),
      takeaway: T('rank 限制的是更新 ΔW，不是对 W₀ 的无损压缩。'),
      book: [T('第 26.4.1 节 · 完整的 LoRA 公式'), BOOK.ch26],
    }),
    lesson({
      name: T('输出相加'),
      title: T('两条支路的输出逐项相加'),
      formula: String.raw`Y=XW_0+\frac\alpha r(XA)B`,
      explanation: (c) => {
        const base = c.values('XW₀'),
          delta = c.values('(α/r)XAB'),
          total = c.values(T('LoRA 输出'));
        return T(
          't0：底座 {#0} + 更新 {#1} = {#2}。α = 0 时更新支路为 0，输出就是原模型。原书 26.4.2 把 B 初始化为全零，也是为了让训练刚开始时模型和原来完全一样。',
          vec(pair(base, 0)),
          vec(pair(delta, 0)),
          vec(pair(total, 0)),
        );
      },
      prompt: T('只改 α，底座输出 XW₀ 会变吗？'),
      takeaway: T('更新只从旁路进入输出；底座的数值没有被改动。'),
      book: [T('第 26.4.4 节 · 前向传播'), BOOK.ch26],
    }),
    lesson({
      name: T('rank 与合并'),
      title: T('训练完，可以把更新并回权重'),
      formula: String.raw`XW_0+X\Delta W=X(W_0+\Delta W)`,
      explanation: (c) =>
        T(
          '因为 $XW_0 + X\\Delta W = X(W_0 + \\Delta W)$，训练结束后可以把 ΔW 直接加进 $W_0$，推理时就不再需要旁路；左图的结果与上一步相同。适配器参数量是 r(D + K)，当前为 {#0} 个，底座 {#1} 个。本例 D = K = 2 太小，r = 2 时已没有节省；原书 1024 × 512、r = 32 的例子才省到约 9.4%。',
          fmt(c.metric(T('适配器参数'))),
          fmt(c.metric(T('底座参数'))),
        ),
      prompt: T('合并为什么成立？更大的 rank 一定效果更好吗？'),
      takeaway: T('合并是代数恒等式，可以验证；任务效果要靠训练和评估。'),
      book: [T('第 26.3 节 · Rank 的选择'), BOOK.ch26],
    }),
    lesson({
      name: T('QLoRA 数据流'),
      title: T('QLoRA：底座用 4 bit 存，适配器保持高精度'),
      formula: String.raw`Y=X\operatorname{dequant}(W_{0,q})+\frac\alpha r(XA)B`,
      explanation: T(
        'QLoRA 把冻结的底座压成 4 bit 存储，计算前再还原到计算精度；A、B 仍用高精度训练。左图的底座输出来自量化后还原的 $W_0$，所以和上一步略有差别。真实 QLoRA 用 NF4 格式，本页用均匀 4 bit 代替，只说明三种精度的分工。',
      ),
      prompt: T('存储位宽、计算精度和适配器训练精度为什么不能混为一谈？'),
      takeaway: T('QLoRA 的量化底座仍然冻结，只有适配器参与训练。'),
      book: [T('第 26.6.2 节 · QLoRA 的核心思想'), BOOK.ch26],
      source: ['QLoRA §3', QLORA],
    }),
    lesson({
      name: T('账本与对照'),
      title: T('账本：底座、适配器与计算分开算'),
      formula: String.raw`N_{adapter}=r(D+K)`,
      explanation: T(
        '表中分别列出底座、量化底座与适配器的字节数。先预测：只改变 α，LoRA 输出会不会变？α 缩放的是更新旁路，所以输出通常会变；只有旁路恰好为 0 时才不变。',
      ),
      prompt: T('α = 0 时是什么模型？B 全零初始化怎样保证训练开始时输出不变？'),
      takeaway: T('底座与更新分开核算；训练的参数少，不等于整个模型不用存储。'),
      book: [T('第 26.5.3 节 · 参数量对比'), BOOK.ch26],
    }),
  ],
  examples: [
    { name: T('两行输入'), input: [1, 2, -1, 1] },
    { name: T('零更新（α = 0）'), input: [1, 2, -1, 1], parameters: { alpha: 0 } },
  ],
  compute: lora,
  prediction: {
    key: 'alpha',
    target: 2,
    question: T('只改变 α，当前 LoRA 输出会保持不变吗？'),
    label: T('LoRA 输出'),
  },
  sources: [
    { title: T('原书第 26 章 · LoRA 与 QLoRA'), url: BOOK.ch26 },
    { title: 'LoRA §4.1', url: 'https://arxiv.org/html/2106.09685v2' },
    { title: T('QLoRA §3 与附录E'), url: QLORA },
  ],
  limits: [
    T('没有反向、优化器或真实微调数据；二维教学矩阵只演示一个线性层，A、B 是固定的非零教学数字。'),
    T(
      '量化分支以对称均匀 4bit 代替 NF4，没有 Double Quantization 或 Paged Optimizers，因此只展示 QLoRA 的简化数据流。',
    ),
    T('JS float64 计算与账本中的假想 float32 存储明确分开；账本不是 GPU dtype 或显存实测。'),
  ],
  bookNotes: [
    T(
      '原书 26.4.1 按列向量写 $W_{orig} + (\\alpha/r)BA$（A 为 r × D，B 为 N × r）；本页用行向量写 $XW_0 + (\\alpha/r)XAB$（A 为 D × r，B 为 r × K）。两种写法都是先用 A 降到 r 维，再用 B 还原。',
    ),
    T('原书 26.2.1 说参数“只有原来的 9%”，精确值是 49,152 ÷ 524,288 = 9.375%。'),
    T(
      '原书 26.4.2 中 B 全零初始化，训练开始时 ΔW = 0。本页的 A、B 是非零教学数字，演示已经学到的更新；“零更新”预设用 α = 0 得到同样的起点输出。',
    ),
  ],
} satisfies TopicDefinition;
