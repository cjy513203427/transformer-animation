import { memorySymbols } from '../notation/memory';
import { formatNumber as fmt } from '../../math/attention';
import { memory } from '../../math/p1/memory';
import { BOOK } from '../book';
import { choice, field, lesson, type TopicDefinition } from './types';
import { T } from '../../i18n';
const LONGFORMER = 'https://arxiv.org/html/2004.05150v2',
  LINEAR = 'https://arxiv.org/html/2006.16236v3',
  INFINI = 'https://arxiv.org/html/2404.07143v1';
export default {
  id: 'memory',
  path: '/attention-variants',
  title: T('历史访问：稀疏连接与压缩状态'),
  chapter: T('第 24 章'),
  intro: T(
    '同一份 Q、K、V，先用完整 Attention 算一遍，再只看一个滑动窗口算一遍，比一比差在哪里。接着一步步核对线性 Attention 的状态递推和 Infini-attention 的分段记忆，看看“少连几条线”和“把历史压缩”各要付出什么代价。',
  ),
  question: T('这一页想弄明白：历史太长时，怎样少看一些，或把它压缩起来'),
  headline: T('历史太长时，要么只看一部分，要么压缩成一个小状态：两种省法，代价不同。'),
  why: {
    question: T('上下文越来越长，每个 token 都要看全部历史吗？'),
    points: [
      T(
        '完整 Attention 里，每个位置都要和所有可见历史打分，配对数随长度按平方增长；生成时 KV 缓存也随长度线性增长（原书 24.1）。',
      ),
      T(
        '三类思路：稀疏 Attention 只连最近或重要的位置；Linear Attention 换一种打分方式，把历史压进固定大小的状态；Infini-attention 把旧片段压进记忆，再与当前片段的局部 Attention 合并。它们省下的东西不同，代价也不同。',
      ),
    ],
    analogy: {
      title: T('读小说不必回看每一页（24.2.1）'),
      text: T(
        '读到第 500 页时，你不会同时回看前 499 页的每个字，只会记得最近读过的内容、章节开头的关键句和重要人物。稀疏 Attention 就是只让每个 token 关注一部分位置。',
      ),
    },
    book: [T('第 24.1、24.2.1 节 · 为什么 n² 是瓶颈'), BOOK.ch24],
  },
  input: [1, 0, 0, 1, 1, 1, -1, 0.5],
  inputNames: ['t0·d0', 't0·d1', 't1·d0', 't1·d1', 't2·d0', 't2·d1', 't3·d0', 't3·d1'],
  fields: [
    field('window', T('因果窗口 w'), 2, 1, 4, 1, [1, 5]),
    field('gate', T('记忆门 logit β'), 0, -4, 4, 0.5, [4]),
    choice(
      'delta',
      T('Infini 更新'),
      0,
      [
        { value: 0, label: 'Linear' },
        { value: 1, label: 'Delta' },
      ],
      [4],
    ),
  ],
  steps: [
    lesson({
      name: T('完整历史'),
      title: T('基线：每个位置读取全部可见历史'),
      symbols: memorySymbols[0],
      formula: String.raw`P=softmax(QK^T/\sqrt{d_k}+M),\quad O=PV`,
      explanation: T(
        '先固定一份 Q、K、V：Q 是输入的四个二维行，K 把这两维交换，V 就是输入本身（教学构造）。因果 Mask 下，第 t 行可以读取 t + 1 个位置。左图每条线是一次读取，线越粗权重越大。后面几步都和这个基线对照。',
      ),
      prompt: T('一行可以读取几个历史位置？'),
      takeaway: T('先固定基线，才能分清改变的是算法还是执行方式。'),
      book: [T('第 24.1.1 节 · 从数字看问题'), BOOK.ch24],
    }),
    lesson({
      name: T('稀疏窗口'),
      title: T('滑动窗口：只看最近 w 个位置'),
      symbols: memorySymbols[1],
      formula: String.raw`\max(0,t-w+1)\le j\le t`,
      explanation: (c) =>
        T(
          '窗口 w = {#0}（含自身）：每个位置只保留最近的 Key，Softmax 只在这几个位置里分配权重，所以分母和输出都会变。原书 24.8.2 的估算：N = 8、d = 4、w = 3 时乘法从约 512 次降到约 192 次。本页用稠密矩阵加 Mask 核对数学，不代表实际省下了计算；省不省要看实现。',
          fmt(c.metric(T('窗口 w（含自身）'))),
        ),
      prompt: T('缩小窗口，分母和输出为什么都会变？'),
      takeaway: T('稀疏连接限制可见集合；能否真正省计算取决于实现。'),
      book: [T('第 24.4 节 · Sliding Window Attention'), BOOK.ch24],
      source: ['Longformer §3', LONGFORMER],
    }),
    lesson({
      name: T('线性核'),
      title: T('Linear Attention：换一种打分方式'),
      symbols: memorySymbols[2],
      formula: String.raw`w_{tj}=\phi(q_t)\cdot\phi(k_j),\quad\phi(x)=ELU(x)+1`,
      explanation: T(
        '把 Softmax 里的 exp(q·k) 换成两个正值特征的点积 φ(q)·φ(k)，其中 φ(x) = ELU(x) + 1。这样可以利用矩阵乘法的结合律，先把 K 和 V 合起来（原书 24.5.2）。这一步先按定义显式算一遍，作为下一步递推的基线。它是另一种 Attention，输出一般和 Softmax 版不同。',
      ),
      prompt: T('为什么这一行不能要求等于标准 Softmax 的输出？'),
      takeaway: T('Linear Attention 换了打分核，不是 Softmax 的无损改写。'),
      book: [T('第 24.5.2 节 · 矩阵乘法的结合律'), BOOK.ch24],
      source: ['Linear Attention Eq.7–12', LINEAR],
    }),
    lesson({
      name: T('状态递推'),
      title: T('把历史累加进固定大小的状态'),
      symbols: memorySymbols[3],
      formula: String.raw`S_t=S_{t-1}+\phi(k_t)^Tv_t,\ z_t=z_{t-1}+\phi(k_t),\ o_t=\frac{\phi(q_t)S_t}{\phi(q_t)z_t^T}`,
      explanation: (c) =>
        T(
          '逐个读入 token：把 $\\phi(k_t)^\\top v_t$ 加进矩阵 S，把 $\\phi(k_t)$ 加进向量 z；读出时用 $\\phi(q_t)$ 分别乘它们再相除。无论读了多少 token，S 和 z 一共只有 {#0} 个数。表中递推结果与上一步的显式计算逐项一致。',
          fmt(c.metric(T('状态标量数'))),
        ),
      prompt: T('状态大小依赖序列长度 T 吗？为什么训练能组织成扫描？'),
      takeaway: T('固定特征维度时，历史被压进固定大小的状态，扫描工作随 T 线性增长。'),
      book: [T('第 24.5.4 节 · Linear Attention 的实现'), BOOK.ch24],
      source: ['Linear Attention Eq.7–12', LINEAR],
    }),
    lesson({
      name: T('Infini 分段'),
      title: T('Infini-attention：旧片段进记忆，当前片段局部算'),
      symbols: memorySymbols[4],
      formula: String.raw`A=\sigma(\beta)A_{mem}+(1-\sigma(\beta))A_{local}`,
      explanation: (c) =>
        T(
          '每 2 个 token 一段。处理一段时，先从上一段留下的记忆里读出一份结果，再和本段内部的因果 Attention 按门控合并：记忆占 σ(β) = {#0}。读完才把本段写入记忆，同一段都读同一个旧记忆。当前更新规则 {#1}：Linear 直接写入 V；Delta 只写入记忆还没预测准的部分。',
          fmt(c.metric(T('记忆门 sigmoid(β)'))),
          c.text(T('更新规则')),
        ),
      prompt: T('把记忆更新放在当前段读取之前，会泄漏什么信息？'),
      takeaway: T('局部 Attention 与压缩记忆分工合作，不能画成一个无限长的完整缓存。'),
      book: [T('第 24.6.2 节 · Infini Attention 的核心思想'), BOOK.ch24],
      source: ['Infini-attention Eq.3–6', INFINI],
    }),
    lesson({
      name: T('压缩边界'),
      title: T('压缩会丢信息：两段不同的历史，可能得到同一个状态'),
      symbols: memorySymbols[5],
      formula: String.raw`M_{new}=M_{old}+\phi(K)^T(V-\widehat V_{old})`,
      explanation: T(
        '表中两段历史的 V 不同，压缩后的 S 和 z 却完全相同，之后读出时就无法区分它们。原书 24.9 的误区 3 也提醒：Infini Attention 不能无损记住所有历史。最后先预测：只扩大窗口，当前稀疏输出会不会变？',
      ),
      prompt: T('两段不同的历史压到同一个状态后，还能恢复各自的值吗？'),
      takeaway: T('固定大小的状态会丢信息；能算更长的序列，不等于能可靠记住。'),
      book: [T('第 24.9 节 · 常见误区'), BOOK.ch24],
    }),
  ],
  examples: [
    { name: T('四个位置'), input: [1, 0, 0, 1, 1, 1, -1, 0.5] },
    { name: T('重复内容'), input: [1, 1, 1, 1, 1, 1, 1, 1] },
  ],
  compute: memory,
  prediction: {
    key: 'window',
    target: 4,
    question: T('只扩大窗口，当前稀疏输出会保持不变吗？'),
    label: T('窗口 Attention 输出'),
  },
  sources: [
    { title: T('原书第 24 章 · Sparse 与 Infinite Attention'), url: BOOK.ch24 },
    { title: 'Longformer §3', url: LONGFORMER },
    { title: 'Linear Attention Eq.7–12', url: LINEAR },
    { title: 'Infini-attention Eq.3–6', url: INFINI },
  ],
  limits: [
    T('窗口只是稀疏连接示例，没有实现完整 Longformer 或块稀疏 GPU kernel。'),
    T('Infini 的片段长度固定2；初始空记忆分母为0时返回零。分段更新不逐 token 读取新记忆。'),
    T('本页保留过程数据以便核对，不是节省内存的运行时；不测量 GPU 吞吐。'),
  ],
  bookNotes: [
    T(
      '原书 24.8.2 的 w = 3 计算量是粗略估算：因果窗口下前两个位置的可见邻居不足 3 个，实际配对更少。',
    ),
    T('原书 24.8 只数主要矩阵乘法，是理论估算；本页同样只核对数学结果，不测量运行时间。'),
  ],
} satisfies TopicDefinition;
