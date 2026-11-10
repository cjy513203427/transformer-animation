import { chapterUrl } from './book';
import type { WhyContent } from '../components/WhyIntro';
import { T } from '../i18n';

const CH30 = chapterUrl('第30章-Mixture-of-Experts-稀疏激活的秘密/');
const SPARSE_MOE = 'https://arxiv.org/html/1701.06538v1';
const SWITCH = 'https://arxiv.org/html/2101.03961v3';
const GSHARD = 'https://arxiv.org/html/2006.16668v1';
const MIXTRAL_CODE =
  'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py';

/** Sources were read online on this date; the toy example is our own, untrained model. */
export const moeVerifiedDate = '2026-11-07';

export const moeSteps: {
  name: string;
  title: string;
  explanation: string;
  takeaway: string;
  prompt: string;
  formula: string;
  /** Paper or official code used to check the formula; absent when the book is the reference. */
  source?: [string, string];
  /** The book section that explains this step. */
  book: [string, string];
}[] = [
  {
    name: T('路由打分'),
    title: T('Router 怎样决定每个 token 去哪里？'),
    explanation: T(
      '每个 token 的隐藏状态是一行 $x_t$。路由器（Router）用同一组参数 $W_R$ 给四个专家打分，再沿专家轴做 Softmax，得到一行概率。修改某个 token 的输入，会改变它的路由分数；这里逐 token 独立路由，不读取其他 token。专家只用 E0–E3 编号，没有预先分配的语义职责。',
    ),
    takeaway: T('路由器的输入是隐藏状态，输出是对各专家的分数；分数来自计算。'),
    prompt: T('这一行有几个分数？Softmax 沿哪个轴？只改 t1 的输入，t0 的路由会变吗？'),
    formula: String.raw`Z=XW_R\in\mathbb{R}^{T\times E},\qquad p_{t,e}=\frac{\exp(z_{t,e}-m_t)}{\sum_j\exp(z_{t,j}-m_t)},\quad m_t=\max_j z_{t,j}`,
    source: [T('Switch Transformers §2.1 · Router 分数与概率'), SWITCH],
    book: [T('第 30.2.3 节 · Router（路由器）的作用'), CH30],
  },
  {
    name: T('Top-K 门控'),
    title: T('选择 K 个专家后，权重还要怎样处理？'),
    explanation: T(
      '本实验取概率最高的 K 个专家，再把选中概率除以它们的和，得到门控权重（Gate）。未选中的权重为 0，选中权重之和为 1；因此 K = 1 时唯一的门值就是 1。分数并列时选编号较小的专家。这是明确的教学变体：Softmax → Top-K → 重新归一化；Switch 的 Top-1 保留原概率，不能直接套用本例的门值。',
    ),
    takeaway: T('先选集合，再按本例规则重新归一化；Top-K 的选择和门值是两件事。'),
    prompt: T('只保留两个专家，原概率的和一定是 1 吗？为什么本例 Top-1 的门值恒为 1？'),
    formula: String.raw`A_t=\operatorname{TopK}(p_t,K),\qquad g_{t,e}=\begin{cases}p_{t,e}/\sum_{j\in A_t}p_{t,j},&e\in A_t\\0,&e\notin A_t\end{cases}`,
    source: [T('Transformers · MixtralTopKRouter 的选中后归一化'), MIXTRAL_CODE],
    book: [T('第 30.2.5 节 · Top-K 选择机制'), CH30],
  },
  {
    name: T('专家计算'),
    title: T('专家收到的是原输入，算出的是向量'),
    explanation: T(
      '选中的专家各自收到同一行输入，用独立参数执行两层前馈网络（FFN）：先从 2 维升到 3 维，经 ReLU，再回到 2 维。路由概率不作为专家输入。本例可单独缩放 E0 的输出：它会改变该分支的结果，固定输入和路由器时，专家选择与门值保持不变。',
    ),
    takeaway: T('Router 负责选路，专家负责变换输入；门值在之后的合并中使用。'),
    prompt: T('每个专家的两块权重是什么形状？只把 E0 的输出乘以 2，路由概率为什么不变？'),
    formula: String.raw`f_e(x_t)=\operatorname{ReLU}(x_tW_{1,e})W_{2,e},\qquad [1,2]\,[2,3]\to[1,3]\,[3,2]\to[1,2]`,
    source: [T('Shazeer et al. §2、§3.2 · 独立的前馈专家'), SPARSE_MOE],
    book: [T('第 30.2.4 节 · Expert（专家）网络'), CH30],
  },
  {
    name: T('加权合并'),
    title: T('把专家输出按原 token 身份合回来'),
    explanation: T(
      '每个被选中的专家返回一个 2 维向量；逐维乘对应的门值，再相加，得到这个 token 的输出。派发（Dispatch）后数据按专家分组，返回时必须用 token 编号放回原位置，再做合并（Combine）。本页并列核对分组执行与逐 token 直接计算，两条路径应在浮点容差内一致。',
    ),
    takeaway: T('合并的是同一个 token 的多条专家分支，token 的身份和原始顺序都要保留。'),
    prompt: T('t1 被两个专家处理，会产生两个输出 token 吗？返回顺序改变后怎样恢复正确结果？'),
    formula: String.raw`y_t=\sum_{e\in A_t}g_{t,e}f_e(x_t)\in\mathbb{R}^{1\times D},\qquad Y\in\mathbb{R}^{T\times D}`,
    source: [T('GShard §3.1 · Dispatch 与 Combine'), GSHARD],
    book: [T('第 30.2.2 节 · MoE 层的内部结构'), CH30],
  },
  {
    name: T('负载与参数'),
    title: T('只激活 K 个专家，为什么还要存全部参数？'),
    explanation: T(
      '切换分散与集中输入，观察每个专家实际接到多少条任务。每个 token 都选 K 个专家，任务总数仍为 T × K，分配却可能很不均匀。本例每个专家有 12 个参数，路由器有 8 个，总计 56 个；每 token 使用 8 + 12K 个。未选中的专家仍需保留权重，供后续输入使用。',
    ),
    takeaway: T('总参数决定要保存多少权重；激活参数和任务数描述本次计算，负载描述任务分布。'),
    prompt: T('K 从 1 变成 2，哪些参数账单不变？任务集中在一个专家，会自动变得更快吗？'),
    formula: String.raw`N_{\mathrm{all}}=DE+E(2DF),\quad N_{\mathrm{active/token}}=DE+K(2DF),\quad \sum_e n_e=TK`,
    book: [T('第 30.2.6 节 · 负载均衡问题'), CH30],
  },
  {
    name: T('专家并行'),
    title: T('权重留在设备上，输入过去、结果回来'),
    explanation: T(
      '双设备时，t0、t2 起始在设备 0，t1、t3 在设备 1；E0、E1 位于设备 0，E2、E3 位于设备 1。只有远端专家分支需要传输输入并返回结果，每个方向都是 D 个数。切到单设备，输出不变，跨设备通信为 0。图中表示逻辑数据流，没有中央中转节点；真实实现可以用 All-to-All 批量交换。',
    ),
    takeaway: T('专家并行搬运激活而非广播全部权重；通信量取决于路由、放置和数据大小。'),
    prompt: T(
      '同一个 token 选到本地和远端两个专家，各需要传什么？为什么箭头数量不能表示实际耗时？',
    ),
    formula: String.raw`B_{\mathrm{dispatch}}=RDb,\qquad B_{\mathrm{return}}=RDb,\qquad B_{\mathrm{total}}=2RDb`,
    source: ['Switch Transformers §5.4 · Expert Parallelism', SWITCH],
    book: [T('第 30.5.3 节 · 通信开销'), CH30],
  },
];

export const moeWhy: WhyContent = {
  question: T('为什么要准备很多个专家，却每次只用其中几个？'),
  points: [
    T(
      '原书 30.1.1 的例子：Mixtral 8x7B 总参数约 46.7B，但每个 token 只激活约 12.9B。MoE 把一个大的 FFN 换成多个“专家”FFN，再由路由器给每个 token 挑出少数几个来算。',
    ),
    T(
      '这样总参数可以很大，每个 token 的计算量却只随被选中的专家增长。代价是：全部专家的权重仍要存着，任务可能集中到少数专家，跨设备时还要来回搬运激活。本页用 4 个 token、4 个专家逐步手算这些账。',
    ),
  ],
  analogy: {
    title: T('分诊台与专科会诊（30.1.3）'),
    text: T(
      '全科医生要用全部知识看每个病人；专家会诊先经过分诊台（Router），按症状把病人分给最相关的两个专科（Top-2）。科室只是比喻：本页的专家只用 E0–E3 编号，没有预先分配的语义职责。',
    ),
  },
  book: [T('第 30.1 节 · MoE 的核心思想'), CH30],
};

/** Where the page deliberately differs from the book's wording, with the reason. */
export const moeBookNotes = [
  T(
    '原书 30.1.3 用“心内科、骨科”等科室比喻专家；真实专家的分工来自训练，不是人为指定。本页只用 E0–E3 编号。',
  ),
  T(
    '原书 30.2.5 讨论 Top-1、Top-2 的取舍，没有给出门值的归一化规则；本页明确采用“Softmax → Top-K → 在选中集合内重新归一化”，与 Transformers 的 Mixtral 路由实现一致。',
  ),
];

export const moeSources = [
  { title: T('原书第 30 章 · Mixture of Experts（选题与章节映射）'), url: CH30 },
  { title: T('Shazeer et al., 2017 · 稀疏门控 MoE，§2–4'), url: SPARSE_MOE },
  { title: T('Switch Transformers · §2.1 路由、§5.4 专家并行'), url: SWITCH },
  { title: T('GShard · §3.1 Dispatch / Combine、§3.3.2 AllToAll'), url: GSHARD },
  { title: T('Hugging Face Transformers · Mixtral 路由实现'), url: MIXTRAL_CODE },
];

export const moeMyths = [
  T('E0–E3 没有手工指定的“数学”“代码”等职责；本例权重是可复现的教学数字，不是真实模型数据。'),
  T('本例只演示一个 MoE 层：没有共享专家、容量上限、丢弃、重路由或辅助损失，也没有训练过程。'),
  T('本例 Top-1 重新归一化后门值为 1；其他架构可能保留原概率，门控规则必须随模型说明。'),
  T('稀疏激活不意味着只需存储本次选中的参数；权重卸载是另一种实现选择，本例未实现。'),
  T(
    '负载柱表示任务数量。MAC 计数只统计矩阵乘加，不包括 ReLU、Softmax、选路和加权合并，不是运行时间。',
  ),
  T(
    '通信账本固定每个元素 4 字节，只统计远端输入和返回向量；不含 token 编号、门值、协议、填充与训练反向通信。数值运算仍使用 JavaScript float64。',
  ),
  T('本例逐分支统计通信，不合并发给同一设备的重复输入；实际实现的批处理和去重可能改变字节数。'),
];
