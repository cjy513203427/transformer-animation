import { BOOK_URL } from './topics';

const CH21 = BOOK_URL + '第21章-Flash-Attention-内存优化原理/';
const PAPER = 'https://arxiv.org/html/2205.14135v2';
const OFFICIAL = 'https://github.com/Dao-AILab/flash-attention';

/** Online source check; the tensors, schedule ledger and explanations are our teaching example. */
export const flashVerifiedDate = '2026-11-08';

export const flashSteps: {
  name: string;
  title: string;
  explanation: string;
  takeaway: string;
  prompt: string;
  formula: string;
  source: [string, string];
}[] = [
  {
    name: '全量基线',
    title: '先把完整 Attention 算一遍',
    explanation:
      '本例已有 4 行 Q、K、V 激活，每行 2 维。先计算整张分数矩阵 S，在 Softmax 前屏蔽未来位置，再逐行沿 Key 轴归一化为 P，最后汇总 V。热力图的行是 Query、列是 Key。切换“观察 Query”，记下这一行的输出，后面用同一份输入核对分块结果。',
    takeaway: '分块前后要算同一个 Attention：Q、K、V、缩放和 Mask 都保持一致。',
    prompt: '一行 Query 对应几个分数？Softmax 沿哪个轴？因果 Mask 在归一化之前还是之后？',
    formula: String.raw`S=QK^\top/\sqrt{d_k}\in\mathbb{R}^{T\times T},\quad P=\operatorname{softmax}(S+M,\mathrm{axis=key}),\quad O=PV\in\mathbb{R}^{T\times d_v}`,
    source: ['FlashAttention §2.2 · 朴素 Attention', PAPER],
  },
  {
    name: '分块读取',
    title: '一个 Query 依次读取几行 K 和 V',
    explanation:
      '这里把 Query 块简化成一行，每次读取 B 行相同位置的 K、V，得到 1 × B 个局部分数。切换块大小，观察同一批 Key 怎样重新分组；尾块可以不足 B 行。块只是执行顺序，不能让每一块单独做完 Softmax 就直接平均，因为整行 Key 共享一个分母。',
    takeaway: 'K 和 V 按 token 一起切块；分块改变读取顺序，没有改变可见 Key 集合。',
    prompt: 'B 从 2 改成 3，最后一块有几行？每个块各自的概率和都是 1，能直接拼成整行 P 吗？',
    formula: String.raw`q_t\in\mathbb{R}^{1\times d_k},\quad K_b\in\mathbb{R}^{B\times d_k},\quad V_b\in\mathbb{R}^{B\times d_v},\quad s_b=q_tK_b^\top/\sqrt{d_k}+M_{t,b}`,
    source: ['FlashAttention §3.1 · Tiling', PAPER],
  },
  {
    name: '在线累积',
    title: '保留三个量，就能继续读下一块',
    explanation:
      '处理过的 Key 用三个状态概括：m 是已见最大分数，l 是当前尺度下的指数和，a 是尚未除以 l 的 Value 加权和。新块先选统一尺度，再把旧状态与本块贡献相加。界面中的 a/l 只是已处理 Key 的暂时输出；读完所有块，才是整行的最终结果。',
    takeaway: 'm 保证数值稳定，l 累积分母，向量 a 累积分子；三者一起更新。',
    prompt: '为什么只记一个已经归一化的输出还不够？还没读取的 Key 能提前出现在当前分母中吗？',
    formula: String.raw`m'=\max(m,\max_j s_{b,j}),\quad \alpha=e^{m-m'},\quad r_j=e^{s_{b,j}-m'},\quad l'=\alpha l+\sum_jr_j,\quad a'=\alpha a+\sum_jr_jv_j,\quad o'=a'/l'`,
    source: ['FlashAttention §3.1 · 分块归一化统计量', PAPER],
  },
  {
    name: '重缩放',
    title: '后面的分数更大，旧累积量怎样接上？',
    explanation:
      '递增分数示例让后面的块带来更大的最大值。此时 m 增大，旧指数要乘小于 1 的 α，换到新尺度；l 和 a 必须同时乘。先预测，再修改最后一个 Key 的第一维并执行，观察最大值和重缩放系数。Q、V 固定，这次改变只从 K 进入打分支路。',
    takeaway: '换尺度时，分母 l 与分子 a 一起缩放，才能保留旧 Key 的正确相对贡献。',
    prompt: '如果只缩放 l，不缩放 a，会发生什么？新块分数都更小时，α 应该是多少？',
    formula: String.raw`m'>m\Longrightarrow 0\le\alpha=e^{m-m'}<1,\qquad l_{\mathrm{old,new\ scale}}=\alpha l,\quad a_{\mathrm{old,new\ scale}}=\alpha a`,
    source: ['FlashAttention §3.1 · 统一最大值后的重缩放', PAPER],
  },
  {
    name: '等价核对',
    title: '把块读完，再与整行计算核对',
    explanation:
      '将最后一块后的 a 除以 l，与全量基线逐行、逐维比较。尝试不同块大小、混合分数和因果 Mask；全部屏蔽的块保持已有状态，不计算负无穷减负无穷。若整行没有有效 Key，本实现约定输出零向量。数学结果相同，浮点加法顺序不同，核对使用 1e-12 容差。',
    takeaway: 'Exact 表示没有改变完整 Softmax Attention，不承诺浮点结果逐比特相同。',
    prompt: '切块为什么不改变最终输出？因果 Mask 下，一个块没有任何有效 Key 时应该怎样更新？',
    formula: String.raw`O_{t,:}=a_t/l_t\quad(l_t>0),\qquad \max_{t,d}\left|O^{\mathrm{online}}_{t,d}-O^{\mathrm{full}}_{t,d}\right|\le10^{-12}`,
    source: ['FlashAttention §3.1 与官方实现 · Exact Attention', OFFICIAL],
  },
  {
    name: '存储与 I/O',
    title: '少保存中间矩阵，不等于少做配对',
    explanation:
      '本账本对照一种明确的教学调度：朴素路径保留 S、P，两张矩阵各写入和读出一次；逐行分块路径把块内分数立即用于更新状态，不把完整 S、P 写回。账本只数这部分中间载荷。Q、K、V、输出仍要读写，完整 Attention 的配对计算仍是二次规模；实际耗时还取决于硬件与 kernel。',
    takeaway: '算术量、中间存储、HBM 搬运与运行时间是四个问题，不能互相替代。',
    prompt:
      'S、P 不再写回，是否意味着 HBM 总读写为零？FlashAttention 会自动减少 Decode 的历史 KV 缓存吗？',
    formula: String.raw`N_{S,P}=2T^2,\quad N_{\mathrm{transfer},S,P}=4T^2\to0,\qquad \mathrm{work}=\Theta(T^2(d_k+d_v))`,
    source: ['FlashAttention §2.2、§3.2 · 中间矩阵与 I/O 分析', PAPER],
  },
];

export const flashSources = [
  { title: '原书第 21 章 · Flash Attention（选题与章节映射）', url: CH21 },
  { title: 'Dao et al., 2022 · FlashAttention，§2.2、§3.1–3.2、附录 B', url: PAPER },
  { title: 'Dao-AILab / flash-attention · 作者官方实现与使用约束', url: OFFICIAL },
];

export const flashMyths = [
  '本页是单头前向的教学计算，T = 4、d_k = d_v = 2，忽略 batch，禁用 Dropout；Q、K、V 是已有激活，不是训练参数或真实模型语义。',
  '本例固定 Query 块为一行，沿 Key 轴顺序扫描；真实 kernel 会同时处理多行 Query，并考虑片上空间和并行调度。',
  '递增分数预设固定，混合分数预设使用可复现种子；只改最后一个 K 的第一维不直接改 Q 或 V。',
  '初始 m 为负无穷，l = 0，a 为零向量；首个有效块的 α 约定为 0。全屏蔽块不改变状态；无有效 Key 的整行约定返回零向量。',
  '分块临时空间按 B(d_k+d_v+1)+d_k+2d_v+3 个标量估算：K/V 块、指数块、一个 Query、两份向量累积量与三个标量统计量；不含输入输出、可视化轨迹与运行时开销。小例子不保证该数一定更小。',
  '账本假设每元素 4 字节，实际数值计算使用 JavaScript float64；它是理想化调度计数，既不是浏览器分配量，也不是 GPU 实测显存或完整 HBM 流量。',
  '朴素路径的 S/P 往返按 4T² 个元素计算，包含 S 写/读与 P 写/读；Mask 融合在分数阶段，省略独立 Mask、Dropout、Q/K/V/O 搬运与其他开销。',
  'FlashAttention 减少完整中间矩阵的保存与 HBM 往返；KV Cache 减少生成步骤之间的旧 K/V 重算，两者解决不同的问题。',
  '本页不模拟 CUDA kernel、反向重算、硬件流水线或运行时间，也不据此推算真实 GPU 的加速倍数。',
];
