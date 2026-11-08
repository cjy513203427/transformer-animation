import { memory } from '../../math/p1/memory';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'memory',
  path: '/attention-variants',
  title: '历史访问：稀疏连接与压缩状态',
  chapter: '第 24 章',
  intro:
    '用同一份 Q/K/V 比较完整与窗口 Attention，再逐步核对线性核递推和 Infini 分段记忆。看清连接变少与历史压缩各自的代价。',
  input: [1, 0, 0, 1, 1, 1, -1, 0.5],
  fields: [
    field('window', '因果窗口 w', 2, 1, 4, 1, [1, 5]),
    field('gate', '记忆门 logit β', 0, -4, 4, 0.5, [4]),
    choice(
      'delta',
      'Infini 更新',
      0,
      [
        { value: 0, label: 'Linear' },
        { value: 1, label: 'Delta' },
      ],
      [4],
    ),
  ],
  steps: [
    lesson(
      '完整历史',
      String.raw`P=softmax(QK^T/\sqrt{d_k}+M),\quad O=PV`,
      '每一行 Query 读取所有可见历史。Q 为输入四个二维行，K 交换两维，V 取原输入；这是明确的教学激活构造。',
      '一行可以读取几个历史位置？',
      '先固定基线，才能判断改变了算法还是执行方式。',
    ),
    lesson(
      '稀疏窗口',
      String.raw`\max(0,t-w+1)\le j\le t`,
      '滑动窗口包含自身，仅保留最近 w 个位置；Softmax 沿保留的 Key 归一化。这里只用稠密 Mask 计算，不宣称实际节省 FLOPs。',
      '缩小窗口，分母和输出为什么都会变？',
      '稀疏连接限制可见集合，是否省计算取决于实现。',
    ),
    lesson(
      '线性核',
      String.raw`w_{tj}=\phi(q_t)\cdot\phi(k_j),\quad\phi(x)=ELU(x)+1`,
      '正核特征替换 Softmax 的指数打分。先显式计算核权重与归一化输出，得到后续递推的独立基线。',
      '为什么这一行不能拿来要求等于标准 Softmax 输出？',
      'Linear Attention 使用另一种核，不是普通 Softmax 的无损等价改写。',
    ),
    lesson(
      '状态递推',
      String.raw`S_t=S_{t-1}+\phi(k_t)^Tv_t,\ z_t=z_{t-1}+\phi(k_t),\ o_t=\frac{\phi(q_t)S_t}{\phi(q_t)z_t^T}`,
      '读取当前 token 后更新固定大小 S、z，再用 Query 取出归一化结果。逐行与显式核计算核对，而不是与 Softmax 强行判等。',
      '状态大小依赖 T 吗？为什么训练能组织成扫描？',
      '固定特征维度时，递推压缩历史，扫描工作随 T 线性增长。',
    ),
    lesson(
      'Infini 分段',
      String.raw`A=\sigma(\beta)A_{mem}+(1-\sigma(\beta))A_{local}`,
      '两 token 一段，先从上一段记忆读出，再与本段因果 Softmax 合并。Linear 写入 V；Delta 写入 V 与旧记忆预测的残差。整段更新使用同一个旧记忆。',
      '把记忆更新放在当前段读取之前，会泄漏什么信息？',
      '分段局部 Attention 与压缩记忆分工，不能把两者画成无限完整缓存。',
    ),
    lesson(
      '压缩边界',
      String.raw`M_{new}=M_{old}+\phi(K)^T(V-\widehat V_{old})`,
      '并列展示不同 V 历史却相同 S、z 的碰撞。先预测扩大窗口是否保持当前稀疏输出，再执行。',
      '两个不同历史压到同一状态后，还能恢复每个历史值吗？',
      '固定状态可以丢信息；长序列可计算不等于无损或可靠记忆。',
    ),
  ],
  examples: [
    { name: '四个位置', input: [1, 0, 0, 1, 1, 1, -1, 0.5] },
    { name: '重复内容', input: [1, 1, 1, 1, 1, 1, 1, 1] },
  ],
  compute: memory,
  prediction: {
    key: 'window',
    target: 4,
    question: '只扩大窗口，当前稀疏输出会保持不变吗？',
    label: '窗口 Attention 输出',
  },
  sources: [
    { title: 'Longformer §3', url: 'https://arxiv.org/html/2004.05150v2' },
    { title: 'Linear Attention Eq.7–12', url: 'https://arxiv.org/html/2006.16236v3' },
    { title: 'Infini-attention Eq.3–6', url: 'https://arxiv.org/html/2404.07143v1' },
  ],
  limits: [
    '窗口只是稀疏连接示例，没有实现完整 Longformer 或块稀疏 GPU kernel。',
    'Infini 的片段长度固定2；初始空记忆分母为0时返回零。分段更新不逐 token 读取新记忆。',
    '本页保留过程数据以便核对，不是节省内存的运行时；不测量 GPU 吞吐。',
  ],
} satisfies TopicDefinition;
