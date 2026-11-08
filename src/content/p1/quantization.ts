import { formatNumber as fmt } from '../../math/attention';
import { quantization } from '../../math/p1/quantization';
import { BOOK } from '../book';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'quantization',
  path: '/quantization',
  title: '量化：少存几个比特，会丢掉什么？',
  chapter: '第 27 章',
  intro:
    '编辑八个数，查看格点、舍入、裁剪、分组和逐项误差。把异常值留在同一组或拆开，比较误差和存储账本。',
  question: '同一个问题：连续的小数只能落到有限个格点上，丢掉的是什么',
  headline: '定格点 → 舍入 → 裁剪 → 分组 → 逐项误差 → 存储账本。',
  why: {
    question: '为什么要把权重存成更少的比特？',
    points: [
      '算一笔账（原书 27.1.1）：LLaMA-7B 用 FP16 存储约 14GB，INT8 约 7GB，INT4 约 3.5GB。位数减少，模型就能放进更小的显卡。',
      '代价是精度：INT4 只有 16 个可选整数，连续的小数只能落到最近的格点上，多出来的部分就丢了。本页用 8 个数手算这个过程：先定格点间距，再舍入、裁剪、分组，最后核对误差和存储账本。',
    ],
    book: ['第 27.1.1 节 · 大模型的内存困境', BOOK.ch27],
  },
  input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6],
  inputNames: ['0', '1', '2', '3', '4', '5', '6', '7'],
  fields: [
    choice('bits', '整数位数', 4, [
      { value: 2, label: '2bit' },
      { value: 4, label: '4bit' },
      { value: 8, label: '8bit' },
    ]),
    choice('group', '组大小', 8, [
      { value: 2, label: '2' },
      { value: 4, label: '4' },
      { value: 8, label: '全向量8' },
    ]),
    field('cap', '校准绝对上限', 8, 0.5, 8, 0.5),
  ],
  steps: [
    lesson({
      name: '离散刻度',
      title: '先定格点：位数和最大值决定间距',
      formula: String.raw`q_{max}=2^{b-1}-1,\quad s_g=\min(cap,\max|x_g|)/q_{max}`,
      explanation: (c) => {
        const qmax = c.metric('整数范围 ±qmax'),
          scales = c.values('每组 scale');
        return `${c.config.bits} bit 有符号整数这里取 −${qmax} 到 ${qmax}。格点间距 scale = 这一组的最大绝对值 ÷ ${qmax}；当前 ${scales.length === 1 ? `全向量一组，scale = ${fmt(scales[0], 4)}，左图的短竖线就是全部能表示的值` : `分成 ${scales.length} 组，各组 scale 见表`}。一个异常大的数会把间距撑大，小数值就只能挤在 0 附近的几个格点上。`;
      },
      prompt: '同样位数下，一个大异常值怎样改变格点间距？',
      takeaway: '位数决定格点个数，校准范围决定格点间距。',
      book: ['第 27.2.2 节 · 线性量化', BOOK.ch27],
    }),
    lesson({
      name: '舍入',
      title: '每个数落到最近的格点，再乘回间距',
      formula: String.raw`q_i=\operatorname{clip}(round(x_i/s_g),-q_{max},q_{max}),\quad\hat x_i=s_gq_i`,
      explanation: (c) => {
        const x = c.values('裁剪后的 x')[0],
          q = c.values('整数 q')[0],
          restored = c.values('还原 x̂ = q × scale')[0],
          s = c.metric('第 0 组 scale');
        return `以第 0 个数为例：${fmt(x)} ÷ ${fmt(s, 4)}，取最近的整数得 ${fmt(q)}，再乘回 scale 得到 ${fmt(restored, 4)}，误差 ${fmt(restored - c.input[0], 4)}。正好落在两个格点中间时，本页向远离 0 的方向取整；有的库用“四舍六入五成双”，结果可能差一格。`;
      },
      prompt: '还原值等于原值需要满足什么条件？',
      takeaway: '误差来自舍入和裁剪；还原只是乘回 scale，丢掉的细节回不来。',
      book: ['第 27.2.1 节 · 什么是量化', BOOK.ch27],
    }),
    lesson({
      name: '裁剪',
      title: '把校准上限调小：格点更密，但大数被截断',
      formula: String.raw`x_{clip}=\max(-a_g,\min(a_g,x))`,
      explanation: (c) =>
        `当前校准上限 cap = ${fmt(c.metric('校准上限 cap'))}。把它拖小，格点间距随之变小，普通的小数值落点更准；但超过上限的数会被压到边界上（左图虚线），这一项的误差反而变大。`,
      prompt: '为什么更密的刻度仍可能让总误差变大？',
      takeaway: '校准是在分辨率和截断误差之间取舍。',
      book: ['第 27.2.3 节 · 对称量化（零点固定为 0）', BOOK.ch27],
    }),
    lesson({
      name: '分组',
      title: '分组：每组用自己的格点间距',
      formula: String.raw`B_{scale}=4\lceil N/G\rceil`,
      explanation: (c) =>
        `当前组大小 ${fmt(c.metric('组大小'))}，共 ${fmt(c.metric('组数'))} 组，每组各有一个 scale。异常值只撑大它所在那一组的间距，其他组的小数值能落得更准。代价是每组多存一个 scale（本例 4 字节）；组越小，额外的元数据越多。`,
      prompt: '组越小一定越好吗？文件会按位数的比例变小吗？',
      takeaway: '分组提高精度，但要把元数据一起算进账本。',
      book: ['第 27.2.4 节 · 量化粒度', BOOK.ch27],
    }),
    lesson({
      name: '异常值',
      title: '误差从哪里来：逐项看',
      formula: String.raw`MSE=\frac1N\sum_i(\hat x_i-x_i)^2`,
      explanation: (c) =>
        `均方误差 MSE = ${fmt(c.metric('均方误差 MSE'), 5)}。对照表中的逐项误差，看是哪几个数贡献了大部分误差，通常是和异常值同组的小数值。原书 27.4 节的 AWQ 正是从“少数权重特别重要”出发去保护它们；本页只做均匀对称量化。`,
      prompt: '异常值在最后一组时，哪些 scale 会被它影响？',
      takeaway: '看逐项误差，才能解释总 MSE 从哪里来。',
      book: ['第 27.4.1 节 · AWQ 的核心洞察', BOOK.ch27],
    }),
    lesson({
      name: '存储与误差',
      title: '账本：省下多少字节，换来多少误差',
      formula: String.raw`B=\lceil Nb/8\rceil+4\lceil N/G\rceil`,
      explanation: (c) =>
        `当前合计 ${fmt(c.metric('合计 B'))} 字节（整数 ${fmt(c.metric('打包整数载荷 B'))} + scale ${fmt(c.metric('float32 尺度 B'))}），float32 基线 ${fmt(c.metric('float32 基线 B'))} 字节。先预测：只把位数提高，MSE 会不会变？账本只算整数和 scale，不含文件头与对齐；GGUF 是一种文件格式，不是一种量化算法。`,
      prompt: '为什么理论整数载荷不等于模型文件大小？',
      takeaway: '账本要写明粒度、scale、文件头和对齐是否计入。',
      book: ['第 27.2.5 节 · 常见量化位数', BOOK.ch27],
    }),
  ],
  examples: [
    { name: '含异常值', input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6] },
    { name: '全零边界', input: Array(8).fill(0) },
    { name: '均匀小值', input: [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1] },
  ],
  compute: quantization,
  prediction: {
    key: 'bits',
    target: 8,
    question: '只提高整数位数，当前 MSE 会保持不变吗？',
    label: '还原均方误差',
  },
  sources: [
    { title: '原书第 27 章 · 模型量化', url: BOOK.ch27 },
    {
      title: 'PyTorch fake quantize 定义',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.fake_quantize_per_tensor_affine.html',
    },
    { title: 'GGUF 官方格式', url: 'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md' },
  ],
  limits: [
    '对称有符号编码有一个补码值未使用；不是所有 INT4 或 GGUF 量化的规则。',
    '全零组约定 scale = 1、q = 0，避免除以 0。',
    '理论字节数假设整数紧密打包，不含对齐、文件头或其他元数据；浏览器数组没有实际压缩。',
    '不模拟 GPU kernel、推理速度或量化后的真实任务质量。',
  ],
  bookNotes: [
    '原书 27.2.1 的 INT4 示例写成 −0.5→−8、0.25→2、0.5→4、0.75→6、1.0→7。由后几项可知间距是 0.125，那么 −0.5 应映射到 −4，不是 −8（1.0 映射到 8 后被裁剪为 7）。',
    '原书 27.2.2 用 scale = 2/255 把 [−1, 1] 映射到 INT8：0.5 → 64 → 0.50196（书中用四舍五入后的 scale 写成 0.50176）。本页的对称量化取 scale = 1/127：同一个 0.5 也落在 64，还原为 0.50394。两种约定都成立，比较时要说明用的是哪一种。',
  ],
} satisfies TopicDefinition;
