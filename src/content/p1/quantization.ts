import { quantizationSymbols } from '../notation/quantization';
import { formatNumber as fmt } from '../../math/attention';
import { quantization } from '../../math/p1/quantization';
import { BOOK } from '../book';
import { choice, field, lesson, type TopicDefinition } from './types';
import { T } from '../../i18n';

const topic: TopicDefinition = {
  id: 'quantization',
  path: '/quantization',
  title: T('量化：少存几个比特，会丢掉什么？'),
  chapter: T('第 27 章'),
  intro: T(
    '你手里有八个数，要把它们存成很少的比特。先对比对称与非对称（含零点）刻度，再看舍入、裁剪和分组；最后用同一行权重核对均匀 RTN、AWQ 激活缩放和 GPTQ 逐列误差补偿怎样降低输出误差。',
  ),
  question: T('这一页想弄明白：连续的小数落到有限格点上会丢掉什么，怎样把误差降下来'),
  headline: T('定刻度、舍入裁剪、分组，再对照非对称、AWQ 与 GPTQ 的误差控制。'),
  why: {
    question: T('为什么要把权重存成更少的比特？'),
    points: [
      T(
        '算一笔账（原书 27.1.1）：LLaMA-7B 用 FP16 存储约 14GB，INT8 约 7GB，INT4 约 3.5GB。位数减少，模型就能放进更小的显卡。',
      ),
      T(
        '代价是精度：INT4 只有 16 个可选整数。对称量化对偏态分布会浪费一半刻度，非对称量化引入零点 z 提高利用率；而 AWQ 保护大激活对应的权重通道，GPTQ 利用二阶信息把当前列的量化误差补偿到后续列。',
      ),
    ],
    book: [T('第 27.1.1 节 · 大模型的内存困境'), BOOK.ch27],
  },
  input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6],
  inputNames: ['0', '1', '2', '3', '4', '5', '6', '7'],
  fields: [
    choice('bits', T('整数位数'), 4, [
      { value: 2, label: '2bit' },
      { value: 4, label: '4bit' },
      { value: 8, label: '8bit' },
    ]),
    choice('scheme', T('量化方案'), 0, [
      { value: 0, label: T('对称（零点 0）') },
      { value: 1, label: T('非对称（含零点 z）') },
    ]),
    choice('group', T('组大小'), 8, [
      { value: 2, label: '2' },
      { value: 4, label: '4' },
      { value: 8, label: T('全向量8') },
    ]),
    field('cap', T('校准绝对上限'), 8, 0.5, 8, 0.5),
    choice(
      'method',
      T('量化方法'),
      0,
      [
        { value: 0, label: T('均匀 RTN') },
        { value: 1, label: T('AWQ 激活缩放') },
        { value: 2, label: T('GPTQ 逐列补偿') },
      ],
      [4, 5],
    ),
  ],
  steps: [
    lesson({
      name: T('离散刻度'),
      title: T('先定格点：位数、对称性与最大值决定间距'),
      symbols: quantizationSymbols[0],
      formula: String.raw`\mathrm{sym}:\ s_g=\frac{\min(cap,\max|x_g|)}{2^{b-1}-1},\qquad \mathrm{asym}:\ s_g=\frac{x_{\max}-x_{\min}}{2^b-1},\ z_g=\operatorname{round}\!\left(\frac{-x_{\min}}{s_g}\right)`,
      explanation: (c) => {
        const qmin = c.metric(T('整数范围下界'));
        const qmax = c.metric(T('整数范围 ±qmax'));
        const zp = c.metric(T('第 0 组零点 z'));
        const scales = c.values(T('每组 scale'));
        const asym = c.config.scheme === 1;
        return T(
          '{#0} bit {#1}整数范围 [{#2}, {#3}]。当前第 0 组零点 z = {#4}，{#5}。遇到全正或偏态分布时，切换到非对称量化可用满全部正数编码。',
          c.config.bits,
          asym ? T('非对称') : T('对称'),
          qmin,
          qmax,
          zp,
          scales.length === 1
            ? T('全向量一组，scale = {#0}，左图短竖线就是全部能表示的值', fmt(scales[0], 4))
            : T('分成 {#0} 组，各组 scale 见表', scales.length),
        );
      },
      prompt: T('同样 4 bit 下，面对全正分布，为什么非对称量化的格点比对称量化密一倍？'),
      takeaway: T('位数决定格点总数，对称/非对称与校准范围决定每个格点落到哪里。'),
      book: [T('第 27.2.2 节 · 线性量化'), BOOK.ch27],
    }),
    lesson({
      name: T('舍入'),
      title: T('每个数落到最近的格点，再反量化回来'),
      symbols: quantizationSymbols[1],
      formula: String.raw`q_i=\operatorname{clip}(\operatorname{round}(x_i/s_g)+z_g,\,q_{\min},\,q_{\max}),\qquad \hat x_i=(q_i-z_g)\,s_g`,
      explanation: (c) => {
        const x = c.values(T('裁剪后的 x'))[0];
        const q = c.values(T('整数 q'))[0];
        const asym = c.config.scheme === 1;
        const restored = c.values(
          asym ? T('还原 x̂ = (q − z) × scale') : T('还原 x̂ = q × scale'),
        )[0];
        const s = c.metric(T('第 0 组 scale'));
        const zp = c.metric(T('第 0 组零点 z'));
        return T(
          '以第 0 个数为例：{#0} ÷ {#1} 再加零点 {#2}，取整得整数编码 {#3}，反量化回 {#4}，误差 {#5}。舍入丢掉的尾部信息在反量化时无法恢复。',
          fmt(x),
          fmt(s, 4),
          fmt(zp),
          fmt(q),
          fmt(restored, 4),
          fmt(restored - c.input[0], 4),
        );
      },
      prompt: T('还原值等于原值需要满足什么条件？非对称量化为什么要求 0 落在整数零点 z 上？'),
      takeaway: T('误差来自舍入和裁剪；反量化只是乘回 scale 并减去零点，丢掉的细节回不来。'),
      book: [T('第 27.2.1 节 · 什么是量化'), BOOK.ch27],
    }),
    lesson({
      name: T('裁剪'),
      title: T('把校准上限调小：格点更密，但大数被截断'),
      symbols: quantizationSymbols[2],
      formula: String.raw`x_{clip}=\max(-a_g,\min(a_g,x))`,
      explanation: (c) =>
        T(
          '当前校准上限 cap = {#0}。把它拖小，格点间距随之变小，普通的小数值落点更准；但超过上限的数会被压到边界上（左图虚线），这一项的误差反而变大。',
          fmt(c.metric(T('校准上限 cap'))),
        ),
      prompt: T('为什么更密的刻度仍可能让总误差变大？'),
      takeaway: T('校准是在分辨率和截断误差之间取舍。'),
      book: [T('第 27.2.3 节 · 对称与非对称量化'), BOOK.ch27],
    }),
    lesson({
      name: T('分组'),
      title: T('分组：每组用自己的格点间距'),
      symbols: quantizationSymbols[3],
      formula: String.raw`B_{scale}=4\lceil N/G\rceil`,
      explanation: (c) =>
        T(
          '当前组大小 {#0}，共 {#1} 组，每组各有一个 scale（非对称时还各存一个零点 z）。异常值只撑大它所在那一组的间距，其他组的小数值能落得更准，代价是元数据增多。',
          fmt(c.metric(T('组大小'))),
          fmt(c.metric(T('组数'))),
        ),
      prompt: T('组越小一定越好吗？文件会按位数的比例变小吗？'),
      takeaway: T('分组提高精度，但要把每组的 scale 和零点元数据一起算进账本。'),
      book: [T('第 27.2.4 节 · 量化粒度'), BOOK.ch27],
    }),
    lesson({
      name: T('AWQ 与 GPTQ'),
      title: T('保护关键通道，或把误差补偿给后面的列'),
      symbols: quantizationSymbols[4],
      formula: String.raw`\mathrm{AWQ}:\ \hat w_i=\frac{Q(w_i s_i)}{s_i},\qquad \mathrm{GPTQ}:\ w_k\leftarrow w_k-(w_j-\hat w_j)\frac{[H^{-1}]_{j,k}}{[H^{-1}]_{j,j}}`,
      explanation: (c) => {
        const mse = c.metric(T('均方误差 MSE'));
        const rtnErr = c.metric(T('RTN 输出误差'));
        const awqErr = c.metric(T('AWQ 输出误差'));
        const gptqLoss = c.metric(T('GPTQ 二次损失 ΔᵀHΔ'));
        return T(
          '当前方法 MSE = {#0}。固定激活 u 在通道 6 幅值最大（|u₆|=2）：均匀 RTN 的输出绝对误差为 {#1}，AWQ 按激活放大重要权重再量化后输出误差为 {#2}；GPTQ 则利用校准激活构造的 Hessian 逆矩阵，在量化第 j 列后同步微调后续列，使校准二次损失 ΔᵀHΔ 降到 {#3}。',
          fmt(mse, 5),
          fmt(rtnErr, 4),
          fmt(awqErr, 4),
          fmt(gptqLoss, 4),
        );
      },
      prompt: T('为什么 AWQ 在量化前放大高激活通道的权重，最终线性输出却不需要改写网络结构？'),
      takeaway: T('AWQ 用等价通道缩放保护大激活对应权重；GPTQ 用二阶曲率把当前列误差补偿到未量化列。'),
      book: [T('第 27.3–27.4 节 · GPTQ 与 AWQ 原理'), BOOK.ch27],
    }),
    lesson({
      name: T('存储与误差'),
      title: T('账本：省下多少字节，换来多少输出误差'),
      symbols: quantizationSymbols[5],
      formula: String.raw`B=\lceil Nb/8\rceil+(4+b_z)\lceil N/G\rceil`,
      explanation: (c) =>
        T(
          '当前合计 {#0} 字节（整数 {#1} + scale {#2} + 零点 {#3}），float32 基线 {#4} 字节。对比校准二次损失 ΔᵀHΔ：RTN 为 {#5}，GPTQ 为 {#6}。账本不含文件头与对齐；GGUF 是打包张量与多种量化类型的文件格式，不是单一算法。',
          fmt(c.metric(T('合计 B'))),
          fmt(c.metric(T('打包整数载荷 B'))),
          fmt(c.metric(T('float32 尺度 B'))),
          fmt(c.metric(T('零点元数据 B'))),
          fmt(c.metric(T('float32 基线 B'))),
          fmt(c.metric(T('RTN 二次损失 ΔᵀHΔ')), 4),
          fmt(c.metric(T('GPTQ 二次损失 ΔᵀHΔ')), 4),
        ),
      prompt: T('为什么理论整数载荷不等于模型文件大小？GPTQ 和 AWQ 的校准成本分别用在了哪里？'),
      takeaway: T('账本要写明粒度、scale、零点、文件头和对齐；算法、数值类型与文件格式是三件事。'),
      book: [T('第 27.5 节 · GGUF 与量化选型'), BOOK.ch27],
    }),
  ],
  examples: [
    { name: T('含异常值'), input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6] },
    {
      name: T('偏态正值（非对称）'),
      input: [0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 4.0],
      parameters: { scheme: 1 },
    },
    {
      name: T('AWQ 保护通道'),
      input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6],
      parameters: { method: 1 },
    },
    {
      name: T('GPTQ 误差补偿'),
      input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6],
      parameters: { method: 2, group: 4 },
    },
    { name: T('全零边界'), input: Array(8).fill(0) },
    { name: T('均匀小值'), input: [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1] },
  ],
  compute: quantization,
  prediction: {
    key: 'bits',
    target: 8,
    question: T('只提高整数位数，当前 MSE 会保持不变吗？'),
    label: T('还原均方误差'),
  },
  sources: [
    { title: T('原书第 27 章 · 模型量化'), url: BOOK.ch27 },
    {
      title: T('PyTorch fake quantize 定义'),
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.fake_quantize_per_tensor_affine.html',
    },
    { title: 'GPTQ · arXiv 2210.17323', url: 'https://arxiv.org/abs/2210.17323' },
    { title: 'AWQ · arXiv 2306.00978', url: 'https://arxiv.org/abs/2306.00978' },
    { title: T('GGUF 官方格式'), url: 'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md' },
  ],
  limits: [
    T('对称有符号编码有一个补码值未使用；非对称编码把 0 包含在区间内以确保实数 0 精确落到整数零点 z。'),
    T('AWQ 演示采用固定指数 α = 0.5 的解析通道缩放，未运行论文中的网格搜索；GPTQ 演示在 8 维单行权重与 8 条固定校准激活上计算精确逆 Hessian 与 Schur 补消元，不是百亿参数分块实现。'),
    T('全零组约定 scale = 1、q = 0、z = 0，避免除以 0。'),
    T('理论字节数假设整数紧密打包、每组 4B scale（非对称加 1B zero-point），不含对齐、文件头或其他元数据；不模拟 GPU kernel 或真实模型质量。'),
  ],
  bookNotes: [
    T(
      '原书 27.2.1 的 INT4 示例写成 −0.5→−8、0.25→2、0.5→4、0.75→6、1.0→7。由后几项可知间距是 0.125，那么 −0.5 应映射到 −4，不是 −8（1.0 映射到 8 后被裁剪为 7）。',
    ),
    T(
      '原书 27.2.2 用 scale = 2/255 把 [−1, 1] 映射到 INT8：0.5 → 64 → 0.50196（书中用四舍五入后的 scale 写成 0.50176）。本页的对称量化取 scale = 1/127：同一个 0.5 也落在 64，还原为 0.50394。两种约定都成立，比较时要说明用的是哪一种。',
    ),
  ],
};
export default topic;
