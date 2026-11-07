import { BOOK_URL } from './topics';

const CH22 = BOOK_URL + '第22章-KV-Cache-推理加速/';
const CH23 = BOOK_URL + '第23章-MHA到MQA到GQA演进/';
const CH30 = BOOK_URL + '第30章-Mixture-of-Experts-稀疏激活的秘密/';
const DEEPSEEK_V2 = 'https://arxiv.org/html/2405.04434v5';

export const kvSteps: {
  name: string;
  title: string;
  explanation: string;
  takeaway: string;
  prompt: string;
  formula: string;
  /** Label and full URL (a book chapter or a paper section). */
  source: [string, string];
}[] = [
  {
    name: '缓存账本',
    title: 'KV 缓存有多大：先把账算清楚',
    explanation:
      'Decode 时，每一层都要保留所有历史位置的 K 和 V。缓存的元素个数等于 批大小 × 层数 × token 数 × KV 头数 × $(d_k+d_v)$，再乘以每个元素的字节数。这个式子只统计 KV 缓存，不含模型权重、激活值和显存分配器的额外开销。架构改动能影响的，是 KV 头数和每个头存多少维。',
    takeaway:
      'KV 缓存与层数、token 数、KV 头数和每头维度成正比；改架构时能动的主要是 KV 头数和每头存的维度。',
    prompt: 'KV 缓存的公式里有哪几个乘数？把 KV 头从 32 减到 8，缓存变成原来的多少？',
    formula:
      '\\text{bytes}=B\\times L\\times T\\times H_{KV}\\times\\left(d_k+d_v\\right)\\times b',
    source: ['原书第 22.4 节 · 内存公式：关键是 n_kv_heads', CH22],
  },
  {
    name: '共享 KV 头',
    title: 'MHA、GQA、MQA：Query 头不变，K/V 头分组共享',
    explanation:
      'MHA 里每个 Query 头都有自己的一组 K、V。GQA 把 Query 头分成 $G$ 组，同组的头读同一组 K、V；$G=1$ 就是 MQA，$G=H$ 就是 MHA。Query 头的数量和打分次数都不变，变小的是 K/V 投影的参数和缓存。从 MHA 改成 GQA 时，论文把组内各头的 K/V 投影取平均，然后再继续训练一小段；本页只做了取平均。',
    takeaway: '减少的是 KV 头，不是 Query 头：缓存和 K/V 投影变小，$W_Q$、$W_O$ 和打分次数不变。',
    prompt: 'KV 头从 4 减到 1，哪些量变小，哪些不变？同组的两个 Query 头，注意力权重一定相同吗？',
    formula:
      '\\text{head}_h=\\operatorname{softmax}\\!\\left(\\frac{Q_hK_{g(h)}^{\\top}}{\\sqrt{d}}+M\\right)V_{g(h)},\\qquad g(h)=\\left\\lfloor\\frac{h}{H/G}\\right\\rfloor',
    source: ['原书第 23.4 节 · 一个实现覆盖 MHA、GQA 和 MQA', CH23],
  },
  {
    name: 'MLA 压缩',
    title: 'MLA：只缓存一个低维 latent，用时再还原各头的 K、V',
    explanation:
      'MLA 不减少头数，而是把每个 token 的 K 和 V 一起压进一个低维向量 $c_t=h_tW_{DKV}$，缓存里只存它。需要时，每个头用自己的上投影还原：$k^C_{t,i}=c_tW_{UK}^{(i)}$，$v^C_{t,i}=c_tW_{UV}^{(i)}$。所以每个头仍有各自不同的 K、V，只是都由同一个 $c_t$ 算出。本例 4 个头、每头 2 维：MHA 每个 token 每层要存 16 个数，MLA 存 $d_c=3$ 个，加上下一步要讲的 RoPE Key，一共 5 个。',
    takeaway: 'MLA 缓存联合压缩后的 latent，每个头的 K、V 由它还原；头数不变。',
    prompt:
      'MLA 的缓存里存的是什么？每个头的 K、V 从哪里来？它和 MQA 的“所有头共享一组 K、V”有什么不同？',
    formula:
      'c_t=h_tW_{DKV},\\qquad k^{C}_{t,i}=c_tW_{UK}^{(i)},\\qquad v^{C}_{t,i}=c_tW_{UV}^{(i)}',
    source: ['DeepSeek-V2 论文 §2.1.2 · Low-Rank Key-Value Joint Compression', DEEPSEEK_V2],
  },
  {
    name: '解耦 RoPE',
    title: '为什么 RoPE 要单独走一支？',
    explanation:
      '推理时希望不还原 K，直接拿 Query 和缓存里的 $c_j$ 打分：$q\\,(c_jW_{UK})^{\\top}=(q\\,W_{UK}^{\\top})\\,c_j^{\\top}$，也就是把 $W_{UK}$ 并进 Query 一侧。如果 RoPE 直接加在还原出的 K 上，夹在 $q$ 和 $c_j$ 之间的就变成 $R_tR_j^{\\top}W_{UK}^{\\top}$，它随距离 $t-j$ 变化，没有一个固定矩阵能吸收它。DeepSeek-V2 把位置信息放进另一支：每个头有一个带 RoPE 的 $q^R$，所有头共享一个带 RoPE 的 $k^R_t$，这个 $k^R_t$ 也要缓存。',
    takeaway: 'RoPE 夹在中间会破坏矩阵吸收，所以位置信息单独走一支；共享的 RoPE Key 也要进缓存。',
    prompt:
      '如果把 RoPE 直接加在由 latent 还原出的 K 上，同一个吸收矩阵在哪些位置还能算对？解耦之后缓存里多了什么？',
    formula:
      'q_{t,i}=\\left[q^{C}_{t,i};\\,q^{R}_{t,i}\\right],\\quad k_{j,i}=\\left[k^{C}_{j,i};\\,k^{R}_{j}\\right],\\quad k^{R}_j=\\operatorname{RoPE}_j\\!\\left(h_jW_{KR}\\right)',
    source: ['DeepSeek-V2 论文 §2.1.3 · Decoupled Rotary Position Embedding', DEEPSEEK_V2],
  },
  {
    name: '矩阵吸收',
    title: '矩阵吸收：推理时不必还原 K 和 V',
    explanation:
      '利用矩阵乘法的结合律，把还原 K 的 $W_{UK}^{(i)}$ 并进 Query 投影，得到 $\\tilde W_Q^{(i)}=W_{QC}^{(i)}{W_{UK}^{(i)}}^{\\top}$；把还原 V 的 $W_{UV}^{(i)}$ 并进输出投影，得到 $\\tilde W_O^{(i)}=W_{UV}^{(i)}W_O^{(i)}$。这样打分直接用缓存里的 $c_j$ 和 $k^R_j$，加权求和也停在 latent 空间。结果与先还原 K、V 的展开路径一致，只是计算顺序不同。',
    takeaway: '吸收不改变结果，改变的是计算顺序：推理时只读缓存里的 latent 与 RoPE Key。',
    prompt: '吸收之后，内容 Query 的维度变成多少？为什么 RoPE 那一支不能一起吸收？',
    formula:
      'q^{C}_{t,i}\\,{k^{C}_{j,i}}^{\\top}=\\Big(h_tW_{QC}^{(i)}{W_{UK}^{(i)}}^{\\top}\\Big)c_j^{\\top},\\qquad \\Big(\\sum_j p_j\\,c_jW_{UV}^{(i)}\\Big)W_O^{(i)}=\\Big(\\sum_j p_j\\,c_j\\Big)W_{UV}^{(i)}W_O^{(i)}',
    source: ['DeepSeek-V2 论文 §2.1.2（推理时的矩阵吸收）', DEEPSEEK_V2],
  },
  {
    name: '对照',
    title: '对照：每种方案省了什么，付出了什么',
    explanation:
      'MHA、GQA、MQA 通过减少 KV 头来缩小缓存；MLA 保留每个头各自的 K、V，改为缓存一个低维 latent 和一个共享的 RoPE Key。按 DeepSeek-V2 的配置，MLA 每个 token 每层缓存 $d_c+d_h^R=512+64=576$ 个数，相当于只有 2.25 组的 GQA；只算 latent 会少算这 64 个。缓存小不等于各方面都更好：分组减少了 K、V 的表达自由度，MLA 要做额外的上投影或吸收后的更大 Query。哪种更合适，取决于模型、训练与部署条件。',
    takeaway: '这些方案在缓存、参数、计算和效果之间各有取舍，不是一条从差到好的排行榜。',
    prompt:
      'MLA 的缓存为什么不能只算 latent？同样的头数和头维，MHA、GQA-8、MQA、MLA 每个 token 各存多少？',
    formula:
      '\\text{每 token 每层：}\\ \\text{MHA}\\ 2Hd_h,\\quad\\text{GQA}\\ 2Gd_h,\\quad\\text{MQA}\\ 2d_h,\\quad\\text{MLA}\\ d_c+d_h^R',
    source: ['原书第 23.6 节 · 真实模型怎样选择', CH23],
  },
];

export const kvSources = [
  { title: '原书第 22 章 · KV Cache', url: CH22 },
  { title: '原书第 23 章 · MHA 到 MQA 到 GQA', url: CH23 },
  { title: '原书第 30 章 · Mixture of Experts（30.4.3 节介绍 MLA）', url: CH30 },
  {
    title: 'Fast Transformer Decoding: One Write-Head is All You Need · Shazeer, 2019（MQA）',
    url: 'https://arxiv.org/abs/1911.02150',
  },
  {
    title: 'GQA: Training Generalized Multi-Query Transformer Models · Ainslie et al., 2023',
    url: 'https://arxiv.org/abs/2305.13245',
  },
  { title: 'DeepSeek-V2 · §2.1 Multi-Head Latent Attention 与表 1', url: DEEPSEEK_V2 },
];

/** Common misconceptions, each answered by something the page computes or quotes. */
export const kvMyths = [
  '减少 KV 头不会减少 Query 头，也不会让整个 Attention 或整个模型按同样比例变小：$W_Q$、$W_O$ 和打分次数都不变。',
  'KV 缓存公式只统计 K、V，不含模型权重、激活值和显存分配器的开销。',
  'MLA 的缓存不只是 latent：解耦 RoPE 的共享 Key 也要缓存。',
  'MLA 不是“所有头共享一组 K、V”：每个头仍有自己的 K、V，只是由同一个 latent 还原。',
  '原书第 30.4.3 节“KV Cache 减少 75%”是假设 latent 维度为原来 1/4 时的示意计算，不是 DeepSeek 的实测。',
  'GQA 从 MHA 转换时，论文在组内取平均之后还要继续训练（约 5% 的原预训练算力）；本页只取平均，不代表转换后的效果。',
  'MHA、GQA、MQA、MLA 不是一条排行榜：缓存、参数、计算与效果之间各有取舍。',
];
