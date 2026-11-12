import { T } from '../i18n';
import type { TopicId } from './topics';
import type { FormulaSymbol } from './notation/shared';
export const ARCHITECTURE_FAMILIES: {
  id: string;
  name: string;
  headline: string;
  mechanism: string;
  history: string;
  boundary: string;
  source: string;
  topic: TopicId | null;
}[] = [
  {
    id: 'attention',
    name: 'Transformer',
    headline: T('逐位置读取可见历史'),
    mechanism: T('每个 Query 与可见 Key 打分，按权重汇总 Value；因果 Mask 保留方向依赖。'),
    history: T(
      'Decode 保留每层历史 K/V，缓存随 token 数增长。FlashAttention 避免完整分数中间矩阵，并不消除 KV 缓存。',
    ),
    boundary: T(
      '训练可并行算各位置；Decode 的新 Query 仍读取历史。表达能力和实际性能要在具体配置下衡量。',
    ),
    source: 'https://arxiv.org/abs/1706.03762',
    topic: 'attention',
  },
  {
    id: 's4',
    name: 'S4 / LTI SSM',
    headline: T('固定规则的状态空间模型'),
    mechanism: T(
      '在同一序列内使用固定状态转移与读写规则；递推可以展开为卷积。固定规则仍可在训练中学习。',
    ),
    history: T('递归推理维护固定大小状态；卷积训练的工作区与激活需要另算。'),
    boundary: T(
      '固定参数时可利用卷积组织并行训练。把参数改成输入相关后，一般不能沿用同一个全局卷积核。',
    ),
    source: 'https://arxiv.org/abs/2111.00396',
    topic: 'ssm',
  },
  {
    id: 'mamba',
    name: 'Mamba-1',
    headline: T('选择性状态更新'),
    mechanism: T('Δ、B、C 随当前输入变化；短卷积、选择性 SSM、门控和输出投影一起组成 Block。'),
    history: T('逐步生成保留 SSM 状态和短卷积历史；大小由通道、状态维和卷积宽度决定。'),
    boundary: T(
      '有递推依赖不等于只能串行训练：仿射状态更新可用并行 scan 组合。固定大小状态也不保证无损记住任意历史。',
    ),
    source: 'https://arxiv.org/html/2312.00752v2',
    topic: 'ssm',
  },
  {
    id: 'ssd',
    name: 'Mamba-2 / SSD',
    headline: T('状态空间与结构化矩阵的联系'),
    mechanism: T(
      'SSD 对状态转移增加结构约束，用递推形式与结构化矩阵形式描述相关计算，并组织高效的分块算法。',
    ),
    history: T('保留跨块状态；块内可使用矩阵乘法。数学上写出完整矩阵不要求实现中保存它。'),
    boundary: T(
      '这种联系针对受约束的 SSM 与结构化 Attention；不能推成任意 SSM 都等于标准 Softmax Attention。',
    ),
    source: 'https://arxiv.org/abs/2405.21060',
    topic: 'memory',
  },
  {
    id: 'rwkv',
    name: 'RWKV-4 (2023)',
    headline: T('衰减的 Key/Value 聚合与门控'),
    mechanism: T(
      '用 time-mixing、带衰减的 Key/Value 聚合和 receptance 门控建模序列；这里讨论 2023 年论文版本。',
    ),
    history: T(
      '递归生成维护聚合状态，无需按原样保留所有历史 K/V。状态含义和大小随 RWKV 版本而变。',
    ),
    boundary: T(
      '训练与递归推理可以采用不同计算组织；它不等同于把普通 Softmax Attention 改写一下。',
    ),
    source: 'https://arxiv.org/abs/2305.13048',
    topic: null,
  },
  {
    id: 'retnet',
    name: 'RetNet (2023)',
    headline: T('带衰减的 Retention'),
    mechanism: T('Retention 用多尺度衰减控制历史贡献，并保留读出所需的聚合矩阵状态。'),
    history: T('支持并行、递归、分块递归三种计算形式；生成状态不随已处理 token 数逐项追加。'),
    boundary: T(
      '应比较同一机制的不同计算形式，再讨论与 Softmax Attention 的表达差异；不能只比较箭头或大 O 标签。',
    ),
    source: 'https://arxiv.org/abs/2307.08621',
    topic: null,
  },
  {
    id: 'hyena',
    name: 'Hyena (2023)',
    headline: T('长卷积与数据相关门控'),
    mechanism: T('将隐式参数化的长卷积与数据相关的逐元素门控交替组合，作为序列混合算子。'),
    history: T('长卷积可利用 FFT 等算法；不能仅凭“卷积”就假定拥有与标量 SSM 相同的固定递归缓存。'),
    boundary: T('需说明过滤器参数化、门控及具体实现；本页提供机制定位，没有运行完整 Hyena 模型。'),
    source: 'https://arxiv.org/abs/2302.10866',
    topic: null,
  },
  {
    id: 'hybrid',
    name: 'Jamba / Hybrid',
    headline: T('混合序列层，独立配置专家'),
    mechanism: T(
      'Attention 层与 Mamba 层交错；某些层的 MLP 再替换为 MoE。混合比例和专家配置是两个不同设计轴。',
    ),
    history: T('Attention 层仍需 KV 缓存，Mamba 层保留状态与卷积历史；两者一起计入历史存储。'),
    boundary: T(
      '原始 Jamba 发布配置的 Attention:Mamba 比为 1:7。保留完整 Attention 的混合模型仍有二次配对项，不能统称 O(T)。',
    ),
    source: 'https://arxiv.org/html/2403.19887v2',
    topic: 'moe',
  },
];
export const ARCHITECTURE_SYMBOLS = {
  norm: [
    'X,\\operatorname{LN}',
    T(
      'X 是逐 token 归一化后的分支输入；本图选择 LayerNorm，作者实现也可配置 RMSNorm。归一化在特征轴上进行。',
    ),
  ],
  input: [
    'X,u,z,W_{in}',
    T('X 是归一化后的输入 [T,D]；W_in 将其扩维并分成 u、z 两支，各为 [T,D_inner]。'),
  ],
  dims: [
    'T,D,D_{inner},N',
    T('T 是 token 数，D 是隐藏维，D_inner 是扩展通道数，N 是每通道状态维数。'),
  ],
  conv: [
    'v,\\operatorname{DWConv}_{causal},\\operatorname{SiLU}',
    T('DWConv 是逐通道短因果卷积，只读当前及之前位置；v 是卷积后经 SiLU 的主分支激活。'),
  ],
  select: [
    '\\Delta_t,B_t,C_t,v_t',
    T('由当前主分支输入 vₜ 生成步长 Δₜ、写入 Bₜ 和读出 Cₜ；本式的线性投影参数在前向中固定。'),
  ],
  scan: [
    'h_t,\\bar A_t,\\bar B_t,C_t,y_t',
    T(
      '本式用单通道行向量：hₜ 为 [1,N]，Āₜ 为 [N,N]，B̄ₜ 为 [1,N]，Cₜ 为 [N,1]，vₜ 和 yₜ 为标量。横线表示离散化后。',
    ),
  ],
  gate: [
    'g,y,z,\\odot',
    T('y 是 SSM 主分支输出，z 是投影时分出的门控支路；⊙ 表示同形状张量逐元素相乘，得到 g。'),
  ],
  output: ['F,g,W_{out}', T('W_out 将扩展通道压回 D 维；F 是送回残差主路的分支输出 [T,D]。')],
  residual: [
    'X_{res},X_{next},F',
    T('X_res 是归一化之前的残差输入；X_next 是原输入加分支 F 后的下一层表示。'),
  ],
  ssd: [
    'a,x_t,h_t,h_{-1}',
    T(
      'a 是每步保留比例，xₜ 是当前位置标量输入，hₜ 是更新后状态；本例初始 h₋₁=0，写入与读出系数均为 1。',
    ),
  ],
  matrix: [
    'L_{tj},t,j,x,y',
    T(
      'L 的行 t 是输出位置，列 j 是输入位置，未来项为 0；x、y 在这里都是 [1,4] 行向量，因此读出用 y=xLᵀ。',
    ),
  ],
  ledger: [
    'B,L_A,T,H_{KV},d_k,d_v,b',
    T('依次为批大小、Attention 层数、token 数、KV 头数、Key/Value 每头维数和每元素字节数。'),
  ],
  ssmCache: [
    'L_S,D_{inner},N,k',
    T(
      'L_S 是 SSM 层数；每通道 N 个状态，加 k−1 个因果短卷积历史。本账本只数历史状态，不含权重和其他工作区。',
    ),
  ],
} satisfies Record<string, FormulaSymbol>;
export const MAMBA_NODES = [
  {
    id: 'projection',
    title: T('输入投影与两条支路'),
    formula: '[u,z]=XW_{in}',
    explanation: T(
      '先把已经归一化的 X 投影到两倍扩展宽度，再沿特征维分成主分支 u 和门控支路 z。两个分支保留相同 token 顺序。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.input, ARCHITECTURE_SYMBOLS.dims],
  },
  {
    id: 'conv',
    title: T('短因果卷积与 SiLU'),
    formula: 'v=\\operatorname{SiLU}(\\operatorname{DWConv}_{causal}(u))',
    explanation: T(
      '短卷积在每个扩展通道内混合相邻位置，为主分支加入局部信息；生成时需要保留有限个过去的卷积输入。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.input, ARCHITECTURE_SYMBOLS.conv],
  },
  {
    id: 'selection',
    title: T('从内容生成选择性参数'),
    formula:
      '\\Delta_t=\\operatorname{softplus}(\\operatorname{Linear}_\\Delta(v_t)),\\quad B_t=\\operatorname{Linear}_B(v_t),\\quad C_t=\\operatorname{Linear}_C(v_t)',
    explanation: T(
      'Δ、B、C 是当前输入算出的激活，不是每读一个 token 就训练出新参数。连续状态参数 A 在这一序列的前向中固定，离散后的 Ā 仍会随 Δ 改变。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.select, ARCHITECTURE_SYMBOLS.conv],
  },
  {
    id: 'scan',
    title: T('选择性 SSM 扫描'),
    formula: 'h_t=h_{t-1}\\bar A_t+v_t\\bar B_t,\\qquad y_t=h_tC_t',
    explanation: T(
      '每个通道把过去压入 N 维状态。顺序形式便于逐 token 生成；训练可以利用仿射更新的结合律做并行 scan，不能把逐步依赖直接等同于必须串行训练。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.scan, ARCHITECTURE_SYMBOLS.dims],
  },
  {
    id: 'gate',
    title: T('门控汇合'),
    formula: 'g=y\\odot\\operatorname{SiLU}(z)',
    explanation: T(
      'SSM 输出与另一支输入投影的 SiLU 结果逐元素相乘。这里的门作用于特征，不是在多个专家之间做 Top-K 路由。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.gate],
  },
  {
    id: 'output',
    title: T('输出投影'),
    formula: 'F=gW_{out}',
    explanation: T(
      '把扩展通道压回隐藏维数 D，得到可以与原输入相加的分支。这个投影与 SSM 内部的读出 C 是不同操作。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.output, ARCHITECTURE_SYMBOLS.dims],
  },
  {
    id: 'residual',
    title: T('加回残差主路'),
    formula: 'X=\\operatorname{LN}(X_{res}),\\qquad X_{next}=X_{res}+F',
    explanation: T(
      '本图采用预归一化的残差包装：分支读取归一化后的输入，最后加回未经这次归一化替换的原残差输入。图解按 Mamba-1 结构，不代表所有 Mamba 版本。',
    ),
    symbols: [ARCHITECTURE_SYMBOLS.residual, ARCHITECTURE_SYMBOLS.norm],
  },
];
export const ARCHITECTURE_VERIFIED_DATE = '2026-11-12';
