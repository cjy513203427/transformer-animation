import type { ForwardKind } from '../math/forward';
import type { TopicId } from './topics';
import { commonSymbols, type FormulaSymbol } from './notation/shared';
import { T } from '../i18n';
const terms = {
  lastProbability: [
    'p_{T-1,v}',
    T(
      '最后位置给候选 v 的词表概率；v 遍历 a–d。这里选的是输出 token，不是 Attention 中取用某个历史位置的比例。',
    ),
  ],
  ...commonSymbols,
  ids: [
    '\\mathrm{token\\_ids}_t,t',
    T('token_ids 是输入编号；t 是从 0 开始的位置。本例只用 a–d 四个符号，编号为 0–3。'),
  ],
  embedding: [
    'E,H_{\\mathrm{tok}}',
    T('E 是固定的词嵌入表 [4,4]；H_tok 为按各位置的 ID 取出的行，形状 [3,4]。'),
  ],
  position: [
    'H_0,H_{\\mathrm{tok}},PE',
    T(
      'H₀ 是进入 Block 1 的隐藏状态；H_tok 是内容嵌入，PE 是同形状的正弦位置编码，按特征逐项相加。',
    ),
  ],
  hidden: [
    'H_{l-1},H_l,l,L',
    T('Hₗ₋₁、Hₗ 是当前 Block 的输入、输出，均为 [3,4]；l 是从 1 开始的 Block 编号，L 为总层数。'),
  ],
  ln: [
    '\\operatorname{LN}',
    T('LayerNorm 在每个 token 的 4 个特征上独立计算；本例 γ=1、β=0、ε=10⁻⁵。'),
  ],
  u: [
    'U_l',
    T('当前 Block 的第一次归一化结果 [3,4]；只送入 Attention 分支，残差支路仍保留 Hₗ₋₁。'),
  ],
  weights: [
    'W_Q^{(l)},W_K^{(l)},W_V^{(l)}',
    T('当前 Block 的三组固定投影参数，各为 [4,4]；右上角 (l) 标记层，不是乘方。'),
  ],
  heads: [
    'h,Q_h,K_h,V_h,d_h',
    T('h∈{0,1} 是头编号；每头 Q/K/V 为 [3,2]，dₕ=2。头内保留全部 token，只分取特征列。'),
  ],
  headP: ['P_h', T('每个头的 [3,3] 注意力权重，行是 Query、列是 Key；Softmax 沿 Key 轴计算。')],
  a: [
    'A,\\operatorname{Concat},W_O',
    T('各头先汇总自己的 Value，再沿特征列拼接，乘输出投影 W_O 得到 Attention 分支 A，形状 [3,4]。'),
  ],
  residual: ['R_l', T('第一次残差相加后的隐藏状态 [3,4]：原 Block 输入加 Attention 分支输出。')],
  z: ['Z_l', T('第二次 LayerNorm 的结果 [3,4]，送入 FFN；这一页的 Z 不是 logits。')],
  expand: [
    'W_1^{(l)},G,\\operatorname{GELU}',
    T('W₁ 为 [4,6]，把特征扩到 6 维；G 是逐元素 GELU 后的 [3,6] 激活，本例使用 tanh 近似。'),
  ],
  contract: ['G,W_2^{(l)},F', T('G 为 [3,6]，W₂ 为 [6,4]，F 是压回 4 维的 FFN 分支输出 [3,4]。')],
  final: [
    'H_f,H_L',
    T('H_L 是最后一个 Block 的输出；H_f 是最终 LayerNorm 后送往词表输出头的隐藏状态。'),
  ],
  logits: [
    '\\mathrm{logits},E^\\top',
    T('logits 为 [3,4]，每行对 a–d 四个候选打分；输出投影复用嵌入表的转置 Eᵀ，体现权重绑定。'),
  ],
  probability: [
    'p_{t,v},v,u,m_t',
    T(
      '小写 p 是词表概率；v、u 遍历四个候选，t 是位置，mₜ 是该行最大 Logit。它与 Attention 的 P 使用不同的归一化轴。',
    ),
  ],
  selection: [
    'T,\\mathrm{next},\\arg\\max_v',
    T('T=3 是已有 token 数；只用最后一行 p[T−1]，选概率最大的候选作为 next。平局时取较小 ID。'),
  ],
} satisfies Record<string, FormulaSymbol>;
export interface ForwardContent {
  title: string;
  explanation: string;
  prompt: string;
  formula: string;
  symbols: FormulaSymbol[];
  topic: TopicId;
}
export const FORWARD_CONTENT: Record<ForwardKind, ForwardContent> = {
  tokens: {
    title: T('输入编号'),
    explanation: T(
      '先把输入看成离散编号。本例 a、b、c、d 只是四个无语义的词表符号，已人为分词；这一步没有执行真实 BPE。',
    ),
    prompt: T('为什么 ID=3 不意味着比 ID=1 更接近某种语义？'),
    formula: '\\mathrm{token\\_ids}_t\\in\\{0,1,2,3\\}',
    symbols: [terms.ids],
    topic: 'attention',
  },
  embedding: {
    title: T('Embedding 查表'),
    explanation: T(
      '每个 ID 取出 E 的一行，不把编号作为连续数值输入线性层。相同 ID 在不同位置取得相同内容向量。',
    ),
    prompt: T('重复同一个 token 时，查到的两行向量是否相同？'),
    formula: 'H_{\\mathrm{tok}}[t,:]=E[\\mathrm{token\\_ids}_t,:]',
    symbols: [terms.ids, terms.embedding],
    topic: 'position',
  },
  position: {
    title: T('加入位置'),
    explanation: T(
      '把正弦位置编码加到内容上，再进入第一个 Block。后续 Q、K、V 都从带位置的隐藏状态生成。',
    ),
    prompt: T('为什么同一个 token 放在不同位置，H₀ 可以不同？'),
    formula: 'H_0=H_{\\mathrm{tok}}+PE',
    symbols: [terms.position],
    topic: 'position',
  },
  'attention-norm': {
    title: T('Attention 前归一化'),
    explanation: T(
      '采用 Pre-Norm：先逐行归一化，再送入 Attention。残差主路保留原输入；归一化没有把三个 token 混在一起统计。',
    ),
    prompt: T('为什么残差支路不能直接用 U 替换原输入？'),
    formula: 'U_l=\\operatorname{LN}(H_{l-1})',
    symbols: [terms.hidden, terms.ln, terms.u],
    topic: 'normalization',
  },
  qkv: {
    title: T('生成 Q、K、V'),
    explanation: T(
      '同一归一化结果乘三组参数。Q/K 用于匹配，V 提供内容；参数由种子固定生成，播放本次前向不会训练它们。',
    ),
    prompt: T('Q/K/V 与 W_Q/W_K/W_V 哪些是激活，哪些是参数？'),
    formula: 'Q=U_lW_Q^{(l)},\\quad K=U_lW_K^{(l)},\\quad V=U_lW_V^{(l)}',
    symbols: [terms.u, terms.query, terms.key, terms.value, terms.weights],
    topic: 'attention',
  },
  attention: {
    title: T('每头汇总并输出投影'),
    explanation: T(
      '按特征列切成两个头；每头沿 Key 轴归一化并汇总 V。图中两张 P 的行列都是 token，A 已经过拼接和输出投影。',
    ),
    prompt: T('两张 P 的列，与最终词表概率的列有什么不同？'),
    formula:
      '\\begin{gathered}P_h=\\operatorname{softmax}(Q_hK_h^\\top/\\sqrt{d_h}+M)\\\\A=\\operatorname{Concat}(P_0V_0,P_1V_1)W_O\\end{gathered}',
    symbols: [terms.heads, terms.headP, terms.mask, terms.a, terms.transpose],
    topic: 'attention',
  },
  'attention-add': {
    title: T('第一次残差相加'),
    explanation: T(
      '把 Attention 分支加回当前 Block 的输入。两项形状同为 [3,4]；这里没有把特征拼长。',
    ),
    prompt: T('这一加法的两条路径分别从哪里来？'),
    formula: 'R_l=H_{l-1}+A',
    symbols: [terms.hidden, terms.a, terms.residual],
    topic: 'ffn',
  },
  'ffn-norm': {
    title: T('FFN 前归一化'),
    explanation: T(
      '第二次 LayerNorm 读取已经混合上下文的 R，再送入 FFN。FFN 接下来逐 token 计算，参数在各位置之间共享。',
    ),
    prompt: T('FFN 会在这一步直接读取其他 token 的行吗？'),
    formula: 'Z_l=\\operatorname{LN}(R_l)',
    symbols: [terms.residual, terms.ln, terms.z],
    topic: 'normalization',
  },
  'ffn-expand': {
    title: T('扩维与 GELU'),
    explanation: T(
      '从 4 个特征扩到 6 个，再逐元素应用 GELU。非线性前后两个矩阵可直接对照；本例采用常见的 tanh 近似。',
    ),
    prompt: T('如果删掉 GELU，两次线性投影能合成一次吗？'),
    formula: 'G=\\operatorname{GELU}(Z_lW_1^{(l)})',
    symbols: [terms.z, terms.expand],
    topic: 'ffn',
  },
  'ffn-contract': {
    title: T('压回隐藏维数'),
    explanation: T('第二次投影把 6 维中间激活压回 4 维，才能与残差主路逐元素相加。'),
    prompt: T('为什么 FFN 输出的最后一维必须恢复为 4？'),
    formula: 'F=GW_2^{(l)}',
    symbols: [terms.contract],
    topic: 'ffn',
  },
  'ffn-add': {
    title: T('第二次残差与下一层'),
    explanation: T(
      '加上 FFN 分支后，这个 Block 才完成。下一层接收 H_l 作为新的隐藏状态，而不是重新查一次原始 token 向量。',
    ),
    prompt: T('Block 2 的 Q/K/V 与 Block 1 为什么通常不同？'),
    formula: 'H_l=R_l+F',
    symbols: [terms.hidden, terms.residual, terms.contract],
    topic: 'attention',
  },
  'final-norm': {
    title: T('最后一次归一化'),
    explanation: T(
      '所有 Block 结束后，再做最终 LayerNorm。这仍是每个位置的隐藏表示，还没有变成词表概率。',
    ),
    prompt: T('最终 LayerNorm 的输出轴是特征还是词表？'),
    formula: 'H_f=\\operatorname{LN}(H_L)',
    symbols: [terms.final, terms.hidden, terms.ln],
    topic: 'normalization',
  },
  logits: {
    title: T('投影到词表'),
    explanation: T(
      '用隐藏表示给词表里的每个候选打分。输出头绑定到输入嵌入表的转置；本例词表和隐藏维数恰好都为 4，但它们是不同的轴。',
    ),
    prompt: T('logits 的一列为什么现在代表候选 token？'),
    formula: '\\mathrm{logits}=H_fE^\\top',
    symbols: [terms.final, terms.logits, terms.transpose],
    topic: 'decoding',
  },
  probability: {
    title: T('沿词表做 Softmax'),
    explanation: T(
      '对每个位置，沿 a–d 这四个词表候选归一化；这次 Softmax 的用途与 Attention 的 Key 权重不同。',
    ),
    prompt: T('注意力权重和词表概率，哪个决定“取什么内容”，哪个决定“生成什么 token”？'),
    formula:
      'p_{t,v}=\\frac{e^{\\mathrm{logits}_{t,v}-m_t}}{\\sum_{u\\in\\mathcal V}e^{\\mathrm{logits}_{t,u}-m_t}}',
    symbols: [terms.probability, terms.sum],
    topic: 'decoding',
  },
  selection: {
    title: T('用最后位置选择下一 token'),
    explanation: T(
      '贪心选择只使用最后一个位置的分布。要继续生成，需要把选出的 token 追加为新输入；当前页面回放的是一次完整前向。',
    ),
    prompt: T('为什么前两行也有概率，生成却只使用最后一行？'),
    formula: '\\mathrm{next}=\\arg\\max_v p_{T-1,v}',
    symbols: [terms.selection, terms.lastProbability],
    topic: 'inference',
  },
};
