export type TopicId =
  | 'attention'
  | 'position'
  | 'inference'
  | 'kv'
  | 'moe'
  | 'flash'
  | 'normalization'
  | 'ffn'
  | 'lora'
  | 'quantization'
  | 'decoding'
  | 'preference'
  | 'memory'
  | 'ssm';

export type TopicGroupId = 'foundation' | 'generation' | 'architecture' | 'adaptation';
export interface Topic {
  id: TopicId;
  number: string;
  group: TopicGroupId;
  href: string;
  title: string;
  subtitle: string;
}

export const TOPIC_GROUPS: { id: TopicGroupId; title: string; description: string }[] = [
  { id: 'foundation', title: '基础计算', description: '先理解一条前向计算链' },
  { id: 'generation', title: '生成与效率', description: '从生成过程走到缓存与分块' },
  { id: 'architecture', title: '架构扩展', description: '比较历史访问和稀疏激活' },
  { id: 'adaptation', title: '训练与部署', description: '理解更新、压缩与偏好目标' },
];

/** Learning order, independent of the book's chapter numbers and implementation chronology. */
const TOPIC_DEFINITIONS: Omit<Topic, 'number'>[] = [
  {
    id: 'attention',
    group: 'foundation',
    href: '/attention',
    title: '跟图理解 Attention',
    subtitle: '单头、多头，逐步核对',
  },
  {
    id: 'position',
    group: 'foundation',
    href: '/position',
    title: '位置编码',
    subtitle: '顺序从哪里进入计算',
  },
  {
    id: 'normalization',
    group: 'foundation',
    href: '/normalization',
    title: '归一化',
    subtitle: 'LayerNorm、RMSNorm 与 Softmax',
  },
  {
    id: 'ffn',
    group: 'foundation',
    href: '/ffn-residual',
    title: 'FFN 与残差',
    subtitle: '非线性、梯度路径与 Dropout',
  },
  {
    id: 'inference',
    group: 'generation',
    href: '/inference-cache',
    title: '训练、生成与缓存',
    subtitle: 'Prefill、Decode 与 KV 缓存',
  },
  {
    id: 'decoding',
    group: 'generation',
    href: '/decoding',
    title: '解码策略',
    subtitle: '温度、Top-K、Top-P 与搜索',
  },
  {
    id: 'kv',
    group: 'generation',
    href: '/kv-architectures',
    title: 'KV 架构对照',
    subtitle: 'MHA、GQA、MQA 与 MLA 的缓存',
  },
  {
    id: 'flash',
    group: 'generation',
    href: '/flash-attention',
    title: 'FlashAttention',
    subtitle: '分块、在线归一化与中间存储',
  },
  {
    id: 'memory',
    group: 'architecture',
    href: '/attention-variants',
    title: '稀疏与压缩记忆',
    subtitle: 'Sparse、Linear 与 Infini-attention',
  },
  {
    id: 'moe',
    group: 'architecture',
    href: '/moe',
    title: 'MoE 与专家并行',
    subtitle: '路由、加权合并与通信账本',
  },
  {
    id: 'ssm',
    group: 'architecture',
    href: '/ssm-mamba',
    title: 'SSM / Mamba',
    subtitle: '状态递推与选择性参数',
  },
  {
    id: 'lora',
    group: 'adaptation',
    href: '/lora',
    title: 'LoRA / QLoRA',
    subtitle: '冻结底座与低秩更新',
  },
  {
    id: 'quantization',
    group: 'adaptation',
    href: '/quantization',
    title: '量化',
    subtitle: '刻度、分组与误差',
  },
  {
    id: 'preference',
    group: 'adaptation',
    href: '/preference',
    title: '偏好学习 / GRPO',
    subtitle: '偏好对、组优势与策略约束',
  },
];

/** Only working experiments enter the directory. Numbers describe sequence, not completion. */
export const TOPICS: Topic[] = TOPIC_DEFINITIONS.map((topic, index) => ({
  ...topic,
  number: String(index + 1).padStart(2, '0'),
}));

export const ROADMAP = ['Tokenization / BPE', '学习率与 Scaling Law', '提示与推理预算'];

export const BOOK_URL = 'https://waylandz.com/llm-transformer-book/';
