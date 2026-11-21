import { localePath, T } from '../i18n';

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
  { id: 'foundation', title: T('基础计算'), description: T('先理解一条前向计算链') },
  { id: 'generation', title: T('生成与效率'), description: T('从生成过程走到缓存与分块') },
  { id: 'architecture', title: T('架构扩展'), description: T('比较历史访问和稀疏激活') },
  { id: 'adaptation', title: T('训练与部署'), description: T('理解更新、压缩与偏好目标') },
];

/** Learning order, independent of the book's chapter numbers and implementation chronology. */
const TOPIC_DEFINITIONS: Omit<Topic, 'number'>[] = [
  {
    id: 'attention',
    group: 'foundation',
    href: '/attention',
    title: T('跟图理解 Attention'),
    subtitle: T('单头、多头，逐步核对'),
  },
  {
    id: 'position',
    group: 'foundation',
    href: '/position',
    title: T('位置编码'),
    subtitle: T('顺序从哪里进入计算'),
  },
  {
    id: 'normalization',
    group: 'foundation',
    href: '/normalization',
    title: T('归一化'),
    subtitle: T('LayerNorm、RMSNorm 与 Softmax'),
  },
  {
    id: 'ffn',
    group: 'foundation',
    href: '/ffn-residual',
    title: T('FFN 与残差'),
    subtitle: T('非线性、梯度路径与 Dropout'),
  },
  {
    id: 'inference',
    group: 'generation',
    href: '/inference-cache',
    title: T('训练、生成与缓存'),
    subtitle: T('Prefill、Decode 与 KV 缓存'),
  },
  {
    id: 'decoding',
    group: 'generation',
    href: '/decoding',
    title: T('解码策略'),
    subtitle: T('温度、Top-K、Top-P 与搜索'),
  },
  {
    id: 'kv',
    group: 'generation',
    href: '/kv-architectures',
    title: T('KV 架构对照'),
    subtitle: T('MHA、GQA、MQA 与 MLA 的缓存'),
  },
  {
    id: 'flash',
    group: 'generation',
    href: '/flash-attention',
    title: 'FlashAttention',
    subtitle: T('分块、在线归一化与中间存储'),
  },
  {
    id: 'memory',
    group: 'architecture',
    href: '/attention-variants',
    title: T('稀疏与压缩记忆'),
    subtitle: T('Sparse、Linear 与 Infini-attention'),
  },
  {
    id: 'moe',
    group: 'architecture',
    href: '/moe',
    title: T('MoE 与专家并行'),
    subtitle: T('路由、加权合并与通信账本'),
  },
  {
    id: 'ssm',
    group: 'architecture',
    href: '/ssm-mamba',
    title: 'SSM / Mamba',
    subtitle: T('状态递推与选择性参数'),
  },
  {
    id: 'lora',
    group: 'adaptation',
    href: '/lora',
    title: 'LoRA / QLoRA',
    subtitle: T('冻结底座与低秩更新'),
  },
  {
    id: 'quantization',
    group: 'adaptation',
    href: '/quantization',
    title: T('量化'),
    subtitle: T('刻度、分组与误差'),
  },
  {
    id: 'preference',
    group: 'adaptation',
    href: '/preference',
    title: T('偏好学习 / GRPO'),
    subtitle: T('偏好对、组优势与策略约束'),
  },
];

/** Only working experiments enter the directory. Numbers describe sequence, not completion. */
export const TOPICS: Topic[] = TOPIC_DEFINITIONS.map((topic, index) => ({
  ...topic,
  href: localePath(topic.href),
  number: String(index + 1).padStart(2, '0'),
}));

/** The introduction at the site root; it comes before the numbered experiments. */
export const INTRO = {
  href: localePath('/'),
  number: '00',
  title: T('导读'),
  subtitle: T('大模型在做什么，这里怎么学'),
};

export const ROADMAP = [T('学习率与 Scaling Law'), T('提示与推理预算')];
