import { localePath, T } from '../i18n';

export type TopicId =
  | 'attention'
  | 'linear'
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
  | 'ssm'
  | 'training'
  | 'policy'
  | 'prompt-chain'
  | 'prompt-vote'
  | 'prompt-search'
  | 'budget'
  | 'optimizer'
  | 'context'
  | 'bpe';

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
    id: 'linear',
    group: 'foundation',
    href: '/linear-transform',
    title: T('线性变换与几何'),
    subtitle: T('网格变形、行列式、点积与复合'),
  },
  { id:'bpe', group:'foundation', href:'/tokenization', title:T('分词与 BPE'), subtitle:T('合并相邻对、切分新词与查表') },
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
  { id:'prompt-chain', group:'generation', href:'/prompt-engineering', title:T('提示链与核对'), subtitle:T('提取、计算、核对与输出') },
  { id:'prompt-vote', group:'generation', href:'/prompt-voting', title:T('答案投票'), subtitle:T('Self-Consistency 与独立验证') },
  { id:'prompt-search', group:'generation', href:'/prompt-search', title:T('思路搜索'), subtitle:T('展开、检查与回退') },
  { id:'context', group:'architecture', href:'/context-extension', title:T('上下文扩展'), subtitle:T('PI、NTK-aware 与 YaRN') },
  { id:'memory', group:'architecture', href:'/attention-variants', title:T('稀疏与压缩记忆'), subtitle:T('Sparse、Linear 与 Infini-attention') },
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
  { id:'training', group:'adaptation', href:'/training-step', title:T('前向、损失与参数更新'), subtitle:T('用同一例子算完一次训练') },
  { id:'optimizer', group:'adaptation', href:'/optimizer', title:T('优化器：SGD、Adam 与 AdamW'), subtitle:T('动量、按坐标归一化与权重衰减') },
  { id:'budget', group:'adaptation', href:'/model-budget', title:T('参数量与计算量'), subtitle:T('从配置数出参数与 FLOPs') },
  { id:'policy', group:'adaptation', href:'/policy-update', title:T('RLHF 与策略更新'), subtitle:T('奖励、价值、优势与 PPO') },
  { id:'lora', group:'adaptation', href:'/lora', title:'LoRA / QLoRA', subtitle:T('冻结底座与低秩更新') },
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
    title: T('RLHF / 偏好学习 / GRPO'),
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

export const ROADMAP = [T('真实 Tokenizer 与完整模型训练')];
