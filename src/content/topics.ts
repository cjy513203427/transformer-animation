export type TopicId = 'attention' | 'position' | 'inference' | 'kv';

/** Implemented experiments. Everything else stays in ROADMAP until it has a working page. */
export const TOPICS: {
  id: TopicId;
  number: string;
  href: string;
  title: string;
  subtitle: string;
}[] = [
  {
    id: 'attention',
    number: '01',
    href: '/attention',
    title: '跟图理解 Attention',
    subtitle: '单头、多头，逐步核对',
  },
  {
    id: 'position',
    number: '02',
    href: '/position',
    title: '位置编码',
    subtitle: '顺序从哪里进入计算',
  },
  {
    id: 'inference',
    number: '03',
    href: '/inference-cache',
    title: '训练、生成与缓存',
    subtitle: 'Prefill、Decode 与 KV 缓存',
  },
  {
    id: 'kv',
    number: '04',
    href: '/kv-architectures',
    title: 'KV 架构对照',
    subtitle: 'MHA、GQA、MQA 与 MLA 的缓存',
  },
];

export const ROADMAP = ['MoE 与专家并行', 'FlashAttention'];

export const BOOK_URL = 'https://waylandz.com/llm-transformer-book/';
