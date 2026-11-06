export type TopicId = 'attention' | 'position';

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
];

export const ROADMAP = ['训练、生成与缓存', 'KV 架构对照', 'MoE 与专家并行', 'FlashAttention'];

export const BOOK_URL = 'https://waylandz.com/llm-transformer-book/';
