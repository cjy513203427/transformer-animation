import { LOCALE } from '../i18n';

/**
 * The reference book. Chinese pages link the Chinese original; English and German pages link the
 * English edition (there is no German one). The English edition is a rewrite: chapters match one
 * to one, but its sections are organised differently, so step labels there name only the chapter
 * and the topic, never a Chinese section number. Chapter URLs checked 2026-11-09.
 */
const ZH_BOOK = 'https://waylandz.com/llm-transformer-book/';
const EN_BOOK = 'https://waylandz.com/llm-transformer-book-en/';

/** Chinese chapter path → English-edition chapter path. */
const EN_CHAPTERS: Record<string, string> = {
  '第01章-GPT是什么-LLM发展简史与核心思想/': 'chapter-01-what-is-gpt/',
  '第02章-大模型的本质-就是两个文件/': 'chapter-02-large-models-are-two-files/',
  '第03章-Transformer全景图/': 'chapter-03-transformer-map/',
  '第04章-Tokenization-文字如何变成数字/': 'chapter-04-tokenization/',
  '第05章-Positional-Encoding-给文字加位置/': 'chapter-05-positional-encoding/',
  '第06章-LayerNorm与Softmax-数字的缩放与概率化/': 'chapter-06-layernorm-softmax/',
  '第07章-神经网络层-不需要懂也能理解Transformer/': 'chapter-07-neural-network-layers/',
  '第08章-线性变换的几何意义-矩阵乘法的本质/': 'chapter-08-linear-transforms/',
  '第09章-Attention的几何逻辑-为什么是点积/': 'chapter-09-attention-geometry/',
  '第10章-QKV到底是什么-Attention的三个主角/': 'chapter-10-qkv/',
  '第11章-Multi-Head-Attention-多视角理解/': 'chapter-11-multi-head-attention/',
  '第12章-QKV输出的本质/': 'chapter-12-qkv-output/',
  '第13章-残差连接与Dropout-训练稳定的秘密/': 'chapter-13-residual-dropout/',
  '第14章-词嵌入与位置信息的深层逻辑-为什么相加而不是拼接/': 'chapter-14-embedding-plus-position/',
  '第15章-Transformer完整前向传播-从输入到输出/': 'chapter-15-full-forward-pass/',
  '第16章-训练与推理的异同-为什么推理要一个字一个字生成/': 'chapter-16-training-vs-inference/',
  '第17章-学习率的理解-训练稳定的关键/': 'chapter-17-learning-rate/',
  '第21章-Flash-Attention-内存优化原理/': 'chapter-21-flash-attention/',
  '第22章-KV-Cache-推理加速/': 'chapter-22-kv-cache/',
  '第23章-MHA到MQA到GQA演进/': 'chapter-23-mha-mqa-gqa/',
  '第24章-Sparse与Infinite-Attention/': 'chapter-24-sparse-infinite-attention/',
  '第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/': 'chapter-25-positional-encoding-evolution/',
  '第26章-LoRA与QLoRA-高效微调/': 'chapter-26-lora-qlora/',
  '第27章-模型量化-GPTQ-AWQ-GGUF/': 'chapter-27-quantization/',
  '第28章-Prompt-Engineering-提示工程实战/': 'chapter-28-prompt-engineering/',
  '第29章-RLHF与偏好学习-让模型对齐人类/': 'chapter-29-rlhf-preference-learning/',
  '第30章-Mixture-of-Experts-稀疏激活的秘密/': 'chapter-30-mixture-of-experts/',
  '第31章-推理模型革命-从o1到R1/': 'chapter-31-reasoning-models/',
  '第32章-后Transformer架构-Mamba与混合模型/': 'chapter-32-post-transformer-architectures/',
  '附录A-Scaling-Law与计算量估算/': 'appendix-a-scaling-laws-compute/',
  '附录B-解码策略详解/': 'appendix-b-decoding-strategies/',
  '附录C-常见问题FAQ/': 'appendix-c-faq/',
};

/** Book home in the page's language. */
export const BOOK_URL = LOCALE === 'zh' ? ZH_BOOK : EN_BOOK;

/** Chapter URL in the page's language, from the Chinese chapter path. */
export function chapterUrl(zhChapter: string): string {
  if (LOCALE === 'zh') return ZH_BOOK + zhChapter;
  const en = EN_CHAPTERS[zhChapter];
  if (!en) throw new Error(`No English edition path for chapter ${zhChapter}`);
  return EN_BOOK + en;
}

/** Every chapter path the Chinese table of contents lists, for tests. */
export const CHAPTER_PATHS = Object.keys(EN_CHAPTERS);

/** Chapter pages linked from step labels. Step labels name the section; the link opens the chapter. */
export const BOOK = {
  ch06: chapterUrl('第06章-LayerNorm与Softmax-数字的缩放与概率化/'),
  ch07: chapterUrl('第07章-神经网络层-不需要懂也能理解Transformer/'),
  ch13: chapterUrl('第13章-残差连接与Dropout-训练稳定的秘密/'),
  ch16: chapterUrl('第16章-训练与推理的异同-为什么推理要一个字一个字生成/'),
  ch21: chapterUrl('第21章-Flash-Attention-内存优化原理/'),
  ch22: chapterUrl('第22章-KV-Cache-推理加速/'),
  ch23: chapterUrl('第23章-MHA到MQA到GQA演进/'),
  ch24: chapterUrl('第24章-Sparse与Infinite-Attention/'),
  ch26: chapterUrl('第26章-LoRA与QLoRA-高效微调/'),
  ch27: chapterUrl('第27章-模型量化-GPTQ-AWQ-GGUF/'),
  ch29: chapterUrl('第29章-RLHF与偏好学习-让模型对齐人类/'),
  ch30: chapterUrl('第30章-Mixture-of-Experts-稀疏激活的秘密/'),
  ch31: chapterUrl('第31章-推理模型革命-从o1到R1/'),
  ch32: chapterUrl('第32章-后Transformer架构-Mamba与混合模型/'),
  appB: chapterUrl('附录B-解码策略详解/'),
} as const;

/** [section label, chapter URL] */
export type BookLink = [string, string];
