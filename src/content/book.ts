import { BOOK_URL } from './topics';

/**
 * Chapter pages of the reference book, as linked from its table of contents
 * (checked 2026-11-08). Step labels name the section; the link opens the chapter.
 */
export const BOOK = {
  ch06: BOOK_URL + '第06章-LayerNorm与Softmax-数字的缩放与概率化/',
  ch07: BOOK_URL + '第07章-神经网络层-不需要懂也能理解Transformer/',
  ch13: BOOK_URL + '第13章-残差连接与Dropout-训练稳定的秘密/',
  ch16: BOOK_URL + '第16章-训练与推理的异同-为什么推理要一个字一个字生成/',
  ch21: BOOK_URL + '第21章-Flash-Attention-内存优化原理/',
  ch22: BOOK_URL + '第22章-KV-Cache-推理加速/',
  ch23: BOOK_URL + '第23章-MHA到MQA到GQA演进/',
  ch24: BOOK_URL + '第24章-Sparse与Infinite-Attention/',
  ch26: BOOK_URL + '第26章-LoRA与QLoRA-高效微调/',
  ch27: BOOK_URL + '第27章-模型量化-GPTQ-AWQ-GGUF/',
  ch29: BOOK_URL + '第29章-RLHF与偏好学习-让模型对齐人类/',
  ch30: BOOK_URL + '第30章-Mixture-of-Experts-稀疏激活的秘密/',
  ch31: BOOK_URL + '第31章-推理模型革命-从o1到R1/',
  ch32: BOOK_URL + '第32章-后Transformer架构-Mamba与混合模型/',
  appB: BOOK_URL + '附录B-解码策略详解/',
} as const;

/** [section label, chapter URL] */
export type BookLink = [string, string];
