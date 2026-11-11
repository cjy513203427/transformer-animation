import { T } from '../i18n';
import type { TopicGroupId, TopicId } from './topics';

/**
 * The introduction at the site root: what a language model does, where each experiment sits
 * in the pipeline, and how to use the labs. Plain language first; the experiments carry the math.
 */

export interface Candidate {
  token: string;
  /** Invented score for the demo; the probability is computed from it with softmax. */
  score: number;
  /** Choosing this candidate finishes the sentence. */
  ends?: boolean;
  /** Shown instead of the token when the token alone is hard to see, such as a full stop. */
  label?: string;
}

/** Three rounds of “guess the next word”. Scores are made up; probabilities are real softmax. */
export const PROMPT = T('早上我想来一杯');
export const ROUNDS: Candidate[][] = [
  [
    { token: T('咖啡'), score: 2.1 },
    { token: T('茶'), score: 1.5 },
    { token: T('可可'), score: 0.4 },
    { token: T('水'), score: -0.3 },
  ],
  [
    { token: T('加'), score: 1.4 },
    { token: T('。'), label: T('。（句号）'), score: 1.0, ends: true },
    { token: T('和'), score: 0.1 },
  ],
  [
    { token: T('牛奶'), score: 2.0 },
    { token: T('糖'), score: 1.3 },
    { token: T('冰'), score: 0.2 },
  ],
];

export interface Station {
  title: string;
  english: string;
  text: string;
  /** Experiments that explain this station. */
  topics: TopicId[];
}

export const STATIONS: Station[] = [
  {
    title: T('切成小块'),
    english: 'Tokenization',
    text: T('文字先被切成一个个 token，每个 token 在词表里查到一个编号。'),
    topics: [],
  },
  {
    title: T('变成一串数字'),
    english: 'Embedding + Position',
    text: T(
      '每个编号换成一串数字，也就是一个向量。光有这些向量，模型还分不清“我爱你”和“你爱我”，所以还要把“谁排第几”的信息加进去。',
    ),
    topics: ['position'],
  },
  {
    title: T('互相看一看'),
    english: 'Attention',
    text: T(
      '每个词回头看看前面的词，决定从谁那里拿多少信息。比如读到“它”的时候，可以去前文找“它”可能指的是谁。',
    ),
    topics: ['attention'],
  },
  {
    title: T('各自消化一下'),
    english: 'Feed Forward',
    text: T(
      '每个词再单独把拿到的信息加工一遍。旁边还有归一化和残差连接：前者让数字保持在合适的大小，后者保证原来的信息不会丢。',
    ),
    topics: ['normalization', 'ffn'],
  },
  {
    title: T('一层又一层'),
    english: 'Block × N',
    text: T(
      '上面两站合起来叫一个 Block。大模型会把几十个 Block 叠起来，每一层都在上一层的结果上继续加工。',
    ),
    topics: [],
  },
  {
    title: T('打分、挑一个词'),
    english: 'Logits → Softmax → Decoding',
    text: T(
      '最后一层的结果变成词表里每个候选词的分数，经过 Softmax 变成概率，再按某种规则挑出一个词。',
    ),
    topics: ['decoding'],
  },
  {
    title: T('接上去，再来一轮'),
    english: 'Append & repeat',
    text: T(
      '新词接到句子末尾，整个过程从头再来。为了不每次都把前文重算一遍，模型会把算过的一部分结果存进缓存。',
    ),
    topics: ['inference'],
  },
];

export const GLOSSARY: { term: string; english: string; text: string }[] = [
  {
    term: T('向量'),
    english: 'vector',
    text: T('一串数字，比如 [0.5, −1, 2]。每个 token 在每一层都用一个向量来表示。'),
  },
  {
    term: T('矩阵'),
    english: 'matrix',
    text: T('一张数字表格。在这些实验里，通常一行对应一个 token。'),
  },
  {
    term: T('矩阵乘法'),
    english: 'matrix multiplication',
    text: T(
      '拿左边的一行和右边的一列，对应位置相乘再全部加起来，就得到结果里的一个格子。实验里的“投影”“打分”说的都是它。',
    ),
  },
  {
    term: T('参数（权重）'),
    english: 'parameters / weights',
    text: T('训练时学出来的数字，比如 $W_Q$。模型回答问题的时候，它们固定不动。'),
  },
  {
    term: T('激活'),
    english: 'activations',
    text: T('处理当前这段文字时临时算出来的数字，比如 Q、K、V。换一句话，它们就跟着变。'),
  },
  {
    term: T('概率分布'),
    english: 'probability distribution',
    text: T('一组都在 0 到 1 之间、加起来正好等于 1 的数。'),
  },
];

export const HOW_TO: string[] = [
  T('先读开头的问题，自己猜一猜答案。猜错了也没关系，猜过之后答案才记得住。'),
  T('一步一步往下点。图、数字、公式和文字说明会一起换到同一步。'),
  T('改一个数之前，先预测会发生什么，再看实际算出来的结果。'),
  T('最后把说明藏起来，用自己的话讲一遍。讲得出来，才算真的懂了。'),
];

/** One conversational sentence per stage of the learning path. */
export const GROUP_STORY: Record<TopicGroupId, string> = {
  foundation: T('先把一层 Transformer 里的计算走通：Attention、位置、归一化和 FFN。'),
  generation: T('模型怎样一个词一个词地往外生成，以及怎样少算、少存、少搬数据。'),
  architecture: T('上下文很长、模型很大时的几种改造：只看一部分历史、压缩记忆、只叫醒一部分专家。'),
  adaptation: T('怎样便宜地微调和压缩模型，以及怎样让它的回答更合人意。'),
};
