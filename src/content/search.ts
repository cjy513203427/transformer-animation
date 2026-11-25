import { NOTE_DESTINATIONS } from './experimentLinks';
import { KNOWLEDGE_NOTES } from './notebook';
import { INTRO, TOPICS, type TopicId } from './topics';
import { localePath, T } from '../i18n';

export type SearchKind = 'experiment' | 'note';

export interface SearchItem {
  id: string;
  kind: SearchKind;
  title: string;
  subtitle: string;
  badge: string;
  href: string;
  /** Non-rendered bilingual/acronym keywords for cross-language and alias lookup. */
  keywords: string;
}

export interface SearchHit {
  item: SearchItem;
  score: number;
  /** Highlight [start, end) ranges inside item.title. */
  titleRanges: [number, number][];
}

interface TermMatch {
  score: number;
  ranges: [number, number][];
}

const WORD_BOUNDARY = /[\s\-_/·.,:;()（）【】[\]—–+]/;
const ASCII_WORD = /[a-z0-9]+/g;

export function normalizeText(text: string): string {
  return text
    .toLocaleLowerCase()
    .replace(/[\u3000\s]+/g, ' ')
    .trim();
}

/** Optimal String Alignment (Damerau-Levenshtein) distance with early exit when > maxDist. */
export function damerauLevenshtein(a: string, b: string, maxDist = 2): number {
  const m = a.length;
  const n = b.length;
  if (Math.abs(m - n) > maxDist) return maxDist + 1;
  if (m === 0) return n;
  if (n === 0) return m;

  const dp: number[][] = Array.from({ length: m + 1 }, () => Array<number>(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    let rowMin = maxDist + 1;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let val = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        val = Math.min(val, dp[i - 2][j - 2] + 1);
      }
      dp[i][j] = val;
      if (val < rowMin) rowMin = val;
    }
    if (rowMin > maxDist) return maxDist + 1;
  }
  return dp[m][n];
}

function mergeRanges(ranges: [number, number][]): [number, number][] {
  if (ranges.length <= 1) return ranges;
  const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const prev = out[out.length - 1];
    if (sorted[i][0] <= prev[1]) {
      prev[1] = Math.max(prev[1], sorted[i][1]);
    } else {
      out.push(sorted[i]);
    }
  }
  return out;
}

/** Ordered subsequence match with word-boundary and consecutive-run bonuses. */
function subsequenceMatch(term: string, rawLower: string): TermMatch | null {
  if (term.length < 2 || rawLower.length < term.length) return null;
  const indices: number[] = [];
  let searchFrom = 0;
  for (let i = 0; i < term.length; i++) {
    const ch = term[i];
    // Prefer a match right after a previous hit or at a word boundary when available nearby.
    let best = -1;
    let first = -1;
    for (let j = searchFrom; j < rawLower.length; j++) {
      if (rawLower[j] !== ch) continue;
      if (first === -1) first = j;
      const consecutive = indices.length > 0 && j === indices[indices.length - 1] + 1;
      const boundary = j === 0 || WORD_BOUNDARY.test(rawLower[j - 1]);
      if (consecutive || boundary) {
        best = j;
        break;
      }
    }
    const chosen = best !== -1 ? best : first;
    if (chosen === -1) return null;
    indices.push(chosen);
    searchFrom = chosen + 1;
  }

  const span = indices[indices.length - 1] - indices[0] + 1;
  // Prevent noisy matches scattered across a long paragraph.
  const isAscii = /^[a-z0-9]+$/.test(term);
  const maxSpan = isAscii ? Math.max(term.length * 4 + 4, 14) : Math.max(term.length * 4 + 2, 10);
  if (span > maxSpan) return null;

  let consecutiveCount = 0;
  let boundaryCount = 0;
  for (let i = 0; i < indices.length; i++) {
    if (i > 0 && indices[i] === indices[i - 1] + 1) consecutiveCount++;
    if (indices[i] === 0 || WORD_BOUNDARY.test(rawLower[indices[i] - 1])) boundaryCount++;
  }
  const density = term.length / span;
  const score = 34 + density * 20 + consecutiveCount * 3 + boundaryCount * 2;
  const ranges = mergeRanges(indices.map((idx): [number, number] => [idx, idx + 1]));
  return { score, ranges };
}

/** Typo-tolerant match against ASCII words or CJK windows inside a field. */
function typoMatch(term: string, rawLower: string): TermMatch | null {
  if (term.length < 3) return null;
  const maxDist = term.length <= 5 ? 1 : 2;
  let bestScore = 0;
  let bestRange: [number, number] | null = null;

  if (/^[a-z0-9]+$/.test(term)) {
    ASCII_WORD.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = ASCII_WORD.exec(rawLower)) !== null) {
      const word = match[0];
      if (word.length < 3) continue;
      // Check full word and same-length prefix (supports typos while typing a prefix).
      const distFull = damerauLevenshtein(term, word, maxDist);
      const prefix = word.length > term.length ? word.slice(0, term.length) : word;
      const distPrefix =
        word.length > term.length ? damerauLevenshtein(term, prefix, maxDist) : distFull;
      const dist = Math.min(distFull, distPrefix);
      if (dist <= maxDist) {
        const score = 50 - dist * 14 - (distFull > distPrefix ? 4 : 0);
        if (score > bestScore) {
          bestScore = score;
          bestRange = [match.index, match.index + Math.min(word.length, term.length + dist)];
        }
      }
    }
  } else {
    // Non-ASCII (e.g. CJK): check sliding windows of length term.length for 1-char typo tolerance.
    const winLen = term.length;
    if (winLen >= 3 && rawLower.length >= winLen) {
      for (let i = 0; i <= rawLower.length - winLen; i++) {
        const slice = rawLower.slice(i, i + winLen);
        if (slice.includes(' ')) continue;
        const dist = damerauLevenshtein(term, slice, 1);
        if (dist <= 1) {
          const score = 44 - dist * 12;
          if (score > bestScore) {
            bestScore = score;
            bestRange = [i, i + winLen];
          }
        }
      }
    }
  }

  return bestRange ? { score: bestScore, ranges: [bestRange] } : null;
}

/** Score a single normalized query term against a raw field string. */
export function scoreTermInText(term: string, rawText: string): TermMatch | null {
  if (!term || !rawText) return null;
  const rawLower = rawText.toLocaleLowerCase();

  // 1. Direct substring / exact / prefix / word-boundary hit
  const idx = rawLower.indexOf(term);
  if (idx !== -1) {
    let score = 76;
    if (rawLower === term) score = 120;
    else if (idx === 0) score = 102;
    else if (WORD_BOUNDARY.test(rawLower[idx - 1])) score = 90;
    else score = Math.max(68, 78 - Math.min(10, Math.floor(idx / 8)));
    return { score, ranges: [[idx, idx + term.length]] };
  }

  // 2. Ordered subsequence match (e.g. "flsh" -> "FlashAttention", "线变" -> "线性变换")
  const sub = subsequenceMatch(term, rawLower);
  // 3. Edit-distance typo tolerance (e.g. "attnetion" -> "Attention", "qunatize" -> "Quantization")
  const typo = typoMatch(term, rawLower);

  if (sub && typo) return sub.score >= typo.score ? sub : typo;
  return sub ?? typo;
}

export function fuzzyMatchesText(query: string, text: string): boolean {
  const norm = normalizeText(query);
  if (!norm) return true;
  const terms = norm.split(' ').filter(Boolean).slice(0, 6);
  return terms.every((term) => scoreTermInText(term, text) !== null);
}

const TOPIC_ALIASES: Record<TopicId, string> = {
  attention: 'attention self-attention scaled dot product qkv multi-head mha 注意力 点积 多头 权重',
  linear:
    'linear transform matrix multiplication geometry determinant dot product 2d shear rotate 矩阵乘法 线性变换 几何 行列式 剪切 旋转 点积',
  bpe: 'tokenization tokenizer bpe subword byte utf-8 embedding vocab 分词 词表 字节 嵌入',
  position: 'positional encoding sinusoidal rope alibi rotary relative 位置编码 正弦 旋转位置编码 相对位置',
  normalization: 'layernorm rmsnorm softmax variance affine temperature 归一化 均方根 概率',
  ffn: 'ffn feed forward mlp relu gelu swiglu residual dropout jacobian 前馈 非线性 残差 梯度',
  forward:
    'forward pass decoder block model.py pytorch embedding lm head logits pre-norm 完整前向 前向传播 架构地图 模型定义',
  inference: 'training inference prefill decode kv cache teacher forcing causal mask inference.py 训练 推理 缓存 生成',
  decoding: 'decoding sampling temperature top-k top-p nucleus greedy beam search 解码 采样 温度 束搜索 贪心',
  kv: 'kv cache mha mqa gqa mla deepseek-v2 latent absorb rope 缓存 共享头 压缩 矩阵吸收',
  flash: 'flashattention flash attention online softmax tiling hbm sram io 分块 在线归一化 显存',
  'prompt-chain': 'prompt engineering few-shot zero-shot chain 提示工程 提示词 示例 提示链',
  'prompt-vote': 'self-consistency voting majority 答案投票 自洽 多次采样',
  'prompt-search': 'cot tot react chain of thought tree of thoughts 思维链 思路搜索 回溯',
  context: 'context extension pi position interpolation ntk-aware yarn rope 上下文扩展 长上下文 外推 插值',
  memory: 'sparse attention linear attention infini-attention sliding window longformer 稀疏 滑动窗口 线性 压缩记忆',
  moe: 'moe mixture of experts router top-k shared expert capacity drop aux loss deepseek mixtral switch 混合专家 路由 共享专家 容量 丢弃 辅助损失 专家并行',
  ssm: 'ssm mamba s4 state space zoh selective scan rwkv retnet hyena 状态空间 选择性 递推 混合架构',
  hybrid:
    'post-transformer mamba block ssd hybrid jamba rwkv retnet hyena s4 后transformer 混合架构 状态空间 账本',
  training: 'training cross entropy loss backward sgd warmup train.py 交叉熵 反向传播 梯度 参数更新',
  optimizer: 'optimizer sgd adam adamw momentum weight decay l2 优化器 动量 权重衰减 学习率',
  budget: 'parameters flops 12ld2 6nd scaling law chinchilla gpt-2 llama 参数量 计算量 缩放定律',
  policy: 'rlhf ppo actor critic rollout advantage kl dpo 策略更新 奖励模型 价值函数 优势 对齐',
  lora: 'lora qlora nf4 adapter low-rank fine-tuning 微调 低秩 适配器 冻结',
  quantization:
    'quantization int4 int8 fp16 symmetric asymmetric zero-point awq gptq gguf rtn calibration 量化 对称 非对称 零点 异常值 分组 校准',
  preference: 'rlhf preference dpo ppo clip grpo deepseek r1 distillation reward 偏好学习 组优势 裁剪 蒸馏 推理模型',
};

const KIND_BONUS: Record<SearchKind, number> = {
  experiment: 12,
  note: 0,
};

let cachedItems: SearchItem[] | null = null;

/** Builds the complete localized index of experiments and notes. */
export function getSearchItems(): SearchItem[] {
  if (cachedItems) return cachedItems;
  const items: SearchItem[] = [
    {
      id: 'intro',
      kind: 'experiment',
      title: INTRO.title,
      subtitle: INTRO.subtitle,
      badge: '00',
      href: INTRO.href,
      keywords: 'intro overview start gpt next token 导读 首页 开始 流水线',
    },
    ...TOPICS.map((topic): SearchItem => ({
      id: `topic-${topic.id}`,
      kind: 'experiment',
      title: topic.title,
      subtitle: topic.subtitle,
      badge: topic.number,
      href: topic.href,
      keywords: `${topic.id} ${TOPIC_ALIASES[topic.id] ?? ''}`,
    })),
    ...KNOWLEDGE_NOTES.map((note): SearchItem => ({
      id: `note-${note.id}`,
      kind: 'note',
      title: note.title,
      subtitle: note.question,
      badge: T('笔记'),
      href: `${localePath(NOTE_DESTINATIONS[note.id])}?note=${note.id}`,
      keywords: `${note.id} ${note.body.join(' ')} ${note.example}`,
    })),
  ];
  cachedItems = items;
  return items;
}

export function searchSite(
  query: string,
  kind: SearchKind | 'all' = 'all',
  limit = 18,
): SearchHit[] {
  const items = getSearchItems();
  const pool = kind === 'all' ? items : items.filter((item) => item.kind === kind);
  const norm = normalizeText(query);
  if (!norm) {
    return pool.slice(0, limit).map((item) => ({ item, score: 0, titleRanges: [] }));
  }

  const terms = norm.split(' ').filter(Boolean).slice(0, 6);
  const hits: SearchHit[] = [];

  for (const item of pool) {
    let total = 0;
    let matchedAll = true;
    const titleRanges: [number, number][] = [];

    for (const term of terms) {
      const titleHit = scoreTermInText(term, item.title);
      const subHit = scoreTermInText(term, item.subtitle);
      const badgeHit = scoreTermInText(term, item.badge);
      const keyHit = scoreTermInText(term, item.keywords);

      const best = Math.max(
        titleHit ? titleHit.score * 1.45 : 0,
        badgeHit ? badgeHit.score * 1.15 : 0,
        subHit ? subHit.score * 1.05 : 0,
        keyHit ? keyHit.score * 0.85 : 0,
      );
      if (best <= 0) {
        matchedAll = false;
        break;
      }
      total += best;
      if (titleHit) titleRanges.push(...titleHit.ranges);
    }

    if (matchedAll) {
      hits.push({
        item,
        score: total + KIND_BONUS[item.kind],
        titleRanges: mergeRanges(titleRanges),
      });
    }
  }

  hits.sort((a, b) => b.score - a.score || a.item.title.localeCompare(b.item.title));
  return hits.slice(0, limit);
}

/** Split text into highlighted and plain segments using [start, end) ranges. */
export function splitHighlightSegments(
  text: string,
  ranges: [number, number][],
): { text: string; hit: boolean }[] {
  if (!ranges.length) return [{ text, hit: false }];
  const out: { text: string; hit: boolean }[] = [];
  let cursor = 0;
  for (const [start, end] of mergeRanges(ranges)) {
    const s = Math.max(cursor, Math.min(text.length, start));
    const e = Math.max(s, Math.min(text.length, end));
    if (s > cursor) out.push({ text: text.slice(cursor, s), hit: false });
    if (e > s) out.push({ text: text.slice(s, e), hit: true });
    cursor = e;
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor), hit: false });
  return out;
}
