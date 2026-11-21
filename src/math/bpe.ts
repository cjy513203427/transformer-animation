/**
 * Character-level BPE as in Sennrich et al. (arXiv 1508.07909, Algorithm 1): every word is split
 * into characters plus an end-of-word symbol, and each merge joins the most frequent adjacent
 * pair. Ties go to the lexicographically smallest pair, so results are reproducible.
 */
export const END = '</w>';

export interface CorpusWord {
  word: string;
  count: number;
}
export interface Merge {
  pair: [string, string];
  count: number;
  token: string;
}
export interface BPESnapshot {
  /** Segmentation of each corpus word after this many merges. */
  words: string[][];
  /** Adjacent pairs and their weighted counts, most frequent first. */
  pairs: { pair: [string, string]; count: number }[];
  /** Sum of tokens over the corpus, each word weighted by its count. */
  totalTokens: number;
  /** Vocabulary in ID order: base symbols, then merged tokens in the order learned. */
  vocab: string[];
}

const split = (word: string) => [...word, END];
const join = (pair: [string, string]) => (pair[1] === END ? pair[0] + END : pair[0] + pair[1]);

function countPairs(corpus: CorpusWord[], words: string[][]) {
  const counts = new Map<string, { pair: [string, string]; count: number }>();
  words.forEach((symbols, w) => {
    for (let i = 0; i + 1 < symbols.length; i++) {
      const key = symbols[i] + '\u0000' + symbols[i + 1];
      const entry = counts.get(key) ?? { pair: [symbols[i], symbols[i + 1]], count: 0 };
      entry.count += corpus[w].count;
      counts.set(key, entry);
    }
  });
  return [...counts.values()].sort(
    (a, b) =>
      b.count - a.count ||
      (a.pair[0] + ' ' + a.pair[1] < b.pair[0] + ' ' + b.pair[1] ? -1 : 1),
  );
}

/** Replace every left-to-right occurrence of the pair, as Sennrich's apply step does. */
export function applyMerge(symbols: string[], merge: Merge) {
  const out: string[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i + 1 < symbols.length && symbols[i] === merge.pair[0] && symbols[i + 1] === merge.pair[1]) {
      out.push(merge.token);
      i++;
    } else out.push(symbols[i]);
  }
  return out;
}

export function learnBPE(corpus: CorpusWord[], maxMerges: number) {
  const base = [...new Set(corpus.flatMap((c) => [...c.word]))].sort();
  let words = corpus.map((c) => split(c.word));
  const merges: Merge[] = [];
  const snapshot = (): BPESnapshot => ({
    words,
    pairs: countPairs(corpus, words),
    totalTokens: words.reduce((sum, w, i) => sum + w.length * corpus[i].count, 0),
    vocab: [...base, END, ...merges.map((m) => m.token)],
  });
  const snapshots = [snapshot()];
  while (merges.length < maxMerges) {
    const best = snapshots[snapshots.length - 1].pairs[0];
    if (!best) break;
    const merge = { pair: best.pair, count: best.count, token: join(best.pair) };
    merges.push(merge);
    words = words.map((w) => applyMerge(w, merge));
    snapshots.push(snapshot());
  }
  return { base, merges, snapshots };
}

/** Encode a new word with the first k merges, in the order they were learned. */
export function encodeWord(word: string, merges: Merge[], k: number) {
  return merges.slice(0, k).reduce((symbols, merge) => applyMerge(symbols, merge), split(word));
}

export const utf8Bytes = (text: string) => [...new TextEncoder().encode(text)];
