import type { CorpusWord } from '../math/bpe';

/** Sennrich et al.'s toy dictionary (low, lower, newest, widest). */
export const BPE_PRESETS: Record<'sennrich' | 'shared' | 'tie', CorpusWord[]> = {
  sennrich: [
    { word: 'low', count: 5 },
    { word: 'lower', count: 2 },
    { word: 'newest', count: 6 },
    { word: 'widest', count: 3 },
  ],
  shared: [
    { word: 'play', count: 6 },
    { word: 'played', count: 4 },
    { word: 'playing', count: 3 },
    { word: 'replay', count: 2 },
  ],
  tie: [
    { word: 'ab', count: 2 },
    { word: 'ba', count: 2 },
  ],
};
export const MAX_MERGES = 12;
export const WORD_PATTERN = /^[a-z]{1,12}$/;

export interface BPEState {
  version: 1;
  corpus: CorpusWord[];
  /** Number of merges applied, 0 … MAX_MERGES. */
  step: number;
  probe: string;
  view: { recall: boolean };
}
export const initialBPEState = (): BPEState => ({
  version: 1,
  corpus: BPE_PRESETS.sennrich.map((w) => ({ ...w })),
  step: 0,
  probe: 'lowest',
  view: { recall: false },
});

export const validWord = (w: unknown): w is string => typeof w === 'string' && WORD_PATTERN.test(w);
export const validCount = (c: unknown): c is number =>
  typeof c === 'number' && Number.isInteger(c) && c >= 1 && c <= 50;

export const serializeBPEState = (s: BPEState) => '#bpe=' + encodeURIComponent(JSON.stringify(s));

export function restoreBPEState(hash: string): { state: BPEState; invalid: boolean } {
  if (!hash) return { state: initialBPEState(), invalid: false };
  const fallback = { state: initialBPEState(), invalid: true };
  if (!hash.startsWith('#bpe=') || hash.length > 1500) return fallback;
  try {
    const x = JSON.parse(decodeURIComponent(hash.slice(5))) as Record<string, unknown>;
    const view = x?.view as Record<string, unknown> | undefined;
    if (
      !x ||
      x.version !== 1 ||
      !Array.isArray(x.corpus) ||
      x.corpus.length < 1 ||
      x.corpus.length > 6 ||
      !x.corpus.every(
        (e: unknown) =>
          !!e && typeof e === 'object' && validWord((e as CorpusWord).word) && validCount((e as CorpusWord).count),
      ) ||
      new Set(x.corpus.map((e: CorpusWord) => e.word)).size !== x.corpus.length ||
      typeof x.step !== 'number' ||
      !Number.isInteger(x.step) ||
      x.step < 0 ||
      x.step > MAX_MERGES ||
      !validWord(x.probe) ||
      !view ||
      typeof view.recall !== 'boolean'
    )
      return fallback;
    return {
      invalid: false,
      state: {
        version: 1,
        corpus: (x.corpus as CorpusWord[]).map(({ word, count }) => ({ word, count })),
        step: x.step,
        probe: x.probe as string,
        view: { recall: view.recall },
      },
    };
  } catch {
    return fallback;
  }
}
