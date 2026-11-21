export const MAX_MERGES = 8;
export const MAX_TEXT_LENGTH = 48;
export const EMBEDDING_DIMENSIONS = 3;
export const CORPORA = {
  pairs: 'abab abab ab ab aba 猫猫 猫',
  repeats: 'aaa aaa aa aaa ab',
} as const;
export type CorpusId = keyof typeof CORPORA;

const utf8 = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const BYTE_COUNT = 256;
export const isWellFormed = (text: string) => !/[\uD800-\uDFFF]/u.test(text);

// Keep whitespace as pieces too, so joining their UTF-8 bytes recovers the original text.
export function bytePieces(text: string): number[][] {
  if (!isWellFormed(text)) throw new Error('Input contains an unpaired UTF-16 surrogate');
  return (text.match(/\s+|[^\s]+/gu) ?? []).map((piece) => Array.from(utf8.encode(piece)));
}

export type Pair = readonly [number, number];
export interface MergeRule {
  pair: Pair;
  id: number;
  count: number;
  replacements: number;
}

function replacePair(pieces: number[][], [left, right]: Pair, id: number) {
  let replacements = 0;
  const result = pieces.map((piece) => {
    const tokens: number[] = [];
    for (let i = 0; i < piece.length; i++) {
      if (piece[i] === left && piece[i + 1] === right) {
        tokens.push(id);
        i++;
        replacements++;
      } else {
        tokens.push(piece[i]);
      }
    }
    return tokens;
  });
  return { pieces: result, replacements };
}

export function trainBpe(corpus: string, limit: number) {
  const vocabulary = Array.from({ length: BYTE_COUNT }, (_, byte) => [byte]);
  const rules: MergeRule[] = [];
  let pieces = bytePieces(corpus);
  for (let round = 0; round < limit; round++) {
    const counts = new Map<string, { pair: Pair; count: number }>();
    for (const piece of pieces) {
      for (let i = 0; i + 1 < piece.length; i++) {
        const pair: Pair = [piece[i], piece[i + 1]];
        const key = pair.join(',');
        const current = counts.get(key);
        if (current) current.count++;
        else counts.set(key, { pair, count: 1 });
      }
    }
    // Deterministic tie break for this teaching tokenizer, by left then right ID.
    const best = [...counts.values()].sort(
      (a, b) => b.count - a.count || a.pair[0] - b.pair[0] || a.pair[1] - b.pair[1],
    )[0];
    if (!best) break;
    const id = vocabulary.length;
    const merged = replacePair(pieces, best.pair, id);
    pieces = merged.pieces;
    vocabulary.push([...vocabulary[best.pair[0]], ...vocabulary[best.pair[1]]]);
    rules.push({ ...best, id, replacements: merged.replacements });
  }
  return { vocabulary, rules };
}

export function encodeBpe(text: string, rules: MergeRule[]) {
  const bytes = bytePieces(text);
  let pieces = bytes;
  const rounds = rules.map((rule) => {
    const merged = replacePair(pieces, rule.pair, rule.id);
    pieces = merged.pieces;
    return { rule, pieces, replacements: merged.replacements };
  });
  return { bytes, rounds, ids: pieces.flat() };
}

export function decodeIds(ids: number[], vocabulary: number[][]): string {
  return decoder.decode(Uint8Array.from(ids.flatMap((id) => vocabulary[id])));
}

/** An individual token can end inside a Unicode character. Show exact bytes even in that case. */
export function tokenText(bytes: number[]): string | null {
  try {
    return decoder.decode(Uint8Array.from(bytes));
  } catch {
    return null;
  }
}

export function embeddingTable(size: number, seed: number): number[][] {
  let value = seed >>> 0;
  return Array.from({ length: size }, () =>
    Array.from({ length: EMBEDDING_DIMENSIONS }, () => {
      value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
      return (((value >>> 24) % 17) - 8) / 4;
    }),
  );
}

export function tokenize(text: string, corpus: CorpusId, merges: number, seed: number) {
  const model = trainBpe(CORPORA[corpus], merges);
  const encoded = encodeBpe(text, model.rules);
  const embedding = embeddingTable(model.vocabulary.length, seed);
  return {
    ...model,
    ...encoded,
    embedding,
    hidden: encoded.ids.map((id) => embedding[id]),
    decoded: decodeIds(encoded.ids, model.vocabulary),
  };
}
