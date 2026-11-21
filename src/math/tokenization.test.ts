import { describe, expect, it } from 'vitest';
import {
  bytePieces,
  CORPORA,
  decodeIds,
  encodeBpe,
  tokenize,
  tokenText,
  trainBpe,
} from './tokenization';

describe('byte-level BPE', () => {
  it('counts overlapping adjacent pairs and replaces only left-to-right non-overlapping pairs', () => {
    const { rules, vocabulary } = trainBpe('aaa aaa a', 1);
    expect(rules).toEqual([{ pair: [97, 97], count: 4, replacements: 2, id: 256 }]);
    expect(vocabulary[256]).toEqual([97, 97]);
    expect(encodeBpe('aaaaa', rules).ids).toEqual([256, 256, 97]);
  });

  it('uses training ranks at encoding time and resolves training ties by pair ID', () => {
    const { rules } = trainBpe('ab ab ab bc bc', 2);
    expect(rules.map((rule) => rule.pair)).toEqual([
      [97, 98],
      [98, 99],
    ]);
    expect(encodeBpe('abc', rules).ids).toEqual([256, 99]);
    expect(trainBpe('ba ab', 1).rules[0].pair).toEqual([97, 98]);
  });

  it('round trips multibyte characters, combining marks, BOM and whitespace at every stage', () => {
    expect(bytePieces('猫')).toEqual([[0xe7, 0x8c, 0xab]]);
    expect(tokenText([0xe7])).toBeNull();
    for (const corpus of Object.keys(CORPORA) as (keyof typeof CORPORA)[]) {
      for (const input of ['\uFEFFabab 猫猫  🐈\ne\u0301\r\n', 'aaa aa', '', '  ']) {
        const result = tokenize(input, corpus, 8, 42);
        expect(result.decoded).toBe(input);
        let length = result.bytes.flat().length;
        for (const round of result.rounds) {
          const ids = round.pieces.flat();
          expect(decodeIds(ids, result.vocabulary)).toBe(input);
          expect(ids.length).toBe(length - round.replacements);
          length = ids.length;
        }
      }
    }
  });

  it('keeps segment boundaries and stops cleanly when no pairs remain', () => {
    expect(trainBpe('a b', 8).rules).toEqual([]);
    expect(encodeBpe('ab ab', trainBpe('ab ab', 8).rules).ids).toEqual([256, 32, 256]);
    expect(() => bytePieces('\uD800')).toThrow();
  });

  it('looks up identical rows for repeated IDs and changes only the table when reseeded', () => {
    const first = tokenize('ab ab', 'pairs', 3, 42);
    const second = tokenize('ab ab', 'pairs', 3, 101);
    expect(first.ids).toEqual([256, 32, 256]);
    expect(first.hidden[0]).toEqual(first.hidden[2]);
    expect(first.hidden[0]).toEqual(first.embedding[256]);
    expect(first.hidden[0]).not.toEqual(second.hidden[0]);
    expect(first.ids).toEqual(second.ids);
    expect(first.rules).toEqual(second.rules);
  });
});
