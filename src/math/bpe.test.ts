import { describe, expect, it } from 'vitest';
import { END, encodeWord, learnBPE, utf8Bytes } from './bpe';
import { BPE_PRESETS, initialBPEState, restoreBPEState, serializeBPEState } from '../state/bpe';

describe('字符级 BPE', () => {
  const r = learnBPE(BPE_PRESETS.sennrich, 12);
  it('Sennrich 玩具词典的前五次合并与手算一致（平局取字典序最小）', () => {
    expect(r.merges.slice(0, 5).map((m) => m.token)).toEqual(['es', 'est', 'est' + END, 'lo', 'low']);
    expect(r.merges.slice(0, 5).map((m) => m.count)).toEqual([9, 9, 9, 7, 7]);
    // 4·5 + 6·2 + 7·6 + 7·3 symbols before any merge.
    expect(r.snapshots[0].totalTokens).toBe(95);
  });

  it('每次合并词表加 1；不重叠时 token 总数恰好减少该对的次数', () => {
    r.merges.forEach((m, k) => {
      expect(r.snapshots[k + 1].vocab.length).toBe(r.snapshots[k].vocab.length + 1);
      expect(r.snapshots[k].totalTokens - r.snapshots[k + 1].totalTokens).toBe(m.count);
    });
  });

  it('重叠的相同字符对计数 2 次，但从左到右只能合并 1 次', () => {
    const overlap = learnBPE([{ word: 'aaa', count: 1 }], 1);
    expect(overlap.merges[0]).toMatchObject({ pair: ['a', 'a'], count: 2 });
    expect(overlap.snapshots[0].totalTokens - overlap.snapshots[1].totalTokens).toBe(1);
    expect(overlap.snapshots[1].words[0]).toEqual(['aa', 'a', END]);
  });

  it('新词按学到的顺序套用合并；没见过的字母留在词表之外', () => {
    expect(encodeWord('lowest', r.merges, 5)).toEqual(['low', 'est' + END]);
    expect(encodeWord('lowest', r.merges, 0)).toEqual(['l', 'o', 'w', 'e', 's', 't', END]);
    expect(encodeWord('zoo', r.merges, 12)).toContain('z');
    expect(r.snapshots[12].vocab).not.toContain('z');
  });

  it('没有相邻对时停止，不会凭空合并', () => {
    const single = learnBPE([{ word: 'a', count: 3 }], 5);
    // "a </w>" can merge once into "a</w>", then nothing is left.
    expect(single.merges.map((m) => m.token)).toEqual(['a' + END]);
    expect(single.snapshots).toHaveLength(2);
  });

  it('UTF-8 字节：ASCII 一个字节，常用汉字三个字节', () => {
    expect(utf8Bytes('a')).toEqual([97]);
    expect(utf8Bytes('中')).toEqual([0xe4, 0xb8, 0xad]);
  });
});

describe('BPE 分享状态', () => {
  it('往返并拒绝非法单词、计数、步数与重复词', () => {
    const s = initialBPEState();
    s.step = 4;
    s.probe = 'newer';
    expect(restoreBPEState(serializeBPEState(s))).toEqual({ state: s, invalid: false });
    const bad = (patch: Record<string, unknown>) =>
      restoreBPEState('#bpe=' + encodeURIComponent(JSON.stringify({ ...initialBPEState(), ...patch })))
        .invalid;
    expect(bad({ corpus: [{ word: 'Low', count: 1 }] })).toBe(true);
    expect(bad({ corpus: [{ word: 'low', count: 0 }] })).toBe(true);
    expect(bad({ corpus: [{ word: 'low', count: 1 }, { word: 'low', count: 2 }] })).toBe(true);
    expect(bad({ step: 13 })).toBe(true);
    expect(bad({ probe: '中文' })).toBe(true);
    expect(bad({ version: 2 })).toBe(true);
    expect(restoreBPEState('#bpe=%E0').invalid).toBe(true);
  });
});
