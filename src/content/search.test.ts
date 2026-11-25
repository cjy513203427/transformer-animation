import { describe, expect, it } from 'vitest';
import {
  damerauLevenshtein,
  fuzzyMatchesText,
  searchSite,
  splitHighlightSegments,
} from './search';

describe('全站模糊搜索引擎', () => {
  it('Damerau-Levenshtein 正确处理插入、删除、替换和相邻字符颠倒', () => {
    expect(damerauLevenshtein('attention', 'attention')).toBe(0);
    expect(damerauLevenshtein('attnetion', 'attention')).toBe(1);
    expect(damerauLevenshtein('qunatize', 'quantize')).toBe(1);
    expect(damerauLevenshtein('trasformer', 'transformer')).toBe(1);
    expect(damerauLevenshtein('abc', 'xyz', 2)).toBeGreaterThan(2);
  });

  it('支持精确匹配、中英文缩写、有序子序列与拼写容错模糊检索', () => {
    // Typo tolerance: "attnetion" -> Attention lab at top
    const typoAttn = searchSite('attnetion', 'experiment');
    expect(typoAttn.length).toBeGreaterThan(0);
    expect(typoAttn[0].item.href).toContain('/attention');

    // Ordered subsequence: "flsh" -> FlashAttention
    const subFlash = searchSite('flsh', 'experiment');
    expect(subFlash.some((h) => h.item.href.includes('/flash-attention'))).toBe(true);

    // Chinese subsequence: "线变" -> 线性变换与几何
    const cnSub = searchSite('线变', 'experiment');
    expect(cnSub.length).toBeGreaterThan(0);
    expect(cnSub[0].item.href).toContain('/linear-transform');

    // Bilingual acronym & multi-term AND query: "gptq 量化" -> quantization
    const multi = searchSite('gptq 量化', 'experiment');
    expect(multi.length).toBeGreaterThan(0);
    expect(multi[0].item.href).toContain('/quantization');

    // Chapter lookup by number / file name: "model.py" -> Ch 18
    const ch18 = searchSite('model.py', 'chapter');
    expect(ch18.some((h) => h.item.id === 'chapter-18')).toBe(true);

    // Typo in English acronym/word: "quantizatoin" -> quantization
    const typoQuant = searchSite('quantizatoin', 'experiment');
    expect(typoQuant.some((h) => h.item.href.includes('/quantization'))).toBe(true);
  });

  it('分类筛选和空查询返回有效结果，高亮分段完整保留原文', () => {
    const notesOnly = searchSite('dpo', 'note');
    expect(notesOnly.length).toBeGreaterThan(0);
    expect(notesOnly.every((h) => h.item.kind === 'note')).toBe(true);

    const chaptersOnly = searchSite('15', 'chapter');
    expect(chaptersOnly.some((h) => h.item.id === 'chapter-15')).toBe(true);

    expect(fuzzyMatchesText('flsh', 'FlashAttention')).toBe(true);
    expect(fuzzyMatchesText('attnetion', 'Multi-Head Attention')).toBe(true);
    expect(fuzzyMatchesText('zzzzzz', 'FlashAttention')).toBe(false);

    const segments = splitHighlightSegments('FlashAttention', [
      [0, 5],
      [5, 9],
    ]);
    expect(segments.map((s) => s.text).join('')).toBe('FlashAttention');
    expect(segments[0]).toEqual({ text: 'FlashAtte', hit: true });
    expect(segments[1]).toEqual({ text: 'ntion', hit: false });
  });
});
