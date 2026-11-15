import { describe, expect, it } from 'vitest';
import katex from 'katex';
import { CHAPTERS, CHAPTER_PARTS } from './chapters';
import { FORWARD_CONTENT } from './forward';
import { ARCHITECTURE_FAMILIES, ARCHITECTURE_SYMBOLS, MAMBA_NODES } from './postTransformer';
import { TOPICS } from './topics';

describe('章节覆盖与讲解边界', () => {
  it('32章和3附录均有实质导读，引用和先备关系没有遗漏或循环', () => {
    expect(CHAPTERS.map((c) => c.id)).toEqual([
      ...Array.from({ length: 32 }, (_, i) => String(i + 1).padStart(2, '0')),
      'A',
      'B',
      'C',
    ]);
    const visit = (id: string, path: string[] = []): void => {
      expect(path, `先备关系循环 ${id}`).not.toContain(id);
      const chapter = CHAPTERS.find((c) => c.id === id);
      expect(chapter).toBeDefined();
      chapter!.prerequisites.forEach((p) => visit(p, [...path, id]));
    };
    for (const c of CHAPTERS) {
      expect(c.question.length).toBeGreaterThan(5);
      expect(c.summary.length).toBeGreaterThan(15);
      expect(CHAPTER_PARTS.some((p) => p.id === c.part)).toBe(true);
      expect(c.bookUrl).toMatch(/^https:\/\/waylandz.com\//);
      expect(c.units.length).toBeGreaterThan(0);
      visit(c.id);
      for (const unit of c.units) {
        unit.topics.forEach((id) => expect(TOPICS.some((t) => t.id === id)).toBe(true));
        if (unit.status === 'interactive') expect(!!unit.href || unit.topics.length > 0).toBe(true);
      }
    }
  });
  it('保留明确缺口，不能用局部实验冒充完整模型实现', () => {
    expect(CHAPTERS.find((c) => c.id === '04')!.units.some((u) => u.status === 'reference')).toBe(
      true,
    );
    expect(CHAPTERS.find((c) => c.id === '17')!.units.some((u) => u.status === 'interactive')).toBe(
      true,
    );
    expect(CHAPTERS.find((c) => c.id === '27')!.units.some((u) => u.status === 'reference')).toBe(
      true,
    );
    expect(ARCHITECTURE_FAMILIES.map((f) => f.id)).toEqual(
      expect.arrayContaining(['s4', 'mamba', 'ssd', 'rwkv', 'retnet', 'hyena', 'hybrid']),
    );
    expect(ARCHITECTURE_FAMILIES.find((f) => f.id === 'ssd')!.boundary).toContain('不能推成');
  });
  it('新增公式与所有符号可严格排版', () => {
    for (const content of [...Object.values(FORWARD_CONTENT), ...MAMBA_NODES]) {
      expect(content.symbols.length).toBeGreaterThan(0);
      expect(() =>
        katex.renderToString(content.formula, { throwOnError: true, strict: 'ignore' }),
      ).not.toThrow();
      for (const [tex, meaning] of content.symbols) {
        expect(meaning).toBeTruthy();
        expect(() =>
          katex.renderToString(tex, { throwOnError: true, strict: 'ignore' }),
        ).not.toThrow();
      }
    }
    for (const [tex] of Object.values(ARCHITECTURE_SYMBOLS))
      expect(() =>
        katex.renderToString(tex, { throwOnError: true, strict: 'ignore' }),
      ).not.toThrow();
  });
});
