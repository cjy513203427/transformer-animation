import { describe, expect, it } from 'vitest';
import katex from 'katex';
import { FORWARD_CONTENT } from './forward';
import { ARCHITECTURE_FAMILIES, ARCHITECTURE_SYMBOLS, MAMBA_NODES } from './postTransformer';

describe('完整前向与后 Transformer 架构内容核对', () => {
  it('架构全景覆盖各路线并保留范围边界', () => {
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
