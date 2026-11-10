import { afterEach, describe, expect, it } from 'vitest';
import { isValidElement } from 'react';
import { CHAPTER_PATHS } from '../content/book';
import { checkCatalog, extractMessages, readCatalog, TRANSLATED, type Catalog } from './extract';
import { localePath, setLocale, splitLocale, T } from '.';
import { Tx } from './jsx';

afterEach(() => setLocale('zh'));

describe('locale paths', () => {
  it('reads the language prefix and keeps the experiment path', () => {
    expect(splitLocale('/moe')).toEqual({ locale: 'zh', path: '/moe' });
    expect(splitLocale('/en/moe')).toEqual({ locale: 'en', path: '/moe' });
    expect(splitLocale('/de/')).toEqual({ locale: 'de', path: '/' });
    expect(splitLocale('/de')).toEqual({ locale: 'de', path: '/' });
    // Only a whole first segment counts as a language.
    expect(splitLocale('/english/moe')).toEqual({ locale: 'zh', path: '/english/moe' });
    expect(splitLocale('/fr/moe')).toEqual({ locale: 'zh', path: '/fr/moe' });
  });

  it('builds paths that round-trip through splitLocale', () => {
    for (const locale of ['zh', 'en', 'de'] as const)
      for (const path of ['/', '/attention', '/ssm-mamba']) {
        expect(splitLocale(localePath(path, locale))).toEqual({ locale, path });
      }
  });
});

describe('T and Tx', () => {
  it('fills placeholders in the translation, in its own order', () => {
    setLocale('en', { '第 {#0} 层第 {#1} 头': 'head {#1} of layer {#0}' });
    expect(T('第 {#0} 层第 {#1} 头', 2, 5)).toBe('head 5 of layer 2');
  });

  it('falls back to the Chinese source when a translation is missing', () => {
    setLocale('de', {});
    const warn = console.warn;
    console.warn = () => {};
    expect(T('只有中文 {#0}', 1)).toBe('只有中文 1');
    console.warn = warn;
  });

  it('keeps TeX braces that are not placeholders', () => {
    setLocale('zh');
    expect(T('$x_{0}$ 与 {#0}', 'y')).toBe('$x_{0}$ 与 y');
  });

  it('places React nodes where the translation puts them', () => {
    setLocale('en', { '先看 {#0} 再看 {#1}': 'look at {#1} before {#0}' });
    const parts = Tx('先看 {#0} 再看 {#1}', 'A', 'B') as unknown[];
    const text = parts
      .map((part) => (isValidElement(part) ? (part.props as { children: string }).children : part))
      .join('');
    expect(text).toBe('look at B before A');
  });
});

describe('translation catalogs', () => {
  const root = process.cwd();
  const { messages, problems } = extractMessages(root);
  const catalogs = Object.fromEntries(
    TRANSLATED.map((locale) => [locale, readCatalog(root, locale)]),
  ) as Record<(typeof TRANSLATED)[number], Catalog>;

  it('finds every message as a literal T(…) or Tx(…) call', () => {
    expect(messages.length).toBeGreaterThan(2000);
    expect(problems).toEqual([]);
  });

  for (const locale of TRANSLATED)
    it(`${locale} translates every message, with matching placeholders and no Chinese left`, () => {
      expect(checkCatalog(locale, messages, catalogs[locale])).toEqual([]);
    });

  it('English and German agree on leading and trailing spaces', () => {
    const differ = Object.keys(catalogs.en).filter((zh) => {
      const [en, de] = [catalogs.en[zh], catalogs.de[zh]];
      return /^\s/.test(en) !== /^\s/.test(de) || /\s$/.test(en) !== /\s$/.test(de);
    });
    expect(differ).toEqual([]);
  });

  it('maps every chapter of the Chinese book to the English edition', async () => {
    setLocale('en');
    const { chapterUrl } = await import('../content/book');
    for (const path of CHAPTER_PATHS)
      expect(chapterUrl(path)).toMatch(
        /^https:\/\/waylandz\.com\/llm-transformer-book-en\/[a-z0-9-]+\/$/,
      );
  });
});
