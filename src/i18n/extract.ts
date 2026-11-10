/**
 * Finds every translatable message in the source and checks the catalogs against them.
 * Used by `scripts/i18n.mjs` and `catalog.test.ts`; never bundled into the site.
 * Node-only and self-contained so that Node can run it with type stripping.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseAst } from 'rolldown/parseAst';

export interface Message {
  /** Exact Chinese source text, placeholders `{#i}` included. */
  zh: string;
  file: string;
  line: number;
  fn: 'T' | 'Tx';
  /** Source of each placeholder argument, for translators. */
  args: string[];
}
export type Catalog = Record<string, string>;
export type TranslatedLocale = 'en' | 'de';
export const TRANSLATED: TranslatedLocale[] = ['en', 'de'];

/** Chinese characters, CJK punctuation and full-width forms. */
export const CJK = /[　-〿一-鿿＀-￯]/;
const PLACEHOLDER = /\{#(\d+)\}/g;

interface Node {
  type: string;
  start: number;
  end: number;
  [key: string]: unknown;
}

function sourceFiles(dir: string): string[] {
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .flatMap((entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return sourceFiles(full);
      return /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [full] : [];
    })
    .sort();
}

/** All `T('…')` and `Tx('…')` calls under `src`, in file order, plus calls that cannot be checked. */
export function extractMessages(root: string): { messages: Message[]; problems: string[] } {
  const messages: Message[] = [];
  const problems: string[] = [];
  const src = path.join(root, 'src');
  for (const file of sourceFiles(src)) {
    const rel = path.relative(root, file).replace(/\\/g, '/');
    if (rel.startsWith('src/i18n/')) continue;
    const code = fs.readFileSync(file, 'utf8');
    const ast = parseAst(code, { lang: file.endsWith('x') ? 'tsx' : 'ts' }) as unknown as Node;
    const lineOf = (offset: number) => code.slice(0, offset).split('\n').length;
    const visit = (value: unknown): void => {
      if (!value || typeof value !== 'object') return;
      if (Array.isArray(value)) return value.forEach(visit);
      const node = value as Node;
      if (node.type === 'CallExpression') {
        const callee = node.callee as Node & { name?: string };
        if (callee.type === 'Identifier' && (callee.name === 'T' || callee.name === 'Tx')) {
          const [first, ...rest] = node.arguments as (Node & { value?: unknown })[];
          if (first?.type === 'Literal' && typeof first.value === 'string') {
            messages.push({
              zh: first.value,
              file: rel,
              line: lineOf(node.start),
              fn: callee.name,
              args: rest.map((arg) => code.slice(arg.start, arg.end).replace(/\s+/g, ' ')),
            });
          } else {
            problems.push(`${rel}:${lineOf(node.start)} ${callee.name}() needs a string literal`);
          }
        }
      }
      for (const [key, child] of Object.entries(node)) if (key !== 'parent') visit(child);
    };
    visit(ast);
  }
  return { messages, problems };
}

/** Short stable handle for a message, used when merging translations. */
export function messageHash(zh: string): string {
  // cyrb53
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < zh.length; i++) {
    const ch = zh.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36).padStart(11, '0');
}

const placeholders = (text: string) =>
  [...text.matchAll(PLACEHOLDER)].map((match) => match[1]).sort();
const count = (text: string, char: string) => text.split(char).length - 1;

/** Everything wrong with one catalog, as readable lines. Empty means complete. */
export function checkCatalog(
  locale: TranslatedLocale,
  messages: Message[],
  catalog: Catalog,
): string[] {
  const problems: string[] = [];
  const used = new Set(messages.map((message) => message.zh));
  const seen = new Set<string>();
  for (const message of messages) {
    if (seen.has(message.zh)) continue;
    seen.add(message.zh);
    const where = `${locale} ${message.file}:${message.line}`;
    const text = catalog[message.zh];
    if (text === undefined) {
      problems.push(`${where} missing: ${message.zh}`);
      continue;
    }
    if (!text.trim() && message.zh.trim()) problems.push(`${where} empty: ${message.zh}`);
    if (CJK.test(text)) problems.push(`${where} contains Chinese: ${text}`);
    if (placeholders(text).join() !== placeholders(message.zh).join())
      problems.push(`${where} placeholders differ: ${message.zh} → ${text}`);
    if (count(text, '$') !== count(message.zh, '$'))
      problems.push(`${where} TeX $ count differs: ${message.zh} → ${text}`);
    // A space at either end of a Chinese fragment separates it from a neighbouring node; keep it.
    if (
      (/^\s/.test(message.zh) && !/^\s/.test(text)) ||
      (/\s$/.test(message.zh) && !/\s$/.test(text))
    )
      problems.push(`${where} edge whitespace lost: “${message.zh}” → “${text}”`);
  }
  for (const key of Object.keys(catalog))
    if (!used.has(key)) problems.push(`${locale} unused: ${key}`);
  return problems;
}

export function catalogPath(root: string, locale: TranslatedLocale) {
  return path.join(root, 'src', 'i18n', 'catalog', `${locale}.json`);
}

export function readCatalog(root: string, locale: TranslatedLocale): Catalog {
  return JSON.parse(fs.readFileSync(catalogPath(root, locale), 'utf8')) as Catalog;
}

/** Writes a catalog in source order so related messages stay together; drops unused entries. */
export function writeCatalog(
  root: string,
  locale: TranslatedLocale,
  messages: Message[],
  catalog: Catalog,
) {
  const ordered: Catalog = {};
  for (const message of messages)
    if (catalog[message.zh] !== undefined) ordered[message.zh] = catalog[message.zh];
  fs.writeFileSync(catalogPath(root, locale), JSON.stringify(ordered, null, 2) + '\n');
}
