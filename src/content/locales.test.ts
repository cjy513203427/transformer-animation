/**
 * Page modules translate when they are imported, so each language reloads them after
 * switching the locale. These tests run every P1 explanation with real numbers – metric lookups
 * by translated label must still match – and scan every exported text for leftover Chinese.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TopicDefinition } from './p1/types';

const CJK = /[　-〿一-鿿＀-￯]/;
/** Chinese chapter paths are lookup keys that chapterUrl() turns into English-edition URLs. */
const CHAPTER_PATH = /^(第\d\d章|附录[A-C]|前言)[^/]*\/$/;

/** Every string reachable from a value, with its path; functions are skipped. */
function strings(value: unknown, path = '', seen = new Set<unknown>()): [string, string][] {
  if (typeof value === 'string') return [[path, value]];
  if (!value || typeof value !== 'object' || seen.has(value)) return [];
  seen.add(value);
  return Object.entries(value).flatMap(([key, child]) =>
    typeof child === 'function' ? [] : strings(child, `${path}.${key}`, seen),
  );
}
/** Paths and texts that still contain Chinese, as one readable line each. */
const chinese = (pairs: [string, string][]) =>
  pairs
    .filter(([, text]) => CJK.test(text) && !CHAPTER_PATH.test(text))
    .map(([path, text]) => `${path}: ${text}`);

const P1 = [
  'normalization',
  'ffn',
  'lora',
  'quantization',
  'decoding',
  'preference',
  'memory',
  'ssm',
  'optimizer',
  'context',
];
/** Imported fresh after each locale switch; explicit so the bundler can resolve them. */
const P0_MODULES = {
  attention: () => import('./attention'),
  position: () => import('./position'),
  inference: () => import('./inference'),
  kv: () => import('./kv'),
  moe: () => import('./moe'),
  flash: () => import('./flash'),
  topics: () => import('./topics'),
  book: () => import('./book'),
  notebook: () => import('./notebook'),
};

for (const locale of ['en', 'de'] as const)
  describe(`${locale} pages`, () => {
    beforeEach(async () => {
      vi.resetModules();
      const i18n = await import('../i18n');
      const catalog = (await import(`../i18n/catalog/${locale}.json`)).default;
      i18n.setLocale(locale, catalog);
    });

    it('P1: every preset and step explains with real numbers and no Chinese', async () => {
      const { explain, lessonContext } = await import('./p1/types');
      const { initialLabState } = await import('../state/p1');
      const topics: TopicDefinition[] = [
        ...await Promise.all(P1.map(async name => (await import(`./p1/${name}.ts`)).default)),
        ...Object.values(await import('./p1/learning')),
      ];
      for (const topic of topics) {
        expect(chinese(strings(topic, topic.id))).toEqual([]);
        const base = initialLabState(topic);
        for (const example of [{ input: topic.input, parameters: {} }, ...topic.examples]) {
          const config = { ...base.config, ...example.parameters };
          const result = topic.compute(example.input, config, base.seed);
          expect(chinese(strings(result, topic.id))).toEqual([]);
          topic.steps.forEach((step, i) => {
            const text = explain(
              step.explanation,
              lessonContext(example.input, config, result.frames[i]),
            );
            expect(text, `${topic.id}/${step.name}`).not.toMatch(CJK);
            expect(text).not.toMatch(/NaN|undefined/);
          });
        }
      }
    });

    it('P0 content and state labels contain no Chinese', async () => {
      for (const [name, load] of Object.entries(P0_MODULES))
        expect(chinese(strings(await load(), name))).toEqual([]);
      for (const name of ['experiment', 'position', 'inference', 'kv', 'moe', 'flash']) {
        const module = await import(`../state/${name}.ts`);
        expect(chinese(strings(module, `state/${name}`))).toEqual([]);
      }
    });
  });
