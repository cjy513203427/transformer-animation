/**
 * Interface language.
 *
 * Chinese is the source language and lives at the site root (`/attention`); English and German
 * pages live under `/en/…` and `/de/…`. Source code keeps the Chinese text inline inside `T(…)`
 * or `Tx(…)` (see ./jsx), and `catalog/<locale>.json` maps each exact Chinese string to its
 * translation. Switching language is a full navigation, so the locale is fixed for a page's
 * lifetime and page modules may translate at import time. See docs/i18n.md.
 */
export type Locale = 'zh' | 'en' | 'de';
export const LOCALES: readonly Locale[] = ['zh', 'en', 'de'];
/** Each language names itself in the switcher. */
export const LOCALE_LABEL: Record<Locale, string> = { zh: '中文', en: 'English', de: 'Deutsch' };
export const HTML_LANG: Record<Locale, string> = { zh: 'zh-CN', en: 'en', de: 'de' };

/** Placeholder for the i-th argument of a message: `{#0}`, `{#1}` … (plain braces clash with TeX). */
export const PLACEHOLDER = /\{#(\d+)\}/g;

/** `/de/moe` → { locale: 'de', path: '/moe' }; `/moe` → { locale: 'zh', path: '/moe' }. */
export function splitLocale(pathname: string): { locale: Locale; path: string } {
  const match = /^\/(en|de)(?=\/|$)/.exec(pathname);
  const rest = match ? pathname.slice(match[0].length) : pathname;
  return {
    locale: (match?.[1] as Locale | undefined) ?? 'zh',
    path: rest.replace(/\/+$/, '') || '/',
  };
}

/** The page's own language; set once by {@link initLocale} before any page module loads. */
export let LOCALE: Locale = 'zh';
let catalog: Readonly<Record<string, string>> = {};
const reported = new Set<string>();

/** Site path in a language: `localePath('/moe', 'en')` → `/en/moe`. */
export function localePath(path: string, locale: Locale = LOCALE): string {
  if (locale === 'zh') return path;
  return path === '/' ? `/${locale}/` : `/${locale}${path}`;
}

const catalogs: Record<
  Exclude<Locale, 'zh'>,
  () => Promise<{ default: Record<string, string> }>
> = {
  en: () => import('./catalog/en.json'),
  de: () => import('./catalog/de.json'),
};

/** Reads the language from the URL and loads its catalog. Chinese pages load no catalog. */
export async function initLocale(pathname = window.location.pathname): Promise<Locale> {
  const { locale } = splitLocale(pathname);
  setLocale(locale, locale === 'zh' ? {} : (await catalogs[locale]()).default);
  document.documentElement.lang = HTML_LANG[locale];
  return locale;
}

/** Sets the language directly; tests use it before importing page modules. */
export function setLocale(locale: Locale, entries: Readonly<Record<string, string>> = {}) {
  LOCALE = locale;
  catalog = entries;
}

/** The current-language text of a Chinese source string, placeholders still in place. */
export function translate(zh: string): string {
  if (LOCALE === 'zh') return zh;
  const text = catalog[zh];
  if (text !== undefined) return text;
  if (!reported.has(zh)) {
    reported.add(zh);
    console.warn(`[i18n] missing ${LOCALE} translation:`, zh);
  }
  return zh;
}

/**
 * Translates a Chinese source string and fills `{#i}` with `args[i]`.
 * `T('第 {#0} 步', n)` → “Step 3” on an English page.
 */
export function T(zh: string, ...args: unknown[]): string {
  const text = translate(zh);
  return args.length ? text.replace(PLACEHOLDER, (_, i: string) => String(args[Number(i)])) : text;
}
