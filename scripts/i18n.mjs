#!/usr/bin/env node
// Translation catalog helper. Run from the project root:
//   node scripts/i18n.mjs check                 list every catalog problem (exit 1 if any)
//   node scripts/i18n.mjs todo [path-filter]    untranslated messages with their hash and context
//   node scripts/i18n.mjs merge <file.tsv>      add translations; lines are  hash<TAB>en<TAB>de[<TAB>|]
// See docs/i18n.md.
import fs from 'node:fs';
import {
  TRANSLATED,
  checkCatalog,
  extractMessages,
  messageHash,
  readCatalog,
  writeCatalog,
} from '../src/i18n/extract.ts';

const root = process.cwd();
const [command, arg] = process.argv.slice(2);
const { messages, problems } = extractMessages(root);
const catalogs = Object.fromEntries(
  TRANSLATED.map((locale) => [locale, readCatalog(root, locale)]),
);

if (command === 'check') {
  const all = [...problems, ...TRANSLATED.flatMap((l) => checkCatalog(l, messages, catalogs[l]))];
  console.log(
    all.length ? all.join('\n') : `OK: ${new Set(messages.map((m) => m.zh)).size} messages`,
  );
  process.exit(all.length ? 1 : 0);
} else if (command === 'todo') {
  const seen = new Map();
  for (const m of messages) {
    if (arg && !m.file.includes(arg)) continue;
    if (TRANSLATED.every((l) => catalogs[l][m.zh] !== undefined)) continue;
    const entry = seen.get(m.zh);
    if (entry) entry.uses++;
    else seen.set(m.zh, { ...m, uses: 1 });
  }
  for (const m of seen.values()) {
    const args = m.args.length
      ? `  [${m.args.map((a, i) => `#${i}=${a.slice(0, 60)}`).join(' | ')}]`
      : '';
    console.log(
      `${messageHash(m.zh)} ${m.file.replace(/^src\//, '')}:${m.line}${m.uses > 1 ? ` ×${m.uses}` : ''}${args}\n  ${m.zh}`,
    );
  }
  console.error(`${seen.size} untranslated`);
} else if (command === 'merge') {
  const byHash = new Map(messages.map((m) => [messageHash(m.zh), m.zh]));
  let added = 0;
  const unknown = [];
  for (const line of fs.readFileSync(arg, 'utf8').split(/\r?\n/)) {
    if (!line.trim() || line.startsWith('#')) continue;
    // An optional trailing “\t|” protects a final translation that ends in a space.
    // “hash:开头” also checks that the Chinese source starts with the given text.
    const [handle, ...texts] = line.replace(/\t\|$/, '').split('\t');
    const [hash, prefix] = handle.trim().split(':');
    const zh = byHash.get(hash);
    if (!zh || (prefix && !zh.startsWith(prefix))) {
      unknown.push(handle);
      continue;
    }
    TRANSLATED.forEach((locale, i) => {
      if (texts[i] !== undefined && texts[i] !== '') catalogs[locale][zh] = texts[i];
    });
    added++;
  }
  for (const locale of TRANSLATED) writeCatalog(root, locale, messages, catalogs[locale]);
  console.log(`merged ${added}${unknown.length ? `; unknown hashes: ${unknown.join(' ')}` : ''}`);
  process.exit(unknown.length ? 1 : 0);
} else {
  console.log('usage: node scripts/i18n.mjs check | todo [filter] | merge <file.tsv>');
  process.exit(1);
}
