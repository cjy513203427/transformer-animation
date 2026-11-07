import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { steps } from './attention';
import { inferenceMyths, inferenceSteps } from './inference';
import { kvMyths, kvSteps } from './kv';
import { moeMyths, moeSteps } from './moe';
import { bookNotes, positionSteps } from './position';

// Prose strings mark inline math as $...$. In an ordinary JS string "\sqrt" silently becomes
// "sqrt", which KaTeX happily renders as italic letters, so check for that explicitly.
const BARE_CONTROL_WORD =
  /(^|[^\\a-zA-Z])(sqrt|frac|mathrm|theta|omega|times|cdot|operatorname|sum|ldots)(?![a-zA-Z])/;

function mathIn(text: string) {
  return [...text.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]);
}

const proseStrings = [
  ...steps.flatMap((s) => [s.explanation, s.caution, s.shape, s.question]),
  ...positionSteps.flatMap((s) => [s.takeaway, s.prompt, s.title]),
  ...bookNotes,
  ...inferenceSteps.flatMap((s) => [s.explanation, s.takeaway, s.prompt, s.title]),
  ...inferenceMyths,
  ...kvSteps.flatMap((s) => [s.explanation, s.takeaway, s.prompt, s.title]),
  ...kvMyths,
  ...moeSteps.flatMap((s) => [s.explanation, s.takeaway, s.prompt, s.title]),
  ...moeMyths,
];
const displayFormulas = [
  ...steps.map((s) => s.formula),
  ...positionSteps.map((s) => s.formula),
  ...inferenceSteps.map((s) => s.formula),
  ...kvSteps.map((s) => s.formula),
  ...moeSteps.map((s) => s.formula),
];

describe('公式标记', () => {
  it('正文里的 $…$ 是合法 TeX，且没有因 JS 转义而丢掉反斜杠', () => {
    const fragments = proseStrings.flatMap(mathIn);
    expect(fragments.length).toBeGreaterThan(10);
    for (const tex of fragments) {
      expect(
        () => katex.renderToString(tex, { throwOnError: true, strict: 'ignore' }),
        tex,
      ).not.toThrow();
      expect(tex, `缺少反斜杠：${tex}`).not.toMatch(BARE_CONTROL_WORD);
    }
  });

  it('每一步的展示公式都能被 KaTeX 严格解析', () => {
    for (const tex of displayFormulas) {
      expect(
        () =>
          katex.renderToString(tex, { throwOnError: true, displayMode: true, strict: 'ignore' }),
        tex,
      ).not.toThrow();
    }
  });
});
