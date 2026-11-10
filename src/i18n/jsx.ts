import { Fragment, createElement, type ReactNode } from 'react';
import { PLACEHOLDER, translate } from '.';

/**
 * Translates a sentence that contains React nodes: `Tx('先看 {#0} 的形状', <Tex tex="Q" />)`.
 * A translation may reorder the placeholders, so formulas and emphasis stay inside the sentence.
 */
export function Tx(zh: string, ...parts: ReactNode[]): ReactNode {
  const text = translate(zh);
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(PLACEHOLDER)) {
    if (match.index > last) out.push(text.slice(last, match.index));
    out.push(createElement(Fragment, { key: out.length }, parts[Number(match[1])]));
    last = match.index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
