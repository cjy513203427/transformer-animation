import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export function Formula({ expression }: { expression: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(expression, { throwOnError: false, displayMode: true, trust: false }),
    [expression],
  );
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}
