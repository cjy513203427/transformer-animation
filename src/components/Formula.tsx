import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import type { FormulaSymbol } from '../content/notation/shared';
import { T } from '../i18n';
import { Rich, Tex } from './Tex';
import '../styles/formula.css';

export function Formula({
  expression,
  symbols,
  showSymbols = true,
}: {
  expression: string;
  symbols: readonly FormulaSymbol[];
  showSymbols?: boolean;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(expression, { throwOnError: false, displayMode: true, trust: false }),
    [expression],
  );
  return (
    <div className="formula-block">
      <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />
      {showSymbols && (
        <div className="formula-symbols">
          <strong className="formula-symbols-title">{T('本式符号')}</strong>
          <dl aria-label={T('本式符号说明')}>
            {symbols.map(([tex, meaning]) => (
              <div className="formula-symbol" key={tex}>
                <dt>
                  <Tex tex={tex} />
                </dt>
                <dd>
                  <Rich text={meaning} />
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}
