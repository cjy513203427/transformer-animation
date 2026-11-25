import type { CodeHighlight, LinkedCodeSpec } from '../content/codeViews';
import { T } from '../i18n';
import { Tx } from '../i18n/jsx';
import '../styles/code.css';

function formatLineRange(lines: number[]): string {
  if (!lines.length) return '';
  const sorted = [...new Set(lines)].sort((a, b) => a - b);
  const ranges: string[] = [];
  let start = sorted[0];
  let end = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] === end + 1) {
      end = sorted[i];
    } else {
      ranges.push(start === end ? String(start) : `${start}–${end}`);
      start = end = sorted[i];
    }
  }
  ranges.push(start === end ? String(start) : `${start}–${end}`);
  return ranges.join(', ');
}

export function LinkedCode({
  spec,
  highlight,
}: {
  spec: LinkedCodeSpec;
  highlight: CodeHighlight;
}) {
  const active = new Set(highlight.lines);
  const range = formatLineRange(highlight.lines);
  return (
    <details className="lesson-formula linked-code" data-testid={`linked-code-${spec.file}`}>
      <summary>
        {Tx('联动 PyTorch 代码 · {#0}（高亮第 {#1} 行）', spec.file, range)}
      </summary>
      <div className="linked-code-body">
        <p className="linked-code-note">{highlight.note}</p>
        <div className="linked-code-scroll" tabIndex={0} aria-label={T('{#0} 联动代码', spec.file)}>
          <pre className="linked-code-pre">
            <code>
              {spec.lines.map((line, i) => {
                const number = i + 1;
                const isActive = active.has(number);
                return (
                  <span
                    key={number}
                    className={isActive ? 'code-line code-line-active' : 'code-line'}
                    aria-current={isActive ? 'step' : undefined}
                  >
                    <span className="code-gutter" aria-hidden="true">
                      {isActive ? '→' : ' '}
                      {String(number).padStart(2, ' ')}
                    </span>
                    <span className="code-text">{line}</span>
                  </span>
                );
              })}
            </code>
          </pre>
        </div>
        <p className="linked-code-meta">
          <span>{spec.scopeNote}</span>{' '}
          <a href={spec.chapterHref} target="_blank" rel="noreferrer">
            {spec.chapterLabel} ↗
          </a>
        </p>
      </div>
    </details>
  );
}
