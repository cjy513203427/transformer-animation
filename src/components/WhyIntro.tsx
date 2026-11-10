import { Rich } from './Tex';
import type { BookLink } from '../content/book';
import { T } from '../i18n';
import { Tx } from '../i18n/jsx';
import '../styles/why-intro.css';

export interface WhyContent {
  /** The motivating question, e.g. “为什么需要 FlashAttention？” */
  question: string;
  /** Short paragraphs; inline TeX in $…$. */
  points: string[];
  /** The book's own everyday comparison, labelled as an analogy. */
  analogy?: { title: string; text: string };
  book: BookLink;
}

/** The book opens every chapter with the problem first; each experiment does the same. */
export function WhyIntro({ why }: { why: WhyContent }) {
  return (
    <details className="why-intro" open>
      <summary>
        <span className="why-kicker">{T('先说为什么')}</span>
        <strong>{why.question}</strong>
        {/* A flex summary loses the native disclosure triangle; say what a click does instead. */}
        <span className="why-toggle" aria-hidden="true">
          <span className="why-toggle-open">{T('收起 ▴')}</span>
          <span className="why-toggle-closed">{T('展开 ▾')}</span>
        </span>
      </summary>
      <div className="why-body">
        <div className="why-text">
          {why.points.map((point) => (
            <p key={point}>
              <Rich text={point} />
            </p>
          ))}
          <a className="lesson-source" href={why.book[1]} target="_blank" rel="noreferrer">
            {Tx('对应原书：{#0} ↗', why.book[0])}
          </a>
        </div>
        {why.analogy && (
          <aside className="why-analogy" aria-label={T('原书类比')}>
            <span>{Tx('原书类比 · {#0}', why.analogy.title)}</span>
            <p>
              <Rich text={why.analogy.text} />
            </p>
          </aside>
        )}
      </div>
    </details>
  );
}

/** Book section first, then the paper or official source used to check the formula. */
export function StepSources({ book, source }: { book?: BookLink; source?: BookLink }) {
  return (
    <div className="step-sources">
      {book && (
        <a className="lesson-source" href={book[1]} target="_blank" rel="noreferrer">
          {Tx('对应原书：{#0} ↗', book[0])}
        </a>
      )}
      {source && (
        <a className="lesson-source" href={source[1]} target="_blank" rel="noreferrer">
          {Tx('核对来源：{#0} ↗', source[0])}
        </a>
      )}
    </div>
  );
}
