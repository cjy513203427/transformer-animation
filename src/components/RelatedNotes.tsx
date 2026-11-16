import type { ReactNode } from 'react';
import { KNOWLEDGE_NOTES, RLHF_STAGES } from '../content/notebook';
import { NOTE_DESTINATIONS } from '../content/experimentLinks';
import { T } from '../i18n';

/** Concept notes whose destination is this page, collapsed below the experiment. */
export function RelatedNotes({ path, children }: { path: string; children?: ReactNode }) {
  const notes = KNOWLEDGE_NOTES.filter((n) => NOTE_DESTINATIONS[n.id] === path);
  if (!notes.length && !children) return null;
  return (
    <details className="details-panel related-notes">
      <summary>{T('进一步理解')}</summary>
      <div className="details-body">
        {notes.map((n) => (
          <section key={n.id} data-note={n.id}>
            <h3>{n.title}</h3>
            {n.body.map((text, i) => (
              <p key={i}>{text}</p>
            ))}
            <p>{n.boundary}</p>
            {n.id === 'rlhf' && (
              <ol className="rlhf-stages">
                {RLHF_STAGES.map((stage) => (
                  <li key={stage.data}>
                    <strong>{stage.name}</strong>
                    <p>{stage.explanation}</p>
                    <code>{stage.data}</code>
                    <p>{stage.updates}</p>
                  </li>
                ))}
              </ol>
            )}
          </section>
        ))}
        {children}
      </div>
    </details>
  );
}
