import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { ShareState } from '../../components/ShareState';
import { CHAPTERS, chapterLabel, chapterHref } from '../../content/chapters';
import {
  KNOWLEDGE_NOTES,
  NOTE_GROUPS,
  NOTE_SOURCES,
  NOTEBOOK_VERIFIED_DATE,
} from '../../content/notebook';
import { TOPICS } from '../../content/topics';
import {
  initialNotebookState,
  notebookAnchor,
  restoreNotebookState,
  serializeNotebookState,
  type NotebookState,
} from '../../state/notebook';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import {
  LearningRateTool,
  PromptBuilder,
  RLHFProcess,
  ScalingTool,
  VoteTool,
} from './NotebookTools';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/experiment.css';
import '../../styles/notebook.css';

export default function KnowledgeNotebook() {
  const [restored] = useState(() => restoreNotebookState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.invalid ? T('分享状态无效，已恢复默认示例。') : '');
  const [search, setSearch] = useState('');
  const [group, setGroup] = useState('all');
  const note = KNOWLEDGE_NOTES.find((n) => n.id === state.view.selected)!;
  const visible = KNOWLEDGE_NOTES.filter(
    (n) =>
      (group === 'all' || n.group === group) &&
      `${n.title} ${n.question} ${n.body.join(' ')} ${n.chapters.join(' ')} ${n.id}`
        .toLocaleLowerCase()
        .includes(search.trim().toLocaleLowerCase()),
  );
  const edit = (next: NotebookState) => {
    setState(next);
    setNotice('');
  };
  useEffect(() => {
    const restore = () => {
      const anchor = notebookAnchor(window.location.hash);
      if (anchor) {
        setState((s) => ({ ...s, view: { selected: anchor, recall: false } }));
      } else {
        const r = restoreNotebookState(window.location.hash);
        setState(r.state);
        setNotice(r.invalid ? T('分享状态无效，已恢复默认示例。') : '');
      }
      setSearch('');
      setGroup('all');
    };
    window.addEventListener('hashchange', restore);
    return () => window.removeEventListener('hashchange', restore);
  }, []);
  return (
    <AppShell
      active="notes"
      title={T('知识笔记')}
      crumb={T('可交互知识笔记')}
      indicator={T('浏览器内计算')}
    >
      <article className="chapter-page notebook-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('按知识点复习 · 关联原书 32 章与 3 个附录')}</div>
            <h1>{T('把原理记成能解释的笔记')}</h1>
            <p>
              {T(
                '从一个问题开始，读解释、改例子、回到机制实验，最后隐藏说明复述。每条笔记标注对应章节与适用边界；章节有关联不代表复现了该章所有实现。',
              )}
            </p>
          </div>
          <ShareState getHash={() => serializeNotebookState(state)} />
        </header>
        {notice && (
          <p className="chapter-notice" role="status">
            {notice}
          </p>
        )}
        <nav className="note-shortcuts" aria-label={T('重点知识点')}>
          {(['rlhf', 'prompt', 'self-consistency', 'learning-rate', 'scaling'] as const).map(
            (id) => (
              <a key={id} href={'#' + id}>
                {KNOWLEDGE_NOTES.find((n) => n.id === id)!.title}
              </a>
            ),
          )}
        </nav>
        <div className="notebook-layout">
          <aside className="notebook-directory" aria-label={T('知识笔记目录')}>
            <label>
              {T('查找知识点')}
              <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} />
            </label>
            <label>
              {T('知识分类')}
              <select value={group} onChange={(e) => setGroup(e.target.value)}>
                <option value="all">{T('全部知识点')}</option>
                {NOTE_GROUPS.map((g) => (
                  <option value={g.id} key={g.id}>
                    {g.title}
                  </option>
                ))}
              </select>
            </label>
            <p className="note-count" role="status">
              {Tx('显示 {#0} / {#1} 项', visible.length, KNOWLEDGE_NOTES.length)}
            </p>
            <nav aria-label={T('选择知识点')}>
              {NOTE_GROUPS.map((g) => {
                const items = visible.filter((n) => n.group === g.id);
                return items.length ? (
                  <section key={g.id}>
                    <h2>{g.title}</h2>
                    <ul>
                      {items.map((n) => (
                        <li key={n.id}>
                          <a
                            href={'#' + n.id}
                            aria-current={n.id === note.id ? 'page' : undefined}
                            onClick={() => {
                              edit({ ...state, view: { selected: n.id, recall: false } });
                              requestAnimationFrame(() =>
                                document.getElementById('note-reading')?.focus(),
                              );
                            }}
                          >
                            {n.title}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </section>
                ) : null;
              })}
            </nav>
            {!visible.length && <p>{T('没有匹配的知识点，试试更短的关键词。')}</p>}
          </aside>
          <section
            id="note-reading"
            data-note={note.id}
            className="note-reading"
            tabIndex={-1}
            aria-labelledby="note-title"
          >
            <div className="note-reading-header">
              <p className="eyebrow">{NOTE_GROUPS.find((g) => g.id === note.group)!.title}</p>
              <h2 id="note-title">{note.title}</h2>
              <p className="note-question">{note.question}</p>
              <button
                className="button secondary"
                onClick={() =>
                  edit({ ...state, view: { ...state.view, recall: !state.view.recall } })
                }
              >
                {state.view.recall ? T('显示说明') : T('隐藏说明，试着复述')}
              </button>
            </div>
            {!state.view.recall && (
              <>
                <div className="note-explanation">
                  {note.body.map((text, i) => (
                    <p key={i}>{text}</p>
                  ))}
                </div>
                <div className="note-example">
                  <h3>{T('最小例子与对照')}</h3>
                  <p>{note.example}</p>
                </div>
                {(note.id === 'prompt' || note.id === 'few-shot') && (
                  <PromptBuilder state={state} onChange={edit} />
                )}
                {note.id === 'self-consistency' && <VoteTool state={state} onChange={edit} />}
                {note.id === 'rlhf' && <RLHFProcess state={state} onChange={edit} />}
                {note.id === 'learning-rate' && <LearningRateTool state={state} onChange={edit} />}
                {note.id === 'scaling' && <ScalingTool state={state} onChange={edit} />}
                <div className="note-boundary">
                  <h3>{T('适用条件与常见误区')}</h3>
                  <p>{note.boundary}</p>
                </div>
              </>
            )}
            <section className="note-recall" aria-label={T('复述提示')}>
              <h3>{T('合上笔记，试着解释')}</h3>
              <p>{note.recall}</p>
              <details key={note.id + String(state.view.recall)}>
                <summary>{T('查看参考解释')}</summary>
                <p>{note.body[0]}</p>
                <p>{note.boundary}</p>
              </details>
            </section>
            {(note.topics.length > 0 || note.related?.length) && (
              <nav className="note-related" aria-label={T('相关机制实验')}>
                <h3>{T('去实验里验证')}</h3>
                {note.topics.map((id) => {
                  const topic = TOPICS.find((t) => t.id === id)!;
                  return (
                    <a key={id} href={topic.href}>
                      {topic.title} →
                    </a>
                  );
                })}
                {note.related?.map((id) => (
                  <a key={id} href={chapterHref(id)}>
                    {CHAPTERS.find((c) => c.id === id)!.title} →
                  </a>
                ))}
              </nav>
            )}
            <details className="note-sources" key={'sources-' + note.id}>
              <summary>{T('对应章节与原始来源')}</summary>
              <p>{T('章节用于定位；本站解释为自行编写，公开算法按论文或作者资料核对。')}</p>
              <ul>
                {note.chapters.map((id) => {
                  const c = CHAPTERS.find((c) => c.id === id)!;
                  return (
                    <li key={id}>
                      <a href={c.bookUrl} target="_blank" rel="noreferrer">
                        {chapterLabel(id)} · {c.title} ↗
                      </a>
                    </li>
                  );
                })}
                {note.sources.map((id) => (
                  <li key={id}>
                    <a href={NOTE_SOURCES[id][1]} target="_blank" rel="noreferrer">
                      {NOTE_SOURCES[id][0]} ↗
                    </a>
                  </li>
                ))}
              </ul>
              <p>{Tx('内容核对日期：{#0}', NOTEBOOK_VERIFIED_DATE)}</p>
            </details>
            <div className="chapter-preset-buttons note-reset">
              <a href={localePath('/chapters')}>{T('查看原书覆盖范围')}</a>
              <button
                onClick={() => {
                  edit(initialNotebookState());
                  setGroup('all');
                  setSearch('');
                  window.history.replaceState(null, '', window.location.pathname);
                }}
              >
                {T('重置全部笔记实验')}
              </button>
            </div>
          </section>
        </div>
      </article>
    </AppShell>
  );
}
