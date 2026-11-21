import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import {
  CHAPTERS,
  CHAPTER_PARTS,
  CHAPTER_VERIFIED_DATE,
  chapterHref,
  chapterLabel,
  coverageLabel,
  type Chapter,
} from '../../content/chapters';
import { TOPICS } from '../../content/topics';
import { T, localePath } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/chapters.css';

export function ChapterTrail({ id }: { id: string }) {
  const index = CHAPTERS.findIndex((c) => c.id === id);
  const neighbors = [CHAPTERS[index - 1], CHAPTERS[index + 1]];
  return (
    <nav className="chapter-trail" aria-label={T('原书相邻章节')}>
      {neighbors.map(
        (c, i) =>
          c && (
            <a key={c.id} href={chapterHref(c.id)}>
              <span>{i === 0 ? T('← 上一章') : T('下一章 →')}</span>
              <strong>
                {chapterLabel(c.id)} · {c.title}
              </strong>
            </a>
          ),
      )}
    </nav>
  );
}

function Coverage({ chapter }: { chapter: Chapter }) {
  return (
    <ul className="chapter-coverage">
      {chapter.units.map((unit) => (
        <li key={unit.name}>
          <span className={`coverage-status status-${unit.status}`}>
            {coverageLabel(unit.status)}
          </span>
          <div>
            <strong>{unit.name}</strong>
            <div className="chapter-unit-links">
              {unit.href && (
                <a href={unit.href}>
                  {unit.status === 'interactive' ? T('打开计算 →') : T('打开讲解 →')}
                </a>
              )}
              {unit.topics.map((id) => {
                const topic = TOPICS.find((t) => t.id === id)!;
                return (
                  <a key={id} href={topic.href}>
                    {topic.title} →
                  </a>
                );
              })}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function ChapterIndex() {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'interactive' | 'gaps'>('all');
  useEffect(() => {
    const reveal = () => {
      const id = /^#chapter-([0-9]{2}|[A-C])$/.exec(window.location.hash)?.[1];
      if (!id) return;
      setSearch('');
      setFilter('all');
      requestAnimationFrame(() => {
        const element = document.getElementById(`chapter-${id}`);
        if (element instanceof HTMLDetailsElement) {
          element.open = true;
          element.scrollIntoView({ block: 'start' });
        }
      });
    };
    reveal();
    window.addEventListener('hashchange', reveal);
    return () => window.removeEventListener('hashchange', reveal);
  }, []);
  const visible = CHAPTERS.filter((c) => {
    const text =
      `${c.id} ${Number(c.id) || ''} ${c.title} ${c.question} ${c.summary} ${c.units.map((u) => u.name).join(' ')}`.toLocaleLowerCase();
    return (
      text.includes(search.trim().toLocaleLowerCase()) &&
      (filter === 'all' ||
        (filter === 'interactive'
          ? c.units.some((u) => u.status === 'interactive')
          : c.units.some((u) => u.status === 'missing' || u.status === 'reference')))
    );
  });
  return (
    <AppShell
      active="chapters"
      title={T('原书章节与覆盖情况')}
      crumb={T('章节目录')}
      indicator={T('浏览器内计算')}
    >
      <article className="chapter-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('章节主线 · 32 章 + 3 个附录')}</div>
            <h1>{T('跟着原书，把知识连起来')}</h1>
            <p>
              {T(
                '先看每章要回答的问题，再进入相关实验。讲解、可操作实验、原书阅读与尚待补充分开标注；这里的状态描述网站覆盖范围，不代表你的学习进度。',
              )}
            </p>
          </div>
        </header>
        <div className="chapter-highlights">
          {['15', '32'].map((id) => {
            const c = CHAPTERS.find((c) => c.id === id)!;
            return (
              <a key={id} href={chapterHref(id)}>
                <span>{chapterLabel(id)}</span>
                <strong>{c.title} →</strong>
              </a>
            );
          })}
          <a href={localePath('/policy-update')}>
            <span>{T('交互实验')}</span>
            <strong>{T('RLHF 与策略更新')} →</strong>
          </a>
          <a href={localePath('/prompt-engineering')}>
            <span>{T('交互实验')}</span>
            <strong>{T('提示链与核对')} →</strong>
          </a>
        </div>
        <section className="chapter-index-controls" aria-label={T('筛选章节')}>
          <label>
            {T('查找章节或概念')}
            <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} />
          </label>
          <div role="group" aria-label={T('章节覆盖筛选')}>
            {(['all', 'interactive', 'gaps'] as const).map((key) => (
              <button key={key} aria-pressed={filter === key} onClick={() => setFilter(key)}>
                {key === 'all'
                  ? T('全部章节')
                  : key === 'interactive'
                    ? T('已有可操作实验')
                    : T('仍有覆盖缺口')}
              </button>
            ))}
          </div>
          <p role="status">{Tx('显示 {#0} / {#1} 项', visible.length, CHAPTERS.length)}</p>
        </section>
        <div className="coverage-key">
          {(['interactive', 'explained', 'reference', 'missing'] as const).map((status) => (
            <span key={status} className={`coverage-status status-${status}`}>
              {coverageLabel(status)}
            </span>
          ))}
          <p>
            {T(
              '“原书阅读”表示当前需要转到原书深入阅读；“尚待补充”表示网站还缺少这一机制的专门讲解或实验。某一条已有实验，不表示整章已经完整覆盖。',
            )}
          </p>
        </div>
        {CHAPTER_PARTS.map((part) => {
          const chapters = visible.filter((c) => c.part === part.id);
          if (!chapters.length) return null;
          return (
            <section key={part.id} className="chapter-part" aria-labelledby={`part-${part.id}`}>
              <h2 id={`part-${part.id}`}>{part.title}</h2>
              {chapters.map((chapter) => (
                <details className="chapter-entry" id={`chapter-${chapter.id}`} key={chapter.id}>
                  <summary>
                    <span className="chapter-number">{chapterLabel(chapter.id)}</span>
                    <span>
                      <strong>{chapter.title}</strong>
                      <small>{chapter.question}</small>
                    </span>
                    <span className="chapter-entry-count">
                      {chapter.units.some((u) => u.status === 'interactive')
                        ? T('有实验')
                        : T('导读 / 阅读')}
                    </span>
                  </summary>
                  <div className="chapter-entry-body">
                    <p>{chapter.summary}</p>
                    {chapter.prerequisites.length > 0 && (
                      <p className="chapter-prerequisites">
                        {T('先备知识：')}
                        {chapter.prerequisites.map((id) => {
                          const c = CHAPTERS.find((c) => c.id === id)!;
                          return (
                            <a key={id} href={chapterHref(id)}>
                              {chapterLabel(id)} · {c.title}
                            </a>
                          );
                        })}
                      </p>
                    )}
                    <Coverage chapter={chapter} />
                    <a
                      className="chapter-book-link"
                      href={chapter.bookUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {T('阅读原书本章 ↗')}
                    </a>
                  </div>
                </details>
              ))}
            </section>
          );
        })}
        {!visible.length && <p>{T('没有匹配的章节，试试更短的关键词。')}</p>}
        <p className="chapter-source-note">
          {Tx(
            '章节映射与覆盖范围核对：{#0}。讲解和图形为本站编写，原书用于章节导航，算法细节另按论文核对。',
            CHAPTER_VERIFIED_DATE,
          )}
        </p>
      </article>
    </AppShell>
  );
}
