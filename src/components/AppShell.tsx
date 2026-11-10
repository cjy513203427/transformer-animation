import { useEffect, useRef, useState, type ReactNode } from 'react';
import { BOOK_URL } from '../content/book';
import { ROADMAP, TOPICS, TOPIC_GROUPS, type TopicId } from '../content/topics';
import {
  HTML_LANG,
  LOCALE,
  LOCALE_LABEL,
  LOCALES,
  localePath,
  splitLocale,
  T,
  type Locale,
} from '../i18n';
import { Tx } from '../i18n/jsx';
import '../styles/shell.css';

const SHORT_LABEL: Record<Locale, string> = { zh: '中文', en: 'EN', de: 'DE' };

/** Links to the same experiment in each language; the hash keeps the current inputs and step. */
function LanguageSwitch() {
  const { path } = splitLocale(window.location.pathname);
  return (
    <nav className="site-language" aria-label={T('界面语言')}>
      {LOCALES.map((locale) => (
        <a
          key={locale}
          href={localePath(path, locale)}
          hrefLang={HTML_LANG[locale]}
          lang={HTML_LANG[locale]}
          aria-label={LOCALE_LABEL[locale]}
          aria-current={locale === LOCALE ? 'true' : undefined}
          onClick={(event) => {
            event.currentTarget.href = localePath(path, locale) + window.location.hash;
          }}
        >
          <span className="language-full">{LOCALE_LABEL[locale]}</span>
          <span className="language-short" aria-hidden="true">
            {SHORT_LABEL[locale]}
          </span>
        </a>
      ))}
    </nav>
  );
}

export function AppShell({
  active,
  crumb,
  indicator,
  title,
  children,
}: {
  active: TopicId;
  crumb: string;
  indicator: string;
  title: string;
  children: ReactNode;
}) {
  const [directoryOpen, setDirectoryOpen] = useState(false);
  const directoryButton = useRef<HTMLButtonElement>(null);
  const activeIndex = TOPICS.findIndex((topic) => topic.id === active);
  const current = TOPICS[activeIndex],
    previous = TOPICS[activeIndex - 1],
    next = TOPICS[activeIndex + 1];
  const groupIndex = TOPIC_GROUPS.findIndex((group) => group.id === current.group);
  useEffect(() => {
    document.title = T('{#0} · LLM 原理实验室', title);
  }, [title]);
  useEffect(() => {
    if (!directoryOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDirectoryOpen(false);
        directoryButton.current?.focus();
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [directoryOpen]);
  return (
    <div className="app-shell lesson-shell editorial-shell">
      <a
        className="skip-to-content"
        href="#main"
        onClick={(event) => {
          // The hash stores the experiment, so keyboard navigation must not replace it.
          event.preventDefault();
          const main = document.getElementById('main');
          main?.focus();
          main?.scrollIntoView({ block: 'start' });
        }}
      >
        {T('跳到实验内容')}
      </a>
      <header className="site-header">
        <div className="masthead">
          <a
            className="site-brand"
            href={localePath('/attention')}
            aria-label={T('LLM 原理实验室首页')}
          >
            <span className="site-brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span>{T('LLM 原理实验室')}</span>
          </a>
          <div className="masthead-tools">
            <a className="site-reading-link" href={BOOK_URL} target="_blank" rel="noreferrer">
              {Tx('阅读原书 {#0}', <span aria-hidden="true">↗</span>)}
            </a>
            <LanguageSwitch />
          </div>
        </div>
        <div className="course-navigation">
          <div className="course-toolbar">
            <div className="course-location">
              <span>
                {Tx(
                  '学习顺序 · 阶段 {#0} / {#1} · {#2}',
                  groupIndex + 1,
                  TOPIC_GROUPS.length,
                  TOPIC_GROUPS[groupIndex].title,
                )}
              </span>
              <strong>
                <span className="course-position">
                  {current.number} / {TOPICS.length}
                </span>
                {current.title}
              </strong>
            </div>
            <div className="course-actions">
              <div className="course-arrows" aria-label={T('相邻实验')}>
                {previous ? (
                  <a href={previous.href} aria-label={T('上一实验：{#0}', previous.title)}>
                    ← <span>{T('上一实验')}</span>
                  </a>
                ) : (
                  <span className="course-arrow-disabled" aria-disabled="true">
                    ← <span>{T('上一实验')}</span>
                  </span>
                )}
                {next ? (
                  <a href={next.href} aria-label={T('下一实验：{#0}', next.title)}>
                    <span>{T('下一实验')}</span> →
                  </a>
                ) : (
                  <span className="course-arrow-disabled" aria-disabled="true">
                    <span>{T('下一实验')}</span> →
                  </span>
                )}
              </div>
              <button
                ref={directoryButton}
                className="course-directory-toggle"
                aria-expanded={directoryOpen}
                aria-controls="course-directory"
                onClick={() => setDirectoryOpen((open) => !open)}
              >
                {Tx(
                  '学习目录 {#0}',
                  <span aria-hidden="true">{directoryOpen ? '−' : T('＋')}</span>,
                )}
              </button>
            </div>
          </div>
          <nav
            id="course-directory"
            className="course-directory"
            aria-label={T('实验目录')}
            hidden={!directoryOpen}
          >
            {TOPIC_GROUPS.map((group, index) => {
              const topics = TOPICS.filter((topic) => topic.group === group.id);
              return (
                <section
                  key={group.id}
                  className="course-group"
                  aria-labelledby={`group-${group.id}`}
                >
                  <h2 id={`group-${group.id}`}>
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    {group.title}
                  </h2>
                  <p>{group.description}</p>
                  <ol start={Number(topics[0].number)}>
                    {topics.map((topic) => (
                      <li key={topic.id}>
                        <a
                          href={topic.href}
                          aria-label={topic.title}
                          aria-current={topic.id === active ? 'page' : undefined}
                          onClick={(event) => {
                            if (topic.id === active) {
                              event.preventDefault();
                              setDirectoryOpen(false);
                              directoryButton.current?.focus();
                            }
                          }}
                        >
                          <span className="course-item-number" aria-hidden="true">
                            {topic.number}
                          </span>
                          <span>
                            <strong>{topic.title}</strong>
                            <small>{topic.subtitle}</small>
                          </span>
                          {topic.id === active && (
                            <span className="course-current-mark" aria-hidden="true">
                              {T('当前')}
                            </span>
                          )}
                        </a>
                      </li>
                    ))}
                  </ol>
                </section>
              );
            })}
            <p className="course-directory-note">
              {T(
                '编号表示建议学习顺序，与原书章节编号不同；可按顺序学习，也可直接跳到需要的实验。',
              )}
            </p>
          </nav>
        </div>
      </header>

      <div className="main-shell">
        <main id="main" className="main-content" tabIndex={-1} aria-label={crumb}>
          {children}
          <nav className="course-pagination" aria-label={T('按顺序继续学习')}>
            {previous ? (
              <a href={previous.href} aria-label={T('上一实验：{#0}', previous.title)}>
                <span>{T('← 上一实验')}</span>
                <strong>
                  {previous.number} · {previous.title}
                </strong>
                <small>{previous.subtitle}</small>
              </a>
            ) : (
              <div className="course-boundary">
                <span>{T('学习起点')}</span>
                <strong>{T('从这里拆开计算过程')}</strong>
                <small>{T('可用顶部学习目录跳到其他实验')}</small>
              </div>
            )}
            {next ? (
              <a
                href={next.href}
                className="course-next"
                aria-label={T('下一实验：{#0}', next.title)}
              >
                <span>{T('下一实验 →')}</span>
                <strong>
                  {next.number} · {next.title}
                </strong>
                <small>{next.subtitle}</small>
              </a>
            ) : (
              <div className="course-boundary">
                <span>{T('学习路径末尾')}</span>
                <strong>{T('回到目录选择需要复习的机制')}</strong>
                <small>{T('编号表示位置，不记录完成状态')}</small>
              </div>
            )}
          </nav>
          <footer className="site-footer">
            <div className="site-footer-primary">
              <span>{T('LLM 原理实验室')}</span>
              <a href={BOOK_URL} target="_blank" rel="noreferrer">
                {Tx('参考：Transformer 架构，从直觉到实现 {#0}', <span aria-hidden="true">↗</span>)}
              </a>
            </div>
            <div className="site-footer-secondary">
              <span>{Tx('{#0}，不上传数据', indicator)}</span>
              <span>
                {Tx(
                  '{#0} 个实验已开放{#1}',
                  TOPICS.length,
                  ROADMAP.length ? T('；后续：{#0}', ROADMAP.join(T('、'))) : '',
                )}
              </span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
