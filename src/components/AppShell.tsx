import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { BOOK_URL } from '../content/book';
import { INTRO, ROADMAP, TOPICS, TOPIC_GROUPS, type TopicId } from '../content/topics';
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
import {
  DEFAULT_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_READING_ZOOM,
  MAX_READING_ZOOM,
  useReadingLayout,
} from './useReadingLayout';
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
  chapter,
  children,
}: {
  /** The current experiment, or the introduction that precedes them. */
  active: TopicId | 'intro' | 'chapters' | 'notes';
  crumb: string;
  indicator: string;
  title: string;
  chapter?: { id: string; title: string };
  children: ReactNode;
}) {
  const { width, zoom, desktop, directoryOpen, setDirectoryOpen, setWidth, setZoom } =
    useReadingLayout();
  const [resizing, setResizing] = useState(false);
  const directoryButton = useRef<HTMLButtonElement>(null);
  const sidebar = useRef<HTMLElement>(null);
  const mainShell = useRef<HTMLDivElement>(null);
  const drag = useRef<{ pointerId: number; startX: number; width: number } | null>(null);
  const intro = active === 'intro';
  const notebookMode = active === 'notes';
  const bookMode = active === 'chapters' || notebookMode;
  const activeIndex = TOPICS.findIndex((topic) => topic.id === active);
  const current = notebookMode
    ? { number: '26', title: T('知识笔记') }
    : bookMode
      ? { number: chapter?.id ?? '32 + 3', title: chapter?.title ?? T('原书章节与覆盖情况') }
      : intro
        ? INTRO
        : TOPICS[activeIndex];
  // The first experiment leads back to the introduction; the introduction leads into it.
  const previous =
      intro || bookMode ? undefined : activeIndex === 0 ? INTRO : TOPICS[activeIndex - 1],
    next = bookMode ? undefined : TOPICS[activeIndex + 1];
  const previousIsIntro = previous === INTRO;
  const group = intro || bookMode ? undefined : TOPICS[activeIndex].group;
  const groupIndex = TOPIC_GROUPS.findIndex((g) => g.id === group);
  const closeDirectory = () => {
    setDirectoryOpen(false);
    directoryButton.current?.focus({ preventScroll: true });
  };
  useEffect(() => {
    document.title = T('{#0} · LLM 原理实验室', title);
  }, [title]);
  useEffect(() => {
    if (!directoryOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDirectoryOpen(false);
        directoryButton.current?.focus({ preventScroll: true });
      }
      if (!desktop && event.key === 'Tab') {
        const controls = sidebar.current?.querySelectorAll<HTMLElement>(
          'a[href], button:not(:disabled)',
        );
        if (!controls?.length) return;
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [directoryOpen, desktop]);
  useEffect(() => {
    if (desktop || !directoryOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    if (mainShell.current) mainShell.current.inert = true;
    sidebar.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true });
    return () => {
      document.body.style.overflow = previousOverflow;
      if (mainShell.current) mainShell.current.inert = false;
      directoryButton.current?.focus({ preventScroll: true });
    };
  }, [desktop, directoryOpen]);
  return (
    <div
      className={`app-shell lesson-shell editorial-shell${directoryOpen ? ' directory-open' : ''}${resizing ? ' directory-resizing' : ''}`}
      style={{ '--sidebar-width': `${width}px` } as CSSProperties}
    >
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
      {!desktop && directoryOpen && (
        <button
          className="directory-backdrop"
          aria-label={T('收起目录')}
          onClick={closeDirectory}
          tabIndex={-1}
        />
      )}
      <aside
        ref={sidebar}
        id="course-sidebar"
        className="course-sidebar"
        hidden={!directoryOpen}
        role={!desktop && directoryOpen ? 'dialog' : undefined}
        aria-modal={!desktop && directoryOpen ? true : undefined}
        aria-label={T('学习目录')}
      >
        <div className="masthead">
          <a className="site-brand" href={localePath('/')} aria-label={T('LLM 原理实验室首页')}>
            <span className="site-brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span>{T('LLM 原理实验室')}</span>
          </a>
          <button
            className="directory-close"
            aria-label={T('收起目录')}
            title={T('收起目录')}
            onClick={closeDirectory}
          >
            <span aria-hidden="true">‹</span>
          </button>
        </div>
        <div className="course-navigation">
          <p className="directory-heading">{T('学习目录')}</p>
          <nav id="course-directory" className="course-directory" aria-label={T('实验目录')}>
            <a
              className="course-introduction"
              href={INTRO.href}
              aria-label={T('{#0} 导读', INTRO.number)}
              aria-current={intro ? 'page' : undefined}
              onClick={(event) => {
                if (intro) {
                  event.preventDefault();
                  closeDirectory();
                }
              }}
            >
              <span className="course-item-number" aria-hidden="true">
                {INTRO.number}
              </span>
              <span>{INTRO.title}</span>
            </a>
            <a
              className="course-introduction"
              href={localePath('/notes')}
              aria-current={notebookMode ? 'page' : undefined}
              onClick={(event) => {
                if (notebookMode) {
                  event.preventDefault();
                  closeDirectory();
                }
              }}
            >
              {T('知识笔记')}
            </a>
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
                  <p className="course-group-description">{group.description}</p>
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
                              closeDirectory();
                            }
                          }}
                        >
                          <span className="course-item-number" aria-hidden="true">
                            {topic.number}
                          </span>
                          <span>
                            <strong>{topic.title}</strong>
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
                  {group.id === 'architecture' && (
                    <a
                      className="course-architecture-guide"
                      href={localePath('/chapters/32') + '#family-rwkv'}
                      aria-label={T('RWKV / RetNet / Hyena 架构导读')}
                    >
                      <span className="course-item-number" aria-hidden="true">
                        ↳
                      </span>
                      <span>
                        <strong>{T('RWKV / RetNet / Hyena 架构导读')}</strong>
                      </span>
                    </a>
                  )}
                  {group.id === 'adaptation' && (
                    <a className="course-note-guide" href={localePath('/notes') + '#rlhf'}>
                      <span className="course-item-number" aria-hidden="true">
                        ↳
                      </span>
                      <span>
                        <strong>{T('RLHF 全流程笔记')}</strong>
                        <small>{T('SFT、奖励模型与策略更新')}</small>
                      </span>
                    </a>
                  )}
                  {group.id === 'generation' && (
                    <a className="course-note-guide" href={localePath('/notes') + '#prompt'}>
                      <span className="course-item-number" aria-hidden="true">
                        ↳
                      </span>
                      <span>
                        <strong>{T('Prompt Engineering 提示工程')}</strong>
                        <small>{T('示例、思维链、投票与工具流程')}</small>
                      </span>
                    </a>
                  )}
                </section>
              );
            })}
            <p className="course-directory-note">
              {T(
                '编号表示建议学习顺序，与原书章节编号不同；可按顺序学习，也可直接跳到需要的实验。',
              )}{' '}
            </p>
          </nav>
        </div>
        <div className="sidebar-footer">
          <a className="site-reading-link" href={BOOK_URL} target="_blank" rel="noreferrer">
            {Tx('阅读原书 {#0}', <span aria-hidden="true">↗</span>)}
          </a>
          <LanguageSwitch />
        </div>
        <div
          className="directory-resize"
          role="separator"
          aria-label={T('调整目录宽度')}
          aria-orientation="vertical"
          aria-valuemin={MIN_SIDEBAR_WIDTH}
          aria-valuemax={MAX_SIDEBAR_WIDTH}
          aria-valuenow={width}
          aria-controls="course-sidebar"
          title={T('拖动调整目录宽度；方向键微调，双击恢复')}
          tabIndex={desktop ? 0 : -1}
          onDoubleClick={() => setWidth(DEFAULT_SIDEBAR_WIDTH)}
          onKeyDown={(event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            setWidth(
              event.key === 'Home'
                ? MIN_SIDEBAR_WIDTH
                : event.key === 'End'
                  ? MAX_SIDEBAR_WIDTH
                  : width + (event.key === 'ArrowLeft' ? -10 : 10),
            );
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.preventDefault();
            event.currentTarget.focus({ preventScroll: true });
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = { pointerId: event.pointerId, startX: event.clientX, width };
            setResizing(true);
          }}
          onPointerMove={(event) => {
            if (drag.current?.pointerId === event.pointerId)
              setWidth(drag.current.width + event.clientX - drag.current.startX);
          }}
          onPointerUp={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId);
            drag.current = null;
            setResizing(false);
          }}
          onLostPointerCapture={() => {
            drag.current = null;
            setResizing(false);
          }}
        />
      </aside>

      <div ref={mainShell} className="main-shell">
        <header className="site-header">
          <div className="course-toolbar">
            <button
              ref={directoryButton}
              className="course-directory-toggle"
              aria-label={T('学习目录')}
              title={T('学习目录')}
              aria-expanded={directoryOpen}
              aria-controls="course-sidebar"
              onClick={() => setDirectoryOpen((open) => !open)}
            >
              <svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true">
                <rect x="2.5" y="3.5" width="15" height="13" rx="1.5" />
                <path d="M7 4v12M10 7h4M10 10h4M10 13h4" />
              </svg>
              <span>{T('学习目录')}</span>
            </button>
            <div className="course-location">
              <span>
                {bookMode
                  ? T('学习目录')
                  : intro
                    ? T('学习顺序 · 开始之前')
                    : TOPIC_GROUPS[groupIndex].title}
              </span>
              <span className="course-location-divider" aria-hidden="true">
                /
              </span>
              <strong title={current.title}>
                <span className="course-position">{current.number}</span>
                {current.title}
              </strong>
            </div>
            <div className="reading-zoom" role="group" aria-label={T('阅读缩放')}>
              <button
                aria-label={T('缩小阅读内容')}
                title={T('缩小阅读内容')}
                disabled={zoom <= MIN_READING_ZOOM}
                onClick={() => setZoom(zoom - 10)}
              >
                −
              </button>
              <button
                className="reading-zoom-reset"
                aria-label={T('恢复 100% 缩放')}
                title={T('恢复 100% 缩放')}
                onClick={() => setZoom(100)}
              >
                {zoom}%
              </button>
              <button
                aria-label={T('放大阅读内容')}
                title={T('放大阅读内容')}
                disabled={zoom >= MAX_READING_ZOOM}
                onClick={() => setZoom(zoom + 10)}
              >
                +
              </button>
            </div>
            <nav className="course-arrows" aria-label={T('相邻实验')}>
              {!bookMode && (
                <>
                  {previous ? (
                    <a
                      href={previous.href}
                      aria-label={
                        previousIsIntro
                          ? T('回到导读：{#0}', previous.subtitle)
                          : T('上一实验：{#0}', previous.title)
                      }
                      title={previous.title}
                    >
                      ←
                    </a>
                  ) : (
                    <span
                      className="course-arrow-disabled"
                      role="link"
                      aria-disabled="true"
                      aria-label={T('上一实验')}
                    >
                      <span aria-hidden="true">←</span>
                    </span>
                  )}
                  {next ? (
                    <a
                      href={next.href}
                      aria-label={T('下一实验：{#0}', next.title)}
                      title={next.title}
                    >
                      →
                    </a>
                  ) : (
                    <span
                      className="course-arrow-disabled"
                      role="link"
                      aria-disabled="true"
                      aria-label={T('下一实验')}
                    >
                      <span aria-hidden="true">→</span>
                    </span>
                  )}
                </>
              )}
            </nav>
          </div>
        </header>
        <main
          id="main"
          className="main-content"
          tabIndex={-1}
          aria-label={crumb}
          style={{ zoom: zoom / 100 }}
        >
          {children}
          {bookMode ? (
            <nav className="course-pagination" aria-label={T('章节学习导航')}>
              <a href={localePath('/attention')}>
                <span>{T('按机制实验')}</span>
                <strong>{T('进入可计算的机制实验')}</strong>
              </a>
            </nav>
          ) : (
            <nav className="course-pagination" aria-label={T('按顺序继续学习')}>
              {previous ? (
                <a
                  href={previous.href}
                  aria-label={
                    previousIsIntro
                      ? T('回到导读：{#0}', previous.subtitle)
                      : T('上一实验：{#0}', previous.title)
                  }
                >
                  <span>{previousIsIntro ? T('← 开始之前') : T('← 上一实验')}</span>
                  <strong>
                    {previous.number} · {previous.title}
                  </strong>
                  <small>{previous.subtitle}</small>
                </a>
              ) : (
                <div className="course-boundary">
                  <span>{T('学习起点')}</span>
                  <strong>{T('从这里拆开计算过程')}</strong>
                  <small>{T('可用左侧学习目录跳到其他实验')}</small>
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
          )}
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
