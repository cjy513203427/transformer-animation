import { useEffect, useRef, useState, type ReactNode } from 'react';
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
  const [directoryOpen, setDirectoryOpen] = useState(false);
  const directoryButton = useRef<HTMLButtonElement>(null);
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
          <a className="site-brand" href={localePath('/')} aria-label={T('LLM 原理实验室首页')}>
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
          <nav className="study-views" aria-label={T('学习入口')}>
            <a
              href={localePath('/chapters')}
              aria-current={active === 'chapters' ? 'page' : undefined}
              onClick={(event) => {
                if (active === 'chapters' && !chapter) event.preventDefault();
              }}
            >
              {T('按原书章节')}
            </a>
            <a href={localePath('/notes')} aria-current={notebookMode ? 'page' : undefined}>
              {T('知识笔记')}
            </a>
            <a
              href={
                bookMode
                  ? localePath('/attention')
                  : (TOPICS[activeIndex]?.href ?? localePath('/attention'))
              }
              aria-current={!bookMode ? 'page' : undefined}
              onClick={(event) => {
                if (!bookMode && !intro) event.preventDefault();
              }}
            >
              {T('按机制实验')}
            </a>
          </nav>
          <div className="course-toolbar">
            <div className="course-location">
              <span>
                {bookMode
                  ? notebookMode
                    ? T('知识点主线 · 解释、对照与复述')
                    : T('章节主线 · 讲解与实验分别标注')
                  : intro
                    ? T('学习顺序 · 开始之前')
                    : Tx(
                        '学习顺序 · 阶段 {#0} / {#1} · {#2}',
                        groupIndex + 1,
                        TOPIC_GROUPS.length,
                        TOPIC_GROUPS[groupIndex].title,
                      )}
              </span>
              <strong>
                <span className="course-position">
                  {current.number}
                  {!bookMode && ` / ${TOPICS.length}`}
                </span>
                {current.title}
              </strong>
            </div>
            <div className="course-actions">
              <div className="course-arrows" aria-label={T('相邻实验')}>
                {bookMode ? (
                  <a href={localePath('/chapters')}>{T('全部章节')}</a>
                ) : previous ? (
                  <a
                    href={previous.href}
                    aria-label={
                      previousIsIntro
                        ? T('回到导读：{#0}', previous.subtitle)
                        : T('上一实验：{#0}', previous.title)
                    }
                  >
                    ← <span>{previousIsIntro ? INTRO.title : T('上一实验')}</span>
                  </a>
                ) : (
                  <span className="course-arrow-disabled" aria-disabled="true">
                    ← <span>{T('上一实验')}</span>
                  </span>
                )}
                {bookMode ? null : next ? (
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
                        <small>{T('历史状态与计算方式对照')}</small>
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
              {Tx(
                '第一次来？先读 {#0}。',
                <a
                  href={INTRO.href}
                  aria-current={intro ? 'page' : undefined}
                  onClick={(event) => {
                    if (intro) {
                      event.preventDefault();
                      setDirectoryOpen(false);
                      directoryButton.current?.focus();
                    }
                  }}
                >
                  {Tx('{#0} 导读', INTRO.number)}
                </a>,
              )}
            </p>
          </nav>
        </div>
      </header>

      <div className="main-shell">
        <main id="main" className="main-content" tabIndex={-1} aria-label={crumb}>
          {children}
          {bookMode ? (
            <nav className="course-pagination" aria-label={T('章节学习导航')}>
              <a href={localePath('/chapters')}>
                <span>{T('全部章节')}</span>
                <strong>{T('检查知识主线与尚未覆盖的内容')}</strong>
              </a>
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
