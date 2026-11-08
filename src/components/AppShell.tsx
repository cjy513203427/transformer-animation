import { useEffect, useRef, useState, type ReactNode } from 'react';
import { BOOK_URL, ROADMAP, TOPICS, TOPIC_GROUPS, type TopicId } from '../content/topics';
import '../styles/shell.css';

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
    document.title = `${title} · LLM 原理实验室`;
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
        跳到实验内容
      </a>
      <header className="site-header">
        <div className="masthead">
          <a className="site-brand" href="/attention" aria-label="LLM 原理实验室首页">
            <span className="site-brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span>LLM 原理实验室</span>
          </a>
          <a className="site-reading-link" href={BOOK_URL} target="_blank" rel="noreferrer">
            阅读原书 <span aria-hidden="true">↗</span>
          </a>
        </div>
        <div className="course-navigation">
          <div className="course-toolbar">
            <div className="course-location">
              <span>
                学习顺序 · 阶段 {groupIndex + 1} / {TOPIC_GROUPS.length} ·{' '}
                {TOPIC_GROUPS[groupIndex].title}
              </span>
              <strong>
                <span className="course-position">
                  {current.number} / {TOPICS.length}
                </span>
                {current.title}
              </strong>
            </div>
            <div className="course-actions">
              <div className="course-arrows" aria-label="相邻实验">
                {previous ? (
                  <a href={previous.href} aria-label={`上一实验：${previous.title}`}>
                    ← <span>上一实验</span>
                  </a>
                ) : (
                  <span className="course-arrow-disabled" aria-disabled="true">
                    ← <span>上一实验</span>
                  </span>
                )}
                {next ? (
                  <a href={next.href} aria-label={`下一实验：${next.title}`}>
                    <span>下一实验</span> →
                  </a>
                ) : (
                  <span className="course-arrow-disabled" aria-disabled="true">
                    <span>下一实验</span> →
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
                学习目录 <span aria-hidden="true">{directoryOpen ? '−' : '＋'}</span>
              </button>
            </div>
          </div>
          <nav
            id="course-directory"
            className="course-directory"
            aria-label="实验目录"
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
                              当前
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
              编号表示建议学习顺序，与原书章节编号不同；可按顺序学习，也可直接跳到需要的实验。
            </p>
          </nav>
        </div>
      </header>

      <div className="main-shell">
        <main id="main" className="main-content" tabIndex={-1} aria-label={crumb}>
          {children}
          <nav className="course-pagination" aria-label="按顺序继续学习">
            {previous ? (
              <a href={previous.href} aria-label={`上一实验：${previous.title}`}>
                <span>← 上一实验</span>
                <strong>
                  {previous.number} · {previous.title}
                </strong>
                <small>{previous.subtitle}</small>
              </a>
            ) : (
              <div className="course-boundary">
                <span>学习起点</span>
                <strong>从这里拆开计算过程</strong>
                <small>可用顶部学习目录跳到其他实验</small>
              </div>
            )}
            {next ? (
              <a href={next.href} className="course-next" aria-label={`下一实验：${next.title}`}>
                <span>下一实验 →</span>
                <strong>
                  {next.number} · {next.title}
                </strong>
                <small>{next.subtitle}</small>
              </a>
            ) : (
              <div className="course-boundary">
                <span>学习路径末尾</span>
                <strong>回到目录选择需要复习的机制</strong>
                <small>编号表示位置，不记录完成状态</small>
              </div>
            )}
          </nav>
          <footer className="site-footer">
            <div className="site-footer-primary">
              <span>LLM 原理实验室</span>
              <a href={BOOK_URL} target="_blank" rel="noreferrer">
                参考：Transformer 架构，从直觉到实现 <span aria-hidden="true">↗</span>
              </a>
            </div>
            <div className="site-footer-secondary">
              <span>{indicator}，不上传数据</span>
              <span>
                {TOPICS.length} 个实验已开放{ROADMAP.length ? `；后续：${ROADMAP.join('、')}` : ''}
              </span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
