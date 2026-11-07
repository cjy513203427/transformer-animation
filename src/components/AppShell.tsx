import { useEffect, useRef, type ReactNode } from 'react';
import { BOOK_URL, ROADMAP, TOPICS, type TopicId } from '../content/topics';
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
  const navigationRef = useRef<HTMLElement>(null);
  useEffect(() => {
    document.title = `${title} · LLM 原理实验室`;
  }, [title]);
  useEffect(() => {
    const navigation = navigationRef.current;
    if (!navigation || navigation.scrollWidth <= navigation.clientWidth) return;
    const current = navigation.querySelector('[aria-current="page"]');
    if (!current) return;
    const viewport = navigation.getBoundingClientRect();
    const item = current.getBoundingClientRect();
    // Scroll this strip only: scrollIntoView would also move the article itself.
    if (item.left < viewport.left + 12) {
      navigation.scrollLeft += item.left - viewport.left - 12;
    } else if (item.right > viewport.right - 12) {
      navigation.scrollLeft += item.right - viewport.right + 12;
    }
  }, [active]);
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
        <div className="topic-navigation-wrap">
          <nav ref={navigationRef} className="topic-navigation" aria-label="实验目录">
            {TOPICS.map((topic) => (
              <a
                key={topic.id}
                href={topic.href}
                aria-current={topic.id === active ? 'page' : undefined}
              >
                {topic.title}
              </a>
            ))}
          </nav>
          <span className="topic-navigation-hint" aria-hidden="true">
            左右滑动查看实验 →
          </span>
        </div>
      </header>

      <div className="main-shell">
        <main id="main" className="main-content" tabIndex={-1} aria-label={crumb}>
          {children}
          <footer className="site-footer">
            <div className="site-footer-primary">
              <span>LLM 原理实验室</span>
              <a href={BOOK_URL} target="_blank" rel="noreferrer">
                参考：Transformer 架构，从直觉到实现 <span aria-hidden="true">↗</span>
              </a>
            </div>
            <div className="site-footer-secondary">
              <span>{indicator}，不上传数据</span>
              <span>待补充：{ROADMAP.join('、')}</span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
