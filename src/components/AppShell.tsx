import { useEffect, type ReactNode } from 'react';
import { BOOK_URL, ROADMAP, TOPICS, type TopicId } from '../content/topics';

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
  useEffect(() => {
    document.title = `${title} · LLM 原理实验室`;
  }, [title]);
  return (
    <div className="app-shell lesson-shell">
      <aside className="sidebar">
        <a className="brand" href="/attention" aria-label="LLM 原理实验室首页">
          <span className="brand-mark">[∑]</span>
          <span>LLM 原理实验室</span>
        </a>
        <div className="nav-label">实验目录</div>
        <nav aria-label="实验目录">
          {TOPICS.map((topic) => (
            <a
              key={topic.id}
              className={topic.id === active ? 'nav-active' : 'nav-link'}
              href={topic.href}
              aria-current={topic.id === active ? 'page' : undefined}
            >
              <span className="nav-number">{topic.number}</span>
              <span>
                {topic.title}
                <small>{topic.subtitle}</small>
              </span>
              {topic.id === active && <span className="nav-dot" />}
            </a>
          ))}
        </nav>
        <div className="roadmap">
          <div className="nav-label">后续主题 · 尚未实现</div>
          {ROADMAP.map((title, i) => (
            <div className="roadmap-item" key={title}>
              <span className="nav-number">{String(i + TOPICS.length + 1).padStart(2, '0')}</span>
              {title}
            </div>
          ))}
        </div>
        <div className="sidebar-note">
          <strong>使用方法</strong>
          <ol>
            <li>改一个量</li>
            <li>先预测变化</li>
            <li>逐步核对数值</li>
            <li>隐藏说明，自己复述</li>
          </ol>
        </div>
        <a className="book-link" href={BOOK_URL} target="_blank" rel="noreferrer">
          参考阅读 <span>↗</span>
        </a>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <span>
            <span className="breadcrumb-root">核心实验</span>
            <span className="slash">/</span>
            {crumb}
          </span>
          <span className="local-indicator">
            <i />
            {indicator}
          </span>
        </header>
        <main id="main" className="main-content">
          {children}
          <footer>
            <span>LLM 原理实验室</span>
            <span>所有数值在浏览器内计算，不上传数据</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
