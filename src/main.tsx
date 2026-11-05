import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom/client';
import './styles/app.css';

const AttentionLab = lazy(() => import('./features/attention/AttentionLab'));
const knownRoute = ['/', '/attention', '/attention/'].includes(window.location.pathname);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {knownRoute ? (
      <Suspense fallback={<div className="loading">正在准备 Attention 计算台…</div>}>
        <AttentionLab />
      </Suspense>
    ) : (
      <main className="not-found">
        <h1>这个实验还不存在</h1>
        <a href="/attention">回到 Attention 计算台</a>
      </main>
    )}
  </React.StrictMode>,
);
