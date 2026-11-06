import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom/client';
import './styles/app.css';

const AttentionLab = lazy(() => import('./features/attention/AttentionLab'));
const PositionLab = lazy(() => import('./features/position/PositionLab'));
const path = window.location.pathname.replace(/\/+$/, '') || '/';
const page =
  path === '/' || path === '/attention' ? (
    <AttentionLab />
  ) : path === '/position' ? (
    <PositionLab />
  ) : null;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {page ? (
      <Suspense fallback={<div className="loading">正在准备实验…</div>}>{page}</Suspense>
    ) : (
      <main className="not-found">
        <h1>这个实验还不存在</h1>
        <p>
          <a href="/attention">回到 Attention 计算台</a> · <a href="/position">位置编码</a>
        </p>
      </main>
    )}
  </React.StrictMode>,
);
