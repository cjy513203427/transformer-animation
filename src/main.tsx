import React, { lazy, Suspense, type ComponentType, type LazyExoticComponent } from 'react';
import ReactDOM from 'react-dom/client';
import { TOPICS } from './content/topics';
import './styles/app.css';

const AttentionLab = lazy(() => import('./features/attention/AttentionLab'));

/** Each experiment loads only when its path is opened. */
const ROUTES: Record<string, LazyExoticComponent<ComponentType>> = {
  '/': AttentionLab,
  '/attention': AttentionLab,
  '/position': lazy(() => import('./features/position/PositionLab')),
  '/inference-cache': lazy(() => import('./features/inference/InferenceLab')),
  '/kv-architectures': lazy(() => import('./features/kv/KVLab')),
  '/moe': lazy(() => import('./features/moe/MoELab')),
  '/flash-attention': lazy(() => import('./features/flash/FlashLab')),
  '/normalization': lazy(() => import('./features/p1/NormalizationLab')),
  '/ffn-residual': lazy(() => import('./features/p1/FFNLab')),
  '/lora': lazy(() => import('./features/p1/LoRALab')),
  '/quantization': lazy(() => import('./features/p1/QuantizationLab')),
  '/decoding': lazy(() => import('./features/p1/DecodingLab')),
  '/preference': lazy(() => import('./features/p1/PreferenceLab')),
  '/attention-variants': lazy(() => import('./features/p1/MemoryLab')),
  '/ssm-mamba': lazy(() => import('./features/p1/SSMLab')),
};

const path = window.location.pathname.replace(/\/+$/, '') || '/';
const Page = ROUTES[path];

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {Page ? (
      <Suspense fallback={<div className="loading">正在准备实验…</div>}>
        <Page />
      </Suspense>
    ) : (
      <main className="not-found">
        <h1>这个实验还不存在</h1>
        <p>
          {TOPICS.map((topic, i) => (
            <span key={topic.id}>
              {i > 0 && ' · '}
              <a href={topic.href}>{topic.title}</a>
            </span>
          ))}
        </p>
      </main>
    )}
  </React.StrictMode>,
);
