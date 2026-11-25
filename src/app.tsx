import React, { lazy, Suspense, useEffect, type ComponentType, type LazyExoticComponent } from 'react';
import ReactDOM from 'react-dom/client';
import { TOPICS } from './content/topics';
import { localePath, splitLocale, T } from './i18n';

const AttentionLab = lazy(() => import('./features/attention/AttentionLab'));

/** Redirects legacy /chapters/* bookmarks to the corresponding experiment page. */
function redirectTo(path: string, keepHash = false): ComponentType {
  return function LegacyRedirect() {
    useEffect(() => {
      window.location.replace(localePath(path) + (keepHash ? window.location.hash : ''));
    }, []);
    return <div className="loading">{T('正在准备实验…')}</div>;
  };
}

/** Each experiment loads only when its path is opened. Paths exclude the /en or /de prefix. */
const ROUTES: Record<string, ComponentType | LazyExoticComponent<ComponentType>> = {
  '/': lazy(() => import('./features/intro/IntroPage')),
  '/chapters': redirectTo('/'),
  '/chapters/04': redirectTo('/tokenization'),
  '/chapters/15': redirectTo('/forward-pass', true),
  '/chapters/32': redirectTo('/post-transformer', true),
  '/notes': lazy(() => import('./features/chapters/KnowledgeNotebook')),
  '/forward-pass': lazy(() => import('./features/chapters/ForwardChapter')),
  '/post-transformer': lazy(() => import('./features/chapters/PostTransformerChapter')),
  '/linear-transform': lazy(() => import('./features/linear/LinearLab')),
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
  '/training-step': lazy(() => import('./features/p1/LearningLabs').then(m => ({default:m.TrainingLab}))),
  '/tokenization': lazy(() => import('./features/bpe/BPELab')),
  '/optimizer': lazy(() => import('./features/p1/OptimizerLab')),
  '/context-extension': lazy(() => import('./features/p1/ContextLab')),
  '/model-budget': lazy(() => import('./features/budget/BudgetLab')),
  '/policy-update': lazy(() => import('./features/p1/LearningLabs').then(m => ({default:m.PolicyLab}))),
  '/prompt-engineering': lazy(() => import('./features/p1/LearningLabs').then(m => ({default:m.PromptChainLab}))),
  '/prompt-voting': lazy(() => import('./features/p1/LearningLabs').then(m => ({default:m.PromptVoteLab}))),
  '/prompt-search': lazy(() => import('./features/p1/LearningLabs').then(m => ({default:m.PromptSearchLab}))),
  '/attention-variants': lazy(() => import('./features/p1/MemoryLab')),
  '/ssm-mamba': lazy(() => import('./features/p1/SSMLab')),
};

const Page = ROUTES[splitLocale(window.location.pathname).path];

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {Page ? (
      <Suspense fallback={<div className="loading">{T('正在准备实验…')}</div>}>
        <Page />
      </Suspense>
    ) : (
      <main className="not-found">
        <h1>{T('这个实验还不存在')}</h1>
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
