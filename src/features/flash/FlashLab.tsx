import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Rich } from '../../components/Tex';
import { StepSources, WhyIntro } from '../../components/WhyIntro';
import { computeFlash } from '../../math/flash';
import {
  flashBookNotes,
  flashMyths,
  flashSources,
  flashSteps,
  flashVerifiedDate,
  flashWhy,
} from '../../content/flash';
import {
  initialFlashState,
  FLASH_STEP_COUNT,
  moveFlashStep,
  restoreFlashState,
  serializeFlashState,
  type FlashState,
} from '../../state/flash';
import { FlashDiagram } from './FlashDiagram';
import { FlashInputs, FlashWidget } from './FlashWidgets';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/experiment.css';
import '../../styles/flash.css';
import '../../styles/editorial.css';

export default function FlashLab() {
  const [restored] = useState(() => restoreFlashState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [revision, setRevision] = useState(0);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const step = flashSteps[state.step];
  const hide = !state.view.explanations;
  const result = computeFlash(state);

  function edit(next: FlashState, preservePrediction = false) {
    setPlaying(false);
    if (!preservePrediction) setRevision((value) => value + 1);
    setState(next);
  }
  function jump(target: number) {
    setPlaying(false);
    setState((s) => ({
      ...s,
      step: Math.max(0, Math.min(FLASH_STEP_COUNT - 1, target)),
      input: {
        ...s.input,
        block: target === 3 ? Math.min(1, Math.ceil(4 / s.config.blockSize) - 1) : s.input.block,
      },
    }));
  }
  function togglePlay() {
    if (!playing && state.step === FLASH_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setState((s) => {
          const next = moveFlashStep(s, 1);
          return {
            ...next,
            input: {
              ...next.input,
              block:
                next.step === 3
                  ? Math.min(1, Math.ceil(4 / next.config.blockSize) - 1)
                  : next.input.block,
            },
          };
        }),
      8000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === FLASH_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      const next = restoreFlashState(window.location.hash);
      setPlaying(false);
      setState(next.state);
      setRevision((value) => value + 1);
      setRecallOpen(false);
      setManualLink('');
      setNotice(next.warning ?? T('已从链接恢复实验。'));
    };
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onMotion = () => {
      setReducedMotion(media.matches);
      setPlaying(false);
    };
    document.addEventListener('visibilitychange', pause);
    window.addEventListener('hashchange', restore);
    media.addEventListener('change', onMotion);
    return () => {
      document.removeEventListener('visibilitychange', pause);
      window.removeEventListener('hashchange', restore);
      media.removeEventListener('change', onMotion);
    };
  }, []);

  async function copyLink() {
    const url = new URL(localePath('/flash-attention'), window.location.origin);
    url.hash = serializeFlashState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice(T('已复制当前实验状态的链接。'));
    } catch {
      setManualLink(url.href);
      setNotice(T('浏览器未允许剪贴板访问，请从下方选择并复制链接。'));
    }
  }

  return (
    <AppShell
      active="flash"
      crumb="FlashAttention"
      indicator={T('浏览器内计算')}
      title="FlashAttention"
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            {Tx(
              '原书导读 {#0} 第 21 章 {#1} 同一公式，换一种计算顺序',
              <span>·</span>,
              <span>·</span>,
            )}
          </div>
          <h1>{Tx('FlashAttention：{#0}', <span>{T('分块，为什么结果不变？')}</span>)}</h1>
          <p>
            {T(
              'FlashAttention 算的还是同一个 Attention 公式，只是换了计算顺序：一块一块地读 K 和 V，从不把完整的分数矩阵存下来。这里有 4 个位置、每个 2 维，你会跟踪三个随读随更新的量：目前的最大值、归一化总量和输出累积。最后和一次算完的 Attention 对照，看结果是否一致。',
            )}
          </p>
        </div>
        <button className="button secondary share-button" onClick={() => void copyLink()}>
          {Tx('{#0} 复制实验链接', <span aria-hidden="true">↗</span>)}
        </button>
      </section>
      {notice && (
        <div className="notice" role="status">
          {notice}
          <button aria-label={T('关闭提示')} onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
      {manualLink && (
        <label className="manual-share">
          {T('手动复制链接')}
          <input readOnly value={manualLink} onFocus={(event) => event.target.select()} />
        </label>
      )}
      <WhyIntro why={flashWhy} />

      <section className="lesson-panel flash-panel" aria-label={T('FlashAttention 逐步实验')}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">{T('动手验证：改变分块大小，最终输出会变吗？')}</span>
            <h2>{T('读一块，更新三个状态，再读下一块。')}</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {FLASH_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps flash-steps" aria-label={T('FlashAttention 步骤')}>
          {flashSteps.map((item, i) => (
            <button
              key={item.name}
              aria-current={state.step === i ? 'step' : undefined}
              onClick={() => jump(i)}
            >
              <span>{i + 1}</span>
              {item.name}
            </button>
          ))}
        </div>
        <FlashInputs state={state} onChange={edit} result={result} />
        <div className="lesson-columns flash-columns">
          <div className="diagram-panel">
            <FlashDiagram state={state} result={result} onChange={edit} />
          </div>
          <div className="lesson-reading">
            <div className="lesson-reading-top">
              <span>{T('这一步')}</span>
              <button
                onClick={() =>
                  edit({
                    ...state,
                    view: { ...state.view, explanations: !state.view.explanations },
                  })
                }
              >
                {hide ? T('显示说明') : T('隐藏说明')}
              </button>
            </div>
            <h3>{step.title}</h3>
            {hide ? (
              <div className="lesson-recall">
                <strong>{T('先自己说一遍')}</strong>
                <p>
                  <Rich text={step.prompt} />
                </p>
              </div>
            ) : (
              <p className="lesson-explanation">
                <Rich text={step.explanation} />
              </p>
            )}
            <FlashWidget
              key={`${state.step}:${revision}`}
              state={state}
              result={result}
              onChange={edit}
            />
            <div className="lesson-takeaway">
              <strong>{T('这一步，记住一句话')}</strong>
              <p>
                <Rich text={hide ? T('先自己复述，再点击“显示说明”核对。') : step.takeaway} />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>{T('需要时再看公式')}</summary>
              <Formula expression={step.formula} />
            </details>
            <StepSources book={step.book} source={step.source} />
          </div>
        </div>
        <div className="lesson-controls">
          <div>
            <button
              className="button secondary"
              disabled={state.step === 0}
              onClick={() => jump(state.step - 1)}
            >
              {T('← 上一步')}
            </button>
            <button
              className="button primary"
              disabled={state.step === FLASH_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === FLASH_STEP_COUNT - 1
                ? T('已到最后一步')
                : T('下一步：{#0} →', flashSteps[state.step + 1].name)}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                edit({ ...initialFlashState(), view: state.view });
                setNotice(T('已恢复默认示例。'));
              }}
            >
              {T('恢复默认示例')}
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? T('暂停自动逐步') : T('播放自动逐步')}
              onClick={togglePlay}
            >
              {playing ? T('Ⅱ 暂停') : T('▶ 自动逐步')}
            </button>
            <label>
              {T('速度')}
              <select
                aria-label={T('播放速度')}
                value={state.view.speed}
                onChange={(e) =>
                  edit({ ...state, view: { ...state.view, speed: Number(e.target.value) } })
                }
              >
                <option value="0.5">0.5×</option>
                <option value="1">1×</option>
                <option value="2">2×</option>
              </select>
            </label>
          </div>
        </div>
        {reducedMotion && (
          <p className="lesson-motion">
            {T('已遵循减少动态效果偏好；图表直接切换，不做过渡动画。')}
          </p>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
          <h2>{T('不看图，解释为什么分块也能得到整行 Softmax。')}</h2>
          <p>
            {T(
              'm、l、a 各保存什么？新块出现更大的最大值时，为什么 l 和 a 都要缩放？省掉完整中间矩阵，为什么仍要计算所有可见的 Query–Key 配对？',
            )}
          </p>
        </div>
        <button
          className="button secondary"
          aria-expanded={recallOpen}
          onClick={() => {
            setPlaying(false);
            setRecallOpen((v) => !v);
            setState((s) => ({ ...s, view: { ...s.view, explanations: false } }));
          }}
        >
          {recallOpen ? T('收起参考解释') : T('查看参考解释')}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            {T(
              'm 保存已见分数的最大值，l 保存相对 m 的指数之和，a 保存同样指数加权的 V 之和。新块把最大值提高到 m′ 时，用 exp(m − m′) 把旧 l 和旧 a 换到同一基准，再加新块的贡献。最后 a / l 得到完整行输出。块大小只改变分组与浮点求和顺序。逐块在片上保留临时状态，就不用把完整 S、P 写回 HBM；完整 Attention 的配对计算仍随序列长度呈二次增长。真实 FlashAttention 使用 GPU 分块调度，本页只验证数学与账本。',
            )}
          </p>
        )}
      </section>
      <details className="details-panel source-panel">
        <summary>
          <span>{T('适用条件与来源')}</span>
          <span className="summary-hint">{T('简化假设 · 常见误区 · 参考文献')}</span>
        </summary>
        <div className="details-body">
          <p>
            {T(
              '教学配置：单头、忽略 batch，Q、K、V 都是 [4, 2] 的行向量矩阵。固定教学激活不代表语义；采用标准缩放点积 Attention，可切换因果 Mask，不含 Dropout。最后一个 K 的第一维可做 −2 到 2 的激活干预。“递增分数”展示后续块提高最大值；“书中 21.4.4 一行分数”复现原书手算的那一行，最大值在第一块；“混合输入”由种子确定。',
            )}
          </p>
          <p>
            {T(
              '内部使用 JavaScript float64，展示时才舍入，等价容差 1e-12。为了核对，页面额外保存完整参考矩阵与过程快照；这不是低内存 CUDA kernel。账本假设每元素 4 B，只统计指定教学调度的临时状态与 S / P 中间载荷，不含完整输入输出、KV 缓存、实际分配器开销或硬件耗时。',
            )}
          </p>
          <strong className="exp-notes-title">{T('与原书说法的对照')}</strong>
          <ul>
            {flashBookNotes.map((note) => (
              <li key={note}>
                <Rich text={note} />
              </li>
            ))}
          </ul>
          <strong className="exp-notes-title">{T('常见误区')}</strong>
          <ul>
            {flashMyths.map((myth) => (
              <li key={myth}>
                <Rich text={myth} />
              </li>
            ))}
          </ul>
          <ul>
            {flashSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">
            {Tx('来源核对日期：{#0}。图形与说明为本项目编写。', flashVerifiedDate)}
          </p>
        </div>
      </details>
    </AppShell>
  );
}
