import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Rich } from '../../components/Tex';
import { StepSources, WhyIntro } from '../../components/WhyIntro';
import { kvBookNotes, kvMyths, kvSources, kvSteps, kvWhy } from '../../content/kv';
import {
  initialKVState,
  KV_STEP_COUNT,
  moveKVStep,
  restoreKVState,
  serializeKVState,
  type KVState,
} from '../../state/kv';
import { KVDiagram } from './KVDiagram';
import {
  AbsorbWidget,
  CalcWidget,
  CompareWidget,
  CompressWidget,
  GroupWidget,
  RopeWidget,
} from './KVWidgets';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/experiment.css';
import '../../styles/kv.css';
import '../../styles/editorial.css';

export default function KVLab() {
  const [restored] = useState(() => restoreKVState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const step = kvSteps[state.step];
  const hide = !state.view.explanations;

  /** Any edit by the learner pauses playback; every number is recomputed from the new state. */
  function edit(next: KVState) {
    setPlaying(false);
    setState(next);
  }
  function jump(target: number) {
    setPlaying(false);
    setState((s) => ({ ...s, step: Math.max(0, Math.min(KV_STEP_COUNT - 1, target)) }));
  }
  function togglePlay() {
    if (!playing && state.step === KV_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => moveKVStep(s, 1)),
      8000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === KV_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      const next = restoreKVState(window.location.hash);
      setPlaying(false);
      setState(next.state);
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
    const url = new URL(localePath('/kv-architectures'), window.location.origin);
    url.hash = serializeKVState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice(T('已复制当前实验状态的链接。'));
    } catch {
      setManualLink(url.href);
      setNotice(T('浏览器未允许剪贴板访问，请从下方选择并复制链接。'));
    }
  }

  const props = { state, onChange: edit };
  const widget = [
    <CalcWidget key="calc" {...props} />,
    <GroupWidget key="group" {...props} />,
    <CompressWidget key="compress" {...props} />,
    <RopeWidget key="rope" {...props} />,
    <AbsorbWidget key="absorb" {...props} />,
    <CompareWidget key="compare" {...props} />,
  ][state.step];

  return (
    <AppShell
      active="kv"
      crumb={T('KV 架构对照')}
      indicator={T('浏览器内计算')}
      title={T('KV 架构对照')}
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            {Tx(
              '原书导读 {#0} 第 22、23、30 章 {#1} DeepSeek-V2 论文',
              <span>·</span>,
              <span>·</span>,
            )}
          </div>
          <h1>{Tx('KV 架构对照：缓存里{#0}', <span>{T('到底存什么')}</span>)}</h1>
          <p>
            {T(
              '生成时，每一层都要把历史 token 的 K 和 V 存起来，这就是 KV 缓存，它常常比你想的大得多。这一页用同一个公式算缓存大小：MHA、GQA、MQA 靠减少 KV 头来缩小它；MLA 换了一种思路，只存一个低维的 latent 和一个共享的 RoPE Key。所有数字都在浏览器里实际算出。',
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

      <WhyIntro why={kvWhy} />
      <section className="lesson-panel kv-panel" aria-label={T('KV 架构逐步实验')}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">
              {T('这一页想弄明白：每个 token 要在缓存里留下多少个数')}
            </span>
            <h2>{T('先列出缓存公式，再看四种 KV 设计分别改动了其中哪一项。')}</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {KV_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps kv-steps" aria-label={T('KV 架构步骤')}>
          {kvSteps.map((item, i) => (
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
        <div className="lesson-columns kv-columns">
          <div className="diagram-panel">
            <KVDiagram state={state} />
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
            {widget}
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
              disabled={state.step === KV_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === KV_STEP_COUNT - 1
                ? T('已到最后一步')
                : T('下一步：{#0} →', kvSteps[state.step + 1].name)}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                setPlaying(false);
                setState({ ...initialKVState(), view: state.view });
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
          <h2>{T('不看图，讲清楚四种设计各自缓存什么。')}</h2>
          <p>
            {T(
              '再解释：KV 缓存公式里有哪几项？GQA 减少了什么、没减少什么？MLA 的缓存为什么要加上 RoPE Key？矩阵吸收为什么不改变结果？',
            )}
          </p>
        </div>
        <button
          className="button secondary"
          aria-expanded={recallOpen}
          onClick={() => {
            setRecallOpen((v) => !v);
            setState((s) => ({ ...s, view: { ...s.view, explanations: false } }));
          }}
        >
          {recallOpen ? T('收起参考解释') : T('查看参考解释')}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            {T(
              'KV 缓存 = 批大小 × 层数 × token 数 × KV 头数 × (d_k + d_v) × 每个数的字节，只算 K、V。MHA 每个 Query 头各有一组 K、V；GQA 让同组的 Query 头共享一组，MQA 是只有一组的特例。它们减少的是 KV 头，Query 头数、W_Q、W_O 和打分次数不变。MLA 不减少头，而是缓存一个低维 latent，每个头的 K、V 用时由它还原。RoPE 若直接加在还原出的 K 上，位置相关的旋转会夹在 Query 和 latent 之间，无法吸收，所以 DeepSeek-V2 让位置信息单独走一支，共享的 RoPE Key 也要缓存。矩阵吸收利用结合律，把还原 K 的矩阵并进 Query、把还原 V 的矩阵并进输出投影，只改变计算顺序，不改变结果。',
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
              '所有数值由浏览器实时计算。玩具配置：4 个位置、d_model = 4、4 个 Query 头、每头 2 维；MLA 的 latent 为 3 维、RoPE Key 为 2 维（θ = 1）。权重由固定种子生成（取值 −1 到 1，步长 0.5），不是训练好的模型。 行向量记法 X W；DeepSeek-V2 论文用列向量 W h，公式已转写。省略了论文中的 Query 压缩（它用于减少训练时的激活内存，不影响 KV 缓存）；RoPE Query 因此直接由 h 投影。GQA 的共享 K/V 投影由组内取平均得到，没有继续训练。',
            )}
          </p>
          <p>
            {T(
              '缓存字节只统计 K、V（或 MLA 的 latent 与 RoPE Key），不含权重、激活和分配器开销。原书第 23 章与第 22 章的例子已由公式逐字节核对。DeepSeek-V2 的层数、头数、头维、d_c 与 d_h^R 取自论文 §3.1.2 与表 1 说明（d_c = 4 d_h，d_h^R = d_h / 2）。',
            )}
          </p>
          <strong className="exp-notes-title">{T('与原书说法的对照')}</strong>
          <ul>
            {kvBookNotes.map((note) => (
              <li key={note}>
                <Rich text={note} />
              </li>
            ))}
          </ul>
          <strong className="exp-notes-title">{T('常见误区')}</strong>
          <ul>
            {kvMyths.map((myth) => (
              <li key={myth}>
                <Rich text={myth} />
              </li>
            ))}
          </ul>
          <ul>
            {kvSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">{T('来源核对日期：2026-11-07。图形与说明为本项目编写。')}</p>
        </div>
      </details>
    </AppShell>
  );
}
