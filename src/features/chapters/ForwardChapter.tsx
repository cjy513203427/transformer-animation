import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { RelatedNotes } from '../../components/RelatedNotes';
import { Formula } from '../../components/Formula';
import { LinkedCode } from '../../components/LinkedCode';
import { Matrix } from '../../components/Matrix';
import { Prediction } from '../../components/Controls';
import { ShareState } from '../../components/ShareState';
import { MODEL_PY_CODE, MODEL_PY_HIGHLIGHTS } from '../../content/codeViews';
import { FORWARD_CONTENT } from '../../content/forward';
import { chapterUrl } from '../../content/book';
import { TOPICS } from '../../content/topics';
import {
  decoderForward,
  DECODER_VOCAB,
  type ForwardFrame,
  type ForwardTensor,
} from '../../math/forward';
import { formatNumber as fmt } from '../../math/attention';
import {
  initialForwardState,
  restoreForwardState,
  serializeForwardState,
  type ForwardState,
} from '../../state/forward';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/lesson.css';
import '../../styles/experiment.css';

const frameTitle = (frame: ForwardFrame): string =>
  frame.layer
    ? T('Block {#0} · {#1}', frame.layer, FORWARD_CONTENT[frame.kind].title)
    : FORWARD_CONTENT[frame.kind].title;
const vector = (values: number[]) => '[' + values.map((v) => fmt(v, 5)).join(', ') + ']';

export default function ForwardChapter() {
  const [restored] = useState(() => restoreForwardState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [playing, setPlaying] = useState(false);
  const [notice, setNotice] = useState(restored.invalid ? T('分享状态无效，已恢复默认示例。') : '');
  const [cell, setCell] = useState<{ name: string; row: number; column: number } | null>(null);
  const [prediction, setPrediction] = useState<'same' | 'change' | null>(null);
  const [comparison, setComparison] = useState<{
    before: number[];
    after: number[];
    holds: boolean;
  } | null>(null);
  const result = useMemo(
    () => decoderForward(state.input.tokens, state.config.depth, state.config.causal, state.seed),
    [state.input.tokens, state.config.depth, state.config.causal, state.seed],
  );
  const frame = result.frames[state.step],
    content = FORWARD_CONTENT[frame.kind];
  const related = TOPICS.find((topic) => topic.id === content.topic)!;
  const change = (next: ForwardState) => {
    setPlaying(false);
    setState(next);
    setCell(null);
    setPrediction(null);
    setComparison(null);
  };
  const jump = (step: number) => {
    setPlaying(false);
    setCell(null);
    setState((s) => ({ ...s, step: Math.max(0, Math.min(result.frames.length - 1, step)) }));
  };
  useEffect(() => {
    if (!playing) return;
    if (state.step === result.frames.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setCell(null);
      setState((s) => ({ ...s, step: Math.min(s.step + 1, result.frames.length - 1) }));
    }, 1800 / state.view.speed);
    return () => window.clearInterval(timer);
  }, [playing, state.step, state.view.speed, result.frames.length]);
  useEffect(() => {
    const stop = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      if (window.location.hash === '#scope') return;
      const next = restoreForwardState(window.location.hash);
      change(next.state);
      setNotice(next.invalid ? T('分享状态无效，已恢复默认示例。') : '');
    };
    if (window.location.hash === '#scope')
      requestAnimationFrame(() => {
        const scope = document.getElementById('scope');
        if (scope instanceof HTMLDetailsElement) {
          scope.open = true;
          scope.scrollIntoView();
        }
      });
    document.addEventListener('visibilitychange', stop);
    window.addEventListener('hashchange', restore);
    return () => {
      document.removeEventListener('visibilitychange', stop);
      window.removeEventListener('hashchange', restore);
    };
  }, []);
  function runComparison() {
    if (!prediction) return;
    const tokens = [...state.input.tokens];
    tokens[2] = (tokens[2] + 1) % 4;
    const after = decoderForward(tokens, state.config.depth, state.config.causal, state.seed)
      .probabilities[0];
    const before = result.probabilities[0];
    setComparison({
      before,
      after,
      holds: Math.max(...before.map((p, i) => Math.abs(p - after[i]))) <= 1e-12,
    });
    setPlaying(false);
    setCell(null);
    setState((s) => ({ ...s, input: { tokens }, step: 0 }));
  }
  function showTensor(item: ForwardTensor, parameter = false) {
    const vocabulary = ['logits', 'p', 'p_last', 'E^T'].includes(item.name);
    const attention = /^P_/.test(item.name);
    const selected = cell?.name === item.name ? cell : undefined;
    return (
      <Matrix
        key={item.name}
        name={item.name}
        data={item.data}
        parameter={parameter}
        tokenColumns={attention}
        tone={
          attention || item.name === 'p'
            ? 'probability'
            : item.name === 'Q'
              ? 'query'
              : item.name === 'K'
                ? 'key'
                : item.name === 'V'
                  ? 'value'
                  : 'neutral'
        }
        row={
          parameter
            ? undefined
            : (selected?.row ?? (frame.kind === 'selection' ? 0 : state.view.query))
        }
        column={selected?.column}
        columnLabels={vocabulary ? DECODER_VOCAB : item.name === 'token_ids' ? ['ID'] : undefined}
        rowLabels={
          item.name === 'E'
            ? DECODER_VOCAB
            : frame.kind === 'selection'
              ? [`t${state.input.tokens.length - 1}`]
              : undefined
        }
        axesLabel={
          vocabulary
            ? parameter
              ? T('特征 ↓ 词表 →')
              : T('位置 ↓ 词表 →')
            : item.name === 'E'
              ? T('词表 ↓ 特征 →')
              : undefined
        }
        onSelect={
          parameter
            ? undefined
            : (row, column) => {
                setPlaying(false);
                setCell({ name: item.name, row, column });
              }
        }
      />
    );
  }
  const selectedTensor = cell && frame.tensors.find((t) => t.name === cell.name);
  const stages = [
    { label: T('输入与位置'), index: 0 },
    ...Array.from({ length: state.config.depth }, (_, i) => ({
      label: `Block ${i + 1}`,
      index: 3 + 8 * i,
    })),
    { label: T('最终归一化'), index: 3 + 8 * state.config.depth },
    { label: T('词表与选择'), index: 4 + 8 * state.config.depth },
  ];
  return (
    <AppShell
      active="forward"
      title={T('完整前向传播')}
      crumb={T('完整前向传播')}
      indicator={T('浏览器内计算')}
    >
      <article className="chapter-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('完整前向 · 把零件连接成一次计算')}</div>
            <h1>{T('从三个 token，走到下一个 token')}</h1>
            <p>
              {T(
                '查表、位置、两个 Pre-Norm Block、最终归一化、词表投影与选择，全部来自同一次数值前向。每一步都可查看公式、符号、形状和矩阵。',
              )}
            </p>
          </div>
          <ShareState getHash={() => serializeForwardState(state)} />
        </header>
        {notice && (
          <p className="chapter-notice" role="status">
            {notice}
          </p>
        )}
        <section className="forward-lab" aria-label={T('完整前向计算台')}>
          <div className="chapter-control-panel">
            <fieldset>
              <legend>{T('输入 token（人为分词）')}</legend>
              <div className="chapter-inline-controls">
                {state.input.tokens.map((id, t) => (
                  <label key={t}>
                    {`t${t}`}
                    <select
                      aria-label={T('位置 {#0} 的 token', t)}
                      value={id}
                      onChange={(e) =>
                        change({
                          ...state,
                          step: 0,
                          input: {
                            tokens: state.input.tokens.map((v, i) =>
                              i === t ? Number(e.target.value) : v,
                            ),
                          },
                        })
                      }
                    >
                      {DECODER_VOCAB.map((word, id) => (
                        <option key={id} value={id}>
                          {word} · ID {id}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="chapter-inline-controls">
              <label>
                {T('Block 数')}
                <select
                  value={state.config.depth}
                  onChange={(e) =>
                    change({
                      ...state,
                      step: 0,
                      config: { ...state.config, depth: Number(e.target.value) as 1 | 2 },
                    })
                  }
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                </select>
              </label>
              <label>
                {T('教学参数种子')}
                <input
                  type="number"
                  min={0}
                  max={4294967295}
                  value={state.seed}
                  onChange={(e) => {
                    const seed = Number(e.target.value);
                    if (Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff)
                      change({ ...state, seed, step: 0 });
                  }}
                />
              </label>
              <label className="chapter-checkbox">
                <input
                  type="checkbox"
                  checked={state.config.causal}
                  onChange={(e) =>
                    change({
                      ...state,
                      step: 0,
                      config: { ...state.config, causal: e.target.checked },
                    })
                  }
                />
                {T('启用因果 Mask')}
              </label>
              <label>
                {T('观察位置')}
                <select
                  value={state.view.query}
                  onChange={(e) =>
                    change({ ...state, view: { ...state.view, query: Number(e.target.value) } })
                  }
                >
                  {state.input.tokens.map((_, i) => (
                    <option key={i} value={i}>{`t${i}`}</option>
                  ))}
                </select>
              </label>
            </div>
            <p>
              {T(
                '词表 4 · 隐藏维 4 · 两个头（每头 2 维）· FFN 中间维 6。a–d 无语义，结果不能衡量真实模型能力。',
              )}
            </p>
            {!state.config.causal && (
              <p className="chapter-notice">
                {T('当前是双向读取对照，不是自回归语言模型的合法生成设置。')}
              </p>
            )}
          </div>
          <ol className="forward-pipeline" aria-label={T('前向计算主线')}>
            {stages.map((stage, i) => (
              <li key={stage.index}>
                <button
                  aria-pressed={
                    state.step >= stage.index &&
                    (i === stages.length - 1 || state.step < stages[i + 1].index)
                  }
                  onClick={() => jump(stage.index)}
                >
                  {stage.label}
                  <span aria-hidden="true"> →</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="forward-controls">
            <button
              className="button secondary"
              disabled={state.step === 0}
              onClick={() => jump(state.step - 1)}
            >
              {T('上一步')}
            </button>
            <button
              className="button primary"
              onClick={() => {
                if (state.step === result.frames.length - 1) setState((s) => ({ ...s, step: 0 }));
                setPlaying((v) => !v);
              }}
            >
              {playing ? T('暂停') : T('播放')}
            </button>
            <button
              className="button secondary"
              disabled={state.step === result.frames.length - 1}
              onClick={() => jump(state.step + 1)}
            >
              {T('下一步')}
            </button>
            <button className="button secondary" onClick={() => change(initialForwardState())}>
              {T('恢复预设并重置')}
            </button>
            <label>
              {T('速度')}
              <select
                aria-label={T('前向回放速度')}
                value={state.view.speed}
                onChange={(e) =>
                  change({ ...state, view: { ...state.view, speed: Number(e.target.value) } })
                }
              >
                {[0.5, 1, 2].map((v) => (
                  <option key={v} value={v}>
                    {v}×
                  </option>
                ))}
              </select>
            </label>
            <label className="forward-jump">
              {T('直接跳步')}
              <select
                aria-label={T('前向计算步骤')}
                value={state.step}
                onChange={(e) => jump(Number(e.target.value))}
              >
                {result.frames.map((f, i) => (
                  <option key={f.id} value={i}>
                    {i + 1}. {frameTitle(f)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="forward-current">
            <div className="forward-step-heading">
              <div>
                <span className="eyebrow" data-testid="forward-progress">
                  {state.step + 1} / {result.frames.length}
                </span>
                <h2>{frameTitle(frame)}</h2>
              </div>
              <button
                className="chapter-text-button"
                onClick={() =>
                  setState((s) => ({
                    ...s,
                    view: { ...s.view, explanations: !s.view.explanations },
                  }))
                }
              >
                {state.view.explanations ? T('隐藏说明') : T('显示说明')}
              </button>
            </div>
            <p>{state.view.explanations ? content.explanation : content.prompt}</p>
            <div className="forward-tensors">{frame.tensors.map((item) => showTensor(item))}</div>
            {cell && selectedTensor && (
              <p className="chapter-cell-inspector" role="status">
                {Tx(
                  '选中 {#0}[{#1},{#2}] = {#3}（显示时舍入）',
                  cell.name,
                  cell.row,
                  cell.column,
                  fmt(selectedTensor.data[cell.row][cell.column], 6),
                )}
              </p>
            )}
            {frame.kind === 'selection' && (
              <p className="forward-selected" data-testid="forward-selected">
                {Tx(
                  '最后位置选择：{#0}（ID {#1}）',
                  DECODER_VOCAB[result.selected],
                  result.selected,
                )}
              </p>
            )}
            <Formula
              expression={content.formula}
              symbols={content.symbols}
              showSymbols={state.view.explanations}
            />
            <LinkedCode spec={MODEL_PY_CODE} highlight={MODEL_PY_HIGHLIGHTS[frame.kind]} />
            {frame.parameters && (
              <details className="chapter-parameters">
                <summary>{T('查看本步固定参数')}</summary>
                <div className="forward-tensors">
                  {frame.parameters.map((item) => showTensor(item, true))}
                </div>
              </details>
            )}
            <div className="chapter-related">
              <strong>{T('试着复述')}</strong>
              <p>{content.prompt}</p>
              <a href={related.href}>{Tx('单独复习：{#0} →', related.title)}</a>
            </div>
          </div>
        </section>
        <section className="chapter-section" aria-labelledby="forward-counterfactual">
          <h2 id="forward-counterfactual">{T('对照：修改未来输入，前面会变吗？')}</h2>
          <Prediction
            question={T('只替换 t2 的 token，t0 最终的词表分布会变化吗？')}
            options={[
              { value: 'same' as const, label: T('不变') },
              { value: 'change' as const, label: T('会变化') },
            ]}
            choice={prediction}
            onChoose={setPrediction}
          />
          <button className="button primary" disabled={!prediction} onClick={runComparison}>
            {T('替换 t2 并核对整条前向')}
          </button>
          {comparison && (
            <div className="chapter-comparison" role="status">
              <p>
                {(prediction === 'same') === comparison.holds
                  ? T('预测与计算一致。')
                  : T('预测与计算不同，请核对 Mask 的作用。')}
              </p>
              <p>
                {T('修改前 t0 词表概率：')}
                <span data-testid="forward-before">{vector(comparison.before)}</span>
              </p>
              <p>
                {T('修改后 t0 词表概率：')}
                <span data-testid="forward-after">{vector(comparison.after)}</span>
              </p>
              <p>
                {comparison.holds
                  ? T('因果约束经过所有 Block 仍成立：未来输入没有进入 t0 的计算。')
                  : T('双向读取允许 t0 访问 t2；这一变化会沿残差和后续 Block 传到输出。')}
              </p>
            </div>
          )}
        </section>
        <RelatedNotes path="/forward-pass" />
        <details id="scope" className="chapter-section">
          <summary>{T('设定、简化与来源')}</summary>
          <p>
            {T(
              '本例采用 Pre-Norm Decoder 顺序、正弦位置编码、tanh 近似 GELU 与绑定的输出权重。它不是 GPT-2 权重或完整配置：省略 batch、偏置、可训练的 Norm 仿射参数、Dropout、反向传播和优化器；不运行真实分词器。',
            )}
          </p>
          <p>
            {T(
              '所有内部运算使用未舍入的 JavaScript float64；界面显示舍入值。改变种子会生成另一组教学参数，播放不会更新参数。',
            )}
          </p>
          <ul>
            <li>
              <a
                href={chapterUrl('第15章-Transformer完整前向传播-从输入到输出/')}
                target="_blank"
                rel="noreferrer"
              >
                {T('原书第 15 章 · 完整前向传播')}
              </a>
            </li>
            <li>
              <a
                href={chapterUrl('第18章-手写Model.py-模型定义/')}
                target="_blank"
                rel="noreferrer"
              >
                {T('原书第 18 章 · Pre-Norm 教学模型')}
              </a>
            </li>
            <li>
              <a
                href="https://github.com/karpathy/minGPT/blob/master/mingpt/model.py"
                target="_blank"
                rel="noreferrer"
              >
                minGPT · Block / NewGELU
              </a>
            </li>
            <li>
              <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">
                Attention Is All You Need · §3.2
              </a>
            </li>
          </ul>
          <p>{T('来源核对：2026-11-12。计算与图形为本站教学实现。')}</p>
        </details>
      </article>
    </AppShell>
  );
}
