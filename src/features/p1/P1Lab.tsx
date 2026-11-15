import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Prediction, PredictionResult } from '../../components/Controls';
import { Rich } from '../../components/Tex';
import { StepSources, WhyIntro } from '../../components/WhyIntro';
import { formatNumber as fmt } from '../../math/attention';
import {
  error,
  randomValues,
  type Computation,
  type Frame,
  type Tensor,
} from '../../math/p1/shared';
import {
  explain,
  inputRange,
  lessonContext,
  verifiedDate,
  type Field,
  type TopicDefinition,
} from '../../content/p1/types';
import { initialLabState, restoreLabState, serializeLabState, type LabState } from '../../state/p1';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/experiment.css';
import '../../styles/p1.css';
import '../../styles/editorial.css';
import { hasMechanism, MechanismDiagram } from './MechanismDiagram';
const errorText = (value: number) => (value === 0 ? '0' : value.toExponential(2));
/** Name a flattened cell by its row and column labels instead of bare indices. */
function cellLabel(t: Tensor, i: number) {
  const width = t.values[0].length,
    row = Math.floor(i / width),
    column = t.columns?.[i % width] ?? `d${i % width}`;
  if (t.rows) return `${t.rows[row]} · ${column}`;
  return t.values.length > 1 ? T('行{#0} · {#1}', row, column) : column;
}
const cellText = (value: number) =>
  value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(2) : fmt(value, 5);
export default function P1Lab({ topic }: { topic: TopicDefinition }) {
  const [restored] = useState(() => restoreLabState(topic, window.location.hash)),
    [state, setState] = useState(restored.state),
    [playing, setPlaying] = useState(false),
    [notice, setNotice] = useState(restored.warning ?? ''),
    [manual, setManual] = useState(''),
    [revision, setRevision] = useState(0),
    [recall, setRecall] = useState(false),
    [reduced, setReduced] = useState(
      () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    );
  const result = topic.compute(state.input, state.config, state.seed),
    step = topic.steps[state.step],
    current = result.frames[state.step],
    context = lessonContext(state.input, state.config, current),
    [low, high] = inputRange(topic);
  function edit(next: LabState, preserve = false) {
    setPlaying(false);
    setState(next);
    setManual('');
    setNotice('');
    if (!preserve) setRevision((r) => r + 1);
  }
  function jump(step: number) {
    edit({ ...state, step: Math.max(0, Math.min(topic.steps.length - 1, step)) });
  }
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => ({ ...s, step: Math.min(topic.steps.length - 1, s.step + 1) })),
      8000 / state.view.speed,
    );
    return () => clearInterval(timer);
  }, [playing, state.view.speed, topic.steps.length]);
  useEffect(() => {
    if (state.step === topic.steps.length - 1) setPlaying(false);
  }, [state.step, topic.steps.length]);
  useEffect(() => {
    const pause = () => {
        if (document.hidden) setPlaying(false);
      },
      restore = () => {
        const next = restoreLabState(topic, location.hash);
        setPlaying(false);
        setState(next.state);
        setRevision((r) => r + 1);
        setRecall(false);
        setManual('');
        setNotice(next.warning ?? T('已从链接恢复实验。'));
      },
      media = window.matchMedia('(prefers-reduced-motion: reduce)'),
      motion = () => {
        setReduced(media.matches);
        setPlaying(false);
      };
    document.addEventListener('visibilitychange', pause);
    window.addEventListener('hashchange', restore);
    media.addEventListener('change', motion);
    return () => {
      document.removeEventListener('visibilitychange', pause);
      window.removeEventListener('hashchange', restore);
      media.removeEventListener('change', motion);
    };
  }, [topic]);
  async function share() {
    const url = new URL(localePath(topic.path), location.origin);
    url.hash = serializeLabState(topic, state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManual('');
      setNotice(T('已复制当前实验状态的链接。'));
    } catch {
      setManual(url.href);
      setNotice(T('剪贴板不可用，请手动复制下方链接。'));
    }
  }
  return (
    <AppShell
      active={topic.id}
      crumb={topic.title}
      indicator={T('浏览器内计算')}
      title={topic.title}
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">{Tx('原书导读 · {#0} · 可手算的机制实验', topic.chapter)}</div>
          <h1>{topic.title}</h1>
          <p>{topic.intro}</p>
          {topic.id === 'preference' && (
            <p>
              <a href={localePath('/notes#rlhf')}>
                {T('先读 RLHF 全流程：SFT、奖励模型、Rollout 与策略更新 →')}
              </a>
            </p>
          )}
        </div>
        <button className="button secondary share-button" onClick={() => void share()}>
          {T('↗ 复制实验链接')}
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
      {manual && (
        <label className="manual-share">
          {T('手动复制链接')}
          <input readOnly value={manual} onFocus={(e) => e.target.select()} />
        </label>
      )}
      <WhyIntro why={topic.why} />
      <section className="lesson-panel p1-panel" aria-label={T('{#0} 逐步实验', topic.id)}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">{topic.question}</span>
            <h2>{topic.headline}</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {topic.steps.length}
          </span>
        </div>
        <div className="lesson-steps p1-steps" aria-label={T('计算步骤')}>
          {topic.steps.map((s, i) => (
            <button
              key={s.name}
              aria-current={i === state.step ? 'step' : undefined}
              onClick={() => jump(i)}
            >
              <span>{i + 1}</span>
              {s.name}
            </button>
          ))}
        </div>
        <div className="p1-inputs">
          <div className="p1-presets" role="group" aria-label={T('预设示例')}>
            <span>{T('输入预设')}</span>
            {topic.examples.map((example) => (
              <button
                key={example.name}
                className={example.book ? 'p1-book-preset' : undefined}
                title={example.book ? T('数字取自原书，可与书中手算结果对照') : undefined}
                onClick={() => {
                  const initial = initialLabState(topic);
                  edit({
                    ...state,
                    input: example.input.slice(),
                    config: { ...initial.config, ...example.parameters },
                  });
                }}
              >
                {example.name}
              </button>
            ))}
            <label>
              {T('随机种子')}
              <input
                type="number"
                aria-label={T('随机种子')}
                min={1}
                max={9999}
                value={state.seed}
                onChange={(e) => {
                  const seed = e.target.valueAsNumber;
                  if (Number.isInteger(seed) && seed >= 1 && seed <= 9999) edit({ ...state, seed });
                }}
              />
            </label>
            <button
              onClick={() =>
                edit({ ...state, input: randomValues(state.seed, state.input.length) })
              }
            >
              {T('按种子生成输入')}
            </button>
          </div>
          <details className="p1-edit-input">
            <summary>
              {Tx('编辑输入向量 · {#0}', state.input.map((v) => fmt(v)).join(', '))}
            </summary>
            <div className="p1-vector-editor">
              {state.input.map((value, i) => (
                <label key={i}>
                  {topic.inputNames?.[i] ?? `x${i}`}
                  <input
                    aria-label={T('输入 x{#0}', i)}
                    type="number"
                    min={low}
                    max={high}
                    step={0.25}
                    value={value}
                    onChange={(e) => {
                      const v = e.target.valueAsNumber;
                      if (Number.isFinite(v) && v >= low && v <= high)
                        edit({ ...state, input: state.input.map((old, j) => (i === j ? v : old)) });
                    }}
                  />
                </label>
              ))}
            </div>
            <p className="exp-caption">
              {Tx(
                '范围 {#0} 到 {#1}，默认步长 0.25，允许输入小数；二维矩阵按行展平。小范围便于核对，不代表真实模型激活范围。',
                fmt(low),
                fmt(high),
              )}
            </p>
          </details>
          <div className="p1-fields">
            {topic.fields
              .filter((f) => !f.steps || f.steps.includes(state.step))
              .map((field) => (
                <ParameterField
                  key={field.key}
                  field={field}
                  value={state.config[field.key]}
                  onChange={(value) =>
                    edit({ ...state, config: { ...state.config, [field.key]: value } })
                  }
                />
              ))}
          </div>
        </div>
        <div className="lesson-columns p1-columns">
          <div className="diagram-panel">
            <p className="p1-figure-title">{Tx('图：{#0}', current.title)}</p>
            <MechanismDiagram frame={current} fallback={<VectorDiagram frame={current} />} />
            <div className="p1-metrics">
              {current.metrics.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{typeof value === 'number' ? fmt(value, 6) : value}</strong>
                </div>
              ))}
            </div>
            <p className="exp-caption">
              {hasMechanism(current)
                ? T(
                    '图中位置、连接或节点对应本步实际计算。数值表保留完整坐标；切步直接重建，不把插值当成算法状态。',
                  )
                : T(
                    '柱长对应本步实际数值，横线表示零；不同形状的张量分别展开坐标，不是语义空间降维。切步直接重建数值。',
                  )}
            </p>
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
                {state.view.explanations ? T('隐藏说明') : T('显示说明')}
              </button>
            </div>
            <h3>{step.title}</h3>
            {state.view.explanations ? (
              <p className="lesson-explanation">
                <Rich text={explain(step.explanation, context)} />
              </p>
            ) : (
              <div className="lesson-recall">
                <strong>{T('先自己说一遍')}</strong>
                <p>
                  <Rich text={step.prompt} />
                </p>
              </div>
            )}
            <div className="exp-widget">
              {current.tensors.map((t, i) => (
                <TensorTable key={i} tensor={t} />
              ))}
              {current.note && <p className="exp-verdict">{current.note}</p>}
            </div>
            {state.step === topic.steps.length - 1 && (
              <Experiment
                key={`${revision}:${state.step}`}
                topic={topic}
                state={state}
                result={result}
                onChange={edit}
              />
            )}
            <div className="lesson-takeaway">
              <strong>{T('这一步，记住一句话')}</strong>
              <p>
                <Rich
                  text={state.view.explanations ? step.takeaway : T('先自己复述，再显示说明核对。')}
                />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>{T('公式与符号说明')}</summary>
              <Formula expression={step.formula} symbols={step.symbols} />
            </details>
            <StepSources book={step.book} source={step.source} />
          </div>
        </div>
        <div className="lesson-controls">
          <div>
            <button
              className="button secondary"
              disabled={!state.step}
              onClick={() => jump(state.step - 1)}
            >
              {T('← 上一步')}
            </button>
            <button
              className="button primary"
              disabled={state.step === topic.steps.length - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === topic.steps.length - 1
                ? T('已到最后一步')
                : T('下一步：{#0} →', topic.steps[state.step + 1].name)}
            </button>
          </div>
          <div>
            <button onClick={() => jump(0)} className="lesson-text-button">
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                edit({ ...initialLabState(topic), view: state.view });
                setNotice(T('已恢复默认示例。'));
              }}
            >
              {T('恢复默认示例')}
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? T('暂停自动逐步') : T('播放自动逐步')}
              onClick={() => {
                if (!playing && state.step === topic.steps.length - 1)
                  setState((s) => ({ ...s, step: 0 }));
                setPlaying((v) => !v);
              }}
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
        {reduced && (
          <p className="lesson-motion">
            {T('已遵循减少动态效果偏好；图表直接切换，不做过渡动画。')}
          </p>
        )}
      </section>
      <details className="details-panel p1-checks">
        <summary>{Tx('数值性质核对 · {#0} 项', result.checks.length)}</summary>
        <div className="details-body">
          {result.checks.map((c) => (
            <p key={c.label} className={`exp-verdict ${c.error <= c.tolerance ? 'ok' : ''}`}>
              {Tx(
                '{#0}：误差 {#1}，容差 {#2} · {#3}',
                c.label,
                errorText(c.error),
                c.tolerance,
                c.error <= c.tolerance ? T('通过') : T('超出容差'),
              )}
            </p>
          ))}
          <p className="exp-caption">
            {T('仅核对当前列出的数学性质；不会把浏览器小例子当作真实模型质量或硬件性能测试。')}
          </p>
        </div>
      </details>
      <section className="recall-card">
        <div>
          <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
          <h2>{Tx('合上说明，把这 {#0} 步从头讲一遍。', topic.steps.length)}</h2>
          <p>
            {Tx(
              '用自己的话串起{#0}：每一步为什么需要？改动一个输入，会影响后面哪些数？哪些结论只在特定条件下才成立？',
              topic.steps.map((s) => s.name).join(T('、')),
            )}
          </p>
        </div>
        <button
          className="button secondary"
          aria-expanded={recall}
          onClick={() => {
            setPlaying(false);
            setRecall((v) => !v);
            setState((s) => ({ ...s, view: { ...s.view, explanations: false } }));
          }}
        >
          {' '}
          {recall ? T('收起参考解释') : T('查看参考解释')}
        </button>
        {recall && (
          <div className="recall-answer">
            {topic.steps.map((s) => (
              <p key={s.name}>{Tx('{#0}：{#1}', s.name, <Rich text={s.takeaway} />)}</p>
            ))}
          </div>
        )}
      </section>
      <details className="details-panel source-panel">
        <summary>
          <span>{T('适用条件与来源')}</span>
          <span className="summary-hint">{T('简化假设 · 常见误区 · 来源日期')}</span>
        </summary>
        <div className="details-body">
          <p>
            {T(
              '全部数值在浏览器中实际计算，内部使用 JavaScript float64，显示时才舍入。随机输入与掩码可由种子重现；修改输入或参数会暂停播放并重建步骤。没有模型服务、真实模型权重或 GPU 性能实测。',
            )}
          </p>
          <ul>
            {topic.limits.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
          {topic.bookNotes && (
            <>
              <strong className="exp-notes-title">{T('与原书说法的对照')}</strong>
              <ul>
                {topic.bookNotes.map((note) => (
                  <li key={note}>
                    <Rich text={note} />
                  </li>
                ))}
              </ul>
            </>
          )}
          <ul>
            {topic.sources.map((s) => (
              <li key={s.url}>
                <a href={s.url} target="_blank" rel="noreferrer">
                  {s.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">
            {Tx('来源核对日期：{#0}。解释、图形和教学数值由本项目编写。', verifiedDate)}
          </p>
        </div>
      </details>
    </AppShell>
  );
}
function ParameterField({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="p1-field">
      {field.label}
      {field.options ? (
        <select
          aria-label={field.label}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        >
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <>
          <input
            type="range"
            aria-label={field.label}
            min={field.min}
            max={field.max}
            step={field.step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
          />
          <output>{fmt(value, 6)}</output>
        </>
      )}
    </label>
  );
}
function TensorTable({ tensor: t }: { tensor: Tensor }) {
  return (
    <div className="exp-table-wrap">
      <table className="exp-table" aria-label={t.name}>
        <caption>
          {Tx('{#0} · [{#1}, {#2}] · 行 ↓ 列 →', t.name, t.values.length, t.values[0].length)}
        </caption>
        <thead>
          <tr>
            <th scope="col">{T('坐标')}</th>
            {t.values[0].map((_, i) => (
              <th scope="col" key={i}>
                {t.columns?.[i] ?? `d${i}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {t.values.map((row, i) => (
            <tr key={i}>
              <th scope="row">
                {t.rows?.[i] ?? (t.values.length > 1 ? T('行 {#0}', i) : T('值'))}
              </th>
              {row.map((v, j) => (
                <td key={j}>{cellText(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function VectorDiagram({ frame: f }: { frame: Frame }) {
  const tensors = f.tensors.length === 1 ? f.tensors : [f.tensors[0], f.tensors.at(-1)!],
    lengths = tensors.map((t) => t.values.flat().length),
    height = 80 + lengths.reduce((s, n) => s + Math.min(n, 12) * 22 + 52, 0),
    limit = Math.max(
      1,
      ...tensors.flatMap((t) => t.values.flat().filter(Number.isFinite).map(Math.abs)),
    );
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 430 ${height}`}
      role="img"
      aria-label={T('{#0}：{#1}', f.title, tensors.map((t) => t.name).join(T('，')))}
    >
      <title>{T('本步实际数值对照，不同张量分别展开')}</title>
      {tensors.map((t, g) => {
        const values = t.values.flat(),
          offset = 30 + lengths.slice(0, g).reduce((s, n) => s + Math.min(n, 12) * 22 + 52, 0);
        return (
          <g key={g}>
            <text x="12" y={offset} className="p1-svg-title">
              {t.name}
            </text>
            {values.slice(0, 12).map((v, i) => {
              const finite = Number.isFinite(v),
                width = finite ? (Math.abs(v) / limit) * 130 : 0;
              return (
                <g key={i}>
                  <text x="12" y={offset + 26 + i * 22} className="p1-svg-small">
                    {cellLabel(t, i)}
                  </text>
                  <line
                    x1="210"
                    x2="210"
                    y1={offset + 11 + i * 22}
                    y2={offset + 29 + i * 22}
                    stroke="#9ba89a"
                  />
                  <rect
                    x={v < 0 ? 210 - width : 210}
                    y={offset + 14 + i * 22}
                    width={width}
                    height="12"
                    className={g ? 'p1-bar-output' : 'p1-bar-input'}
                  />
                  <text x="355" y={offset + 25 + i * 22} className="p1-svg-number">
                    {fmt(v, 4)}
                  </text>
                </g>
              );
            })}
            {values.length > 12 && (
              <text x="12" y={offset + 26 + 12 * 22} className="p1-svg-small">
                {T('其余数值见右侧表格')}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
function Experiment({
  topic,
  state,
  result,
  onChange,
}: {
  topic: TopicDefinition;
  state: LabState;
  result: Computation;
  onChange: (s: LabState, preserve?: boolean) => void;
}) {
  const field = topic.fields.find((f) => f.key === topic.prediction.key)!;
  const [target, setTarget] = useState(
      topic.prediction.target === state.config[field.key] ? field.value : topic.prediction.target,
    ),
    [choice, setChoice] = useState<'same' | 'different' | null>(null),
    [verdict, setVerdict] = useState<{
      difference: number;
      correct: boolean;
      label: string;
    } | null>(null);
  return (
    <div className="exp-experiment">
      <strong>{T('先预测，再执行对照')}</strong>
      <p className="exp-caption">
        {Tx(
          '核对量：{#0}。只有下方待执行参数会改变，当前输入与其他参数保持一致。',
          topic.prediction.label,
        )}
      </p>
      <ParameterField
        field={{ ...field, label: T('待执行：{#0}', field.label) }}
        value={target}
        onChange={(v) => {
          onChange(state, true);
          setTarget(v);
          setVerdict(null);
        }}
      />
      <Prediction
        question={topic.prediction.question}
        options={[
          { value: 'same', label: T('容差内不变') },
          { value: 'different', label: T('会改变') },
        ]}
        choice={choice}
        onChoose={(c) => {
          onChange(state, true);
          setChoice(c);
          setVerdict(null);
        }}
      />
      <button
        className="button secondary"
        disabled={!choice}
        onClick={() => {
          const next = { ...state, config: { ...state.config, [field.key]: target } },
            computed = topic.compute(next.input, next.config, next.seed),
            difference = error(result.signature, computed.signature);
          setVerdict({
            difference,
            correct: choice === (difference <= 1e-10 ? 'same' : 'different'),
            label: choice === 'same' ? T('容差内不变') : T('会改变'),
          });
          onChange(next, true);
        }}
      >
        {T('验证预测并执行 →')}
      </button>
      {verdict && (
        <p className={`exp-verdict ${verdict.correct ? 'ok' : ''}`} role="status">
          {Tx(
            '{#0}最大绝对变化 = {#1}，判等容差 1e-10。',
            topic.prediction.label,
            errorText(verdict.difference),
          )}
          <PredictionResult label={verdict.label} holds={verdict.correct} />
        </p>
      )}
    </div>
  );
}
