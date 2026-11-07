import { useEffect, useMemo, useRef, useState } from 'react';
import { Formula } from '../../components/Formula';
import { Tex } from '../../components/Tex';
import { computeExperiment, formatNumber as fmt } from '../../math/attention';
import { presetState, type ExperimentState } from '../../state/experiment';
import '../../styles/guided-attention.css';

interface Props {
  state: ExperimentState;
  onState: (next: ExperimentState) => void;
  onFinish: () => void;
  onExit: () => void;
}

const words = ['我', '喜欢', '学习'];
const goals = ['选一个位置', '先预测', '改一个量', '核对变化'];
const vector = (values: number[]) => `[${values.map((value) => fmt(value)).join(', ')}]`;

/** A short real experiment. Only executed numerical state is passed back to the lab. */
export function GuidedAttention({ state, onState, onFinish, onExit }: Props) {
  const [stage, setStage] = useState(0);
  const [prediction, setPrediction] = useState<'same' | 'change' | null>(null);
  const [draftDelta, setDraftDelta] = useState(0);
  const [selectionHint, setSelectionHint] = useState('');
  const headingRef = useRef<HTMLHeadingElement>(null);
  const { query } = state.view;
  const result = useMemo(
    () =>
      computeExperiment(
        state.input.x,
        state.input.keyDelta,
        state.input.valueDelta,
        state.config.causal,
        state.config.scaled,
      ),
    [state.input, state.config],
  );
  const baseline = useMemo(
    () =>
      computeExperiment(
        state.input.x,
        state.input.keyDelta,
        0,
        state.config.causal,
        state.config.scaled,
      ),
    [state.input.x, state.input.keyDelta, state.config],
  );
  const probabilityDifference = Math.max(
    ...result.probabilities.flatMap((row, i) =>
      row.map((value, j) => Math.abs(value - baseline.probabilities[i][j])),
    ),
  );
  const outputDifference = result.output[query][0] - baseline.output[query][0];

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, [stage]);

  function selectQuery(nextQuery: number) {
    const selected = nextQuery === 1;
    onState({
      ...state,
      step: selected ? 3 : 0,
      view: { ...state.view, query: nextQuery, lesson: 'numbers' },
    });
    if (selected) {
      setSelectionHint('');
      setStage(1);
    } else {
      setSelectionHint(`当前在观察“${words[nextQuery]}”。这次先点“喜欢”，让它读取前两个位置。`);
    }
  }

  function predict(choice: 'same' | 'change') {
    setPrediction(choice);
    onState({ ...state, step: 4, view: { ...state.view, lesson: 'numbers' } });
    setStage(2);
  }

  function verify() {
    if (draftDelta === 0 || prediction === null) return;
    onState({
      ...state,
      input: { ...state.input, valueDelta: draftDelta },
      step: 4,
      view: { ...state.view, lesson: 'numbers' },
    });
    setStage(3);
  }

  function back() {
    const nextStage = Math.max(0, stage - 1);
    if (nextStage < 2) setPrediction(null);
    setSelectionHint('');
    // Reopening a task clears its old computed verdict; a draft is never an executed input.
    onState({
      ...state,
      input: { ...state.input, valueDelta: 0 },
      step: nextStage === 0 ? 0 : nextStage === 1 ? 3 : 4,
      view: { ...state.view, query: nextStage === 0 ? 0 : 1, lesson: 'numbers' },
    });
    setStage(nextStage);
  }

  function restart() {
    const next = presetState('basic');
    next.view.query = 0;
    onState(next);
    setPrediction(null);
    setDraftDelta(0);
    setSelectionHint('');
    setStage(0);
    headingRef.current?.focus({ preventScroll: true });
  }

  return (
    <section className="guided-attention" aria-label="Attention 操作引导">
      <header className="guided-heading">
        <div>
          <span className="section-kicker">跟着做一次 · 只改变 V</span>
          <h2>改动内容，会改变注意力权重吗？</h2>
          <p>用三个位置做一次小实验。完成后，带着这组结果继续探索。</p>
        </div>
        <button className="guided-exit" onClick={onExit}>
          退出引导，自由探索
        </button>
      </header>

      <ol className="guided-progress" aria-label="引导进度">
        {goals.map((goal, i) => (
          <li key={goal} aria-current={stage === i ? 'step' : undefined}>
            <span aria-hidden="true">{i < stage ? '✓' : i + 1}</span>
            {goal}
          </li>
        ))}
      </ol>

      <div className="guided-layout">
        <div className="guided-task">
          <div className="guided-task-position">
            <span>当前任务</span>
            <output className="guided-progress-count" aria-label="当前引导步骤">
              {stage + 1} / 4
            </output>
          </div>
          <h3 ref={headingRef} tabIndex={-1}>
            {
              [
                '点一下“喜欢”，选择观察位置',
                '先猜一猜：P 会不会变？',
                '拖动 V 的偏移，再验证预测',
                '对照结果，解释变化',
              ][stage]
            }
          </h3>

          {stage === 0 && (
            <>
              <p>三个词块分别对应 t0、t1、t2。这次让“喜欢”作为 Query，观察它取用哪些位置的信息。</p>
              <div
                className="guided-target guided-token-picker"
                role="group"
                aria-label="引导：选择观察位置"
              >
                {words.map((word, i) => (
                  <button
                    key={word}
                    aria-label={`观察“${word}”`}
                    aria-pressed={query === i}
                    onClick={() => selectQuery(i)}
                  >
                    <small>t{i}</small>
                    <strong>{word}</strong>
                    <span>{query === i ? '当前观察' : '点选观察'}</span>
                  </button>
                ))}
              </div>
              <p className="guided-hint" role="status">
                {selectionHint || '请点“喜欢”，“当前已计算结果”会随观察位置更新。'}
              </p>
            </>
          )}

          {stage === 1 && (
            <>
              <p>
                保持 Q、K 和因果 Mask 不变，只改投影后的 <Tex tex="V_{1,0}" />
                。你认为取用比例 P 会怎样？
              </p>
              <div
                className="guided-target guided-prediction"
                role="group"
                aria-label="引导：预测注意力权重变化"
              >
                <button onClick={() => predict('same')}>P 不会变</button>
                <button onClick={() => predict('change')}>P 会变</button>
              </div>
              <p className="guided-hint">选一个预测后，亲手改动 V，再看实际计算。</p>
            </>
          )}

          {stage === 2 && (
            <>
              <p>
                你的预测：<strong>{prediction === 'same' ? 'P 不会变' : 'P 会变'}</strong>。
                拖动滑杆，给“喜欢”提供的内容设置一个非零偏移。
              </p>
              <div className="guided-target guided-input">
                <div className="guided-range-label">
                  <label htmlFor="guided-value-delta">引导：V[1,0] 的偏移 ΔV</label>
                  <output htmlFor="guided-value-delta">{fmt(draftDelta)}</output>
                </div>
                <input
                  id="guided-value-delta"
                  type="range"
                  min="-4"
                  max="4"
                  step="0.5"
                  value={draftDelta}
                  onChange={(event) => setDraftDelta(Number(event.target.value))}
                  aria-describedby="guided-draft-note"
                />
                <div className="guided-range-scale" aria-hidden="true">
                  <span>−4</span>
                  <span>0</span>
                  <span>+4</span>
                </div>
                <p id="guided-draft-note" className="guided-hint" role="status">
                  {draftDelta === 0
                    ? '请选择一个非零偏移，才能比较改动前后的结果。'
                    : `待验证：V[1,0] 将从 ${fmt(baseline.v[1][0])} 变为 ${fmt(baseline.v[1][0] + draftDelta)}。图中仍是上次已计算结果。`}
                </p>
                <button className="button primary" disabled={draftDelta === 0} onClick={verify}>
                  验证预测
                </button>
              </div>
              <p className="guided-hint">
                范围限制在 [−4, 4]，方便手算；Q、K 和参数矩阵 W 保持固定。
              </p>
            </>
          )}

          {stage === 3 && (
            <>
              <p className="guided-verdict" role="status">
                {prediction === 'same' ? '预测正确。' : '实际结果和你的预测不同。'}
                {probabilityDifference <= 1e-12
                  ? ' P 保持不变，O 发生变化。'
                  : ' 请对照下方实际数值。'}
              </p>
              <div className="guided-comparison">
                <table aria-label="引导：P 与 O 的前后对照">
                  <thead>
                    <tr>
                      <th scope="col">观察 t{query}</th>
                      <th scope="col">改动前</th>
                      <th scope="col">改动后</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th scope="row">权重 P[{query}]</th>
                      <td data-testid="guided-probability-before">
                        {vector(baseline.probabilities[query])}
                      </td>
                      <td data-testid="guided-probability-after">
                        {vector(result.probabilities[query])}
                      </td>
                    </tr>
                    <tr>
                      <th scope="row">输出 O[{query}]</th>
                      <td data-testid="guided-output-before">{vector(baseline.output[query])}</td>
                      <td data-testid="guided-output-after" className="guided-changed">
                        {vector(result.output[query])}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p>
                P 只由 Q、K 和 Mask 决定。V 提供被取出的内容，因此 O[{query},0] 变化了{' '}
                <strong>{fmt(outputDifference)}</strong>；第二个输出分量不变。
              </p>
              <Formula
                expression={`\\Delta O_{${query},0}=P_{${query},1}\\,\\Delta V=${fmt(result.probabilities[query][1])}\\times(${fmt(state.input.valueDelta)})\\approx ${fmt(outputDifference)}`}
              />
              <div className="guided-recall">
                <strong>试着复述</strong>
                <p>为什么“取多少”没有变，“取出的内容”却变了？在自由探索中，试试改 K 作对照。</p>
              </div>
              <button className="button primary" onClick={onFinish}>
                保留结果，自由探索
              </button>
            </>
          )}
        </div>

        <figure className="guided-figure">
          <figcaption>
            <strong>当前已计算结果</strong>
            <span>
              观察 t{query}“{words[query]}”
            </span>
          </figcaption>
          <svg
            viewBox="0 0 520 205"
            role="img"
            aria-label="固定 Q 和 K 计算权重 P，再用 P 对 V 加权得到输出 O"
          >
            <defs>
              <marker
                id="guided-arrow"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0 0 L10 5 L0 10Z" fill="#7c8a7b" />
              </marker>
            </defs>
            <path
              d="M147 56H191M332 56H386M263 139H443V83"
              className="guided-wire"
              markerEnd="url(#guided-arrow)"
            />
            <rect x="18" y="28" width="129" height="55" rx="3" className="guided-node-fixed" />
            <text x="82" y="51" textAnchor="middle">
              Q、K 固定
            </text>
            <text x="82" y="70" textAnchor="middle" className="guided-svg-small">
              匹配与打分
            </text>
            <rect
              x="193"
              y="28"
              width="139"
              height="55"
              rx="3"
              className="guided-node-probability"
            />
            <text x="262" y="51" textAnchor="middle">
              权重 P[{query}]
            </text>
            <text x="262" y="70" textAnchor="middle" className="guided-svg-small">
              从哪里取多少
            </text>
            <rect x="387" y="28" width="116" height="55" rx="3" className="guided-node-output" />
            <text x="445" y="51" textAnchor="middle">
              输出 O[{query}]
            </text>
            <text x="445" y="70" textAnchor="middle" className="guided-svg-small">
              加权求和
            </text>
            <rect
              x="193"
              y="112"
              width="139"
              height="55"
              rx="3"
              className={
                stage === 3 ? 'guided-node-value guided-node-changed' : 'guided-node-value'
              }
            />
            <text x="262" y="135" textAnchor="middle">
              内容 V
            </text>
            <text x="262" y="154" textAnchor="middle" className="guided-svg-small">
              这次只改这一项
            </text>
            <text x="385" y="186" textAnchor="middle" className="guided-svg-small">
              P 决定比例，V 提供内容
            </text>
          </svg>
          <div className="guided-weight-row">
            <strong>取用比例 P[{query}]</strong>
            {result.probabilities[query].map((value, j) => (
              <div key={j}>
                <span>
                  t{j} {words[j]}
                </span>
                <strong>{(value * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
          <div className="guided-values">
            <div>
              <span>已计算的 V[1,0]</span>
              <strong>{fmt(result.v[1][0])}</strong>
              <small>
                偏移 ΔV ={' '}
                <output data-testid="guided-applied-delta">{fmt(state.input.valueDelta)}</output>
              </small>
            </div>
            <div>
              <span>输出 O[{query}]</span>
              <strong>{vector(result.output[query])}</strong>
              <small>二维向量 · 非词表概率</small>
            </div>
          </div>
          <p className="guided-figure-note">
            教学示例：3 个位置、2 维行向量。因果 Mask 开启；固定 Q/K，只在投影后干预
            V。显示时舍入，计算保留精度。
          </p>
        </figure>
      </div>

      <footer className="guided-controls">
        <button className="button secondary" onClick={back} disabled={stage === 0}>
          ← 上一步
        </button>
        <button className="guided-restart" onClick={restart}>
          从头再做
        </button>
        <span>从头再做会恢复这组教学示例。</span>
      </footer>
    </section>
  );
}
