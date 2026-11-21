import { useEffect, useState } from 'react';
import { Formula } from '../../components/Formula';
import { Prediction } from '../../components/Controls';

import { gradientTrajectory } from '../../math/notebook';
import { formatNumber as fmt } from '../../math/attention';
import { initialNotebookState, type NotebookState } from '../../state/notebook';
import type { Parameters } from '../../math/p1/shared';
import '../../styles/training-tools.css';
import { T, localePath } from '../../i18n';
import { Tx } from '../../i18n/jsx';

interface ToolProps {
  state: NotebookState;
  onChange: (state: NotebookState) => void;
}

/** The trajectory tool shares the experiment's validated configuration and URL. */
export default function TrainingTools({
  config,
  onChange,
}: {
  config: Parameters;
  onChange: (config: Parameters) => void;
}) {
  const state = initialNotebookState();
  state.config = { ...state.config, eta: config.trajectoryEta, schedule: config.schedule === 1 };
  const change = (next: NotebookState) =>
    onChange({ ...config, trajectoryEta: next.config.eta, schedule: Number(next.config.schedule) });
  return (
    <>
      <details>
        <summary>{T('学习率轨迹与 Warmup')}</summary>
        <LearningRateTool state={state} onChange={change} />
      </details>
      <p>
        <a href={localePath('/model-budget')}>
          {T('整个模型训练一遍要多少计算？打开参数量与计算量 →')}
        </a>
      </p>
    </>
  );
}

export function LearningRateTool({ state, onChange }: ToolProps) {
  const path = gradientTrajectory(state.config.eta, 12, state.config.schedule);
  const last = path[path.length - 1];
  const [prediction, setPrediction] = useState<string | null>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    setShown(false);
    setPrediction(null);
  }, [state.config.eta, state.config.schedule]);
  return (
    <section className="note-tool" aria-label={T('学习率轨迹实验')}>
      <h3>{T('同一条梯度，换一种步长')}</h3>
      <Formula
        expression={String.raw`L(w)=\frac12w^2,\quad w_0=2,\quad w_{t+1}=w_t-\eta_t w_t`}
        symbols={[
          [String.raw`w_t`, T('第 t 次更新前的标量参数')],
          [String.raw`\eta_t`, T('第 t 次更新使用的学习率')],
        ]}
      />
      <div className="chapter-preset-buttons">
        {[0.5, 2, 2.5].map((eta) => (
          <button
            key={eta}
            onClick={() =>
              onChange({ ...state, config: { ...state.config, eta, schedule: false } })
            }
          >
            η={eta}
          </button>
        ))}
      </div>
      <label className="note-slider">
        η = {fmt(state.config.eta, 2)}
        <input
          aria-label={T('基础学习率 η')}
          type="range"
          min="0.05"
          max="2.5"
          step="0.05"
          value={state.config.eta}
          onChange={(e) =>
            onChange({ ...state, config: { ...state.config, eta: Number(e.target.value) } })
          }
        />
      </label>
      <label>
        <input
          type="checkbox"
          checked={state.config.schedule}
          onChange={(e) =>
            onChange({ ...state, config: { ...state.config, schedule: e.target.checked } })
          }
        />
        {T('3 次线性 Warmup，随后余弦衰减')}
      </label>
      <Prediction
        question={T('12 次更新后，loss 比初始值 2 更小吗？')}
        choice={prediction}
        onChoose={setPrediction}
        options={[
          { value: 'yes', label: T('更小') },
          { value: 'no', label: T('不更小') },
        ]}
      />
      <button disabled={!prediction} onClick={() => setShown(true)}>
        {T('执行 12 次更新')}
      </button>
      {shown && (
        <div className="note-result" data-testid="learning-result">
          <p role="status">
            {Tx(
              '最终 w={#0}，loss={#1}。{#2}',
              fmt(last.w, 5),
              fmt(last.loss, 5),
              (prediction === 'yes') === last.loss < 2 - 1e-12
                ? T('预测与计算一致。')
                : T('预测与计算不同，请核对每一步的步长。'),
            )}
          </p>
          <div className="note-table-scroll">
            <table>
              <caption>{T('实际更新轨迹；最后一行是结果，不再执行更新')}</caption>
              <thead>
                <tr>
                  {['t', 'w_t', 'L(w_t)', '∇L', 'η_t'].map((name) => (
                    <th scope="col" key={name}>
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {path.map((row) => (
                  <tr key={row.step}>
                    <th scope="row">{row.step}</th>
                    <td>{fmt(row.w, 5)}</td>
                    <td>{fmt(row.loss, 5)}</td>
                    <td>{fmt(row.gradient, 5)}</td>
                    <td>{row.step === 12 ? '—' : fmt(row.rate, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
