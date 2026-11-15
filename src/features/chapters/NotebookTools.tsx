import { useEffect, useState } from 'react';
import { Formula } from '../../components/Formula';
import { Prediction } from '../../components/Controls';
import { RLHF_STAGES } from '../../content/notebook';
import { gradientTrajectory, tallyAnswers, trainingBudget } from '../../math/notebook';
import { formatNumber as fmt } from '../../math/attention';
import type { NotebookState } from '../../state/notebook';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

interface ToolProps {
  state: NotebookState;
  onChange: (state: NotebookState) => void;
}

export function RLHFProcess({ state, onChange }: ToolProps) {
  const stage = RLHF_STAGES[state.step];
  return (
    <section className="note-tool" aria-label={T('RLHF 分阶段流程')}>
      <h3>{T('沿着一条数据走完 RLHF')}</h3>
      <p className="note-tool-caption">
        {T('可选择任一阶段，核对数据去向与参数更新；这里展示流程，不执行模型训练。')}
      </p>
      <ol className="note-pipeline">
        {RLHF_STAGES.map((item, i) => (
          <li key={i}>
            <button aria-pressed={i === state.step} onClick={() => onChange({ ...state, step: i })}>
              <span aria-hidden="true">{i + 1}</span>
              {item.name}
            </button>
          </li>
        ))}
      </ol>
      <div className="note-stage" aria-live="polite">
        <h4>{stage.name}</h4>
        <p>{stage.explanation}</p>
        <p className="note-data">
          <code>{stage.data}</code>
        </p>
        <strong>{stage.updates}</strong>
      </div>
      <div className="chapter-preset-buttons">
        <button
          disabled={state.step === 0}
          onClick={() => onChange({ ...state, step: state.step - 1 })}
        >
          {T('上一步')}
        </button>
        <output aria-label={T('RLHF 当前阶段')}>
          {state.step + 1} / {RLHF_STAGES.length}
        </output>
        <button
          disabled={state.step === 6}
          onClick={() => onChange({ ...state, step: state.step + 1 })}
        >
          {T('下一步')}
        </button>
        <button onClick={() => onChange({ ...state, step: 0 })}>{T('重置')}</button>
      </div>
    </section>
  );
}

export function PromptBuilder({ state, onChange }: ToolProps) {
  const i = state.input;
  const parts = [
    i.role ? T('角色：你是一位核对资料的阅读助手。') : '',
    T('任务：从给定资料中提取活动名称与日期。缺少日期时返回 null。'),
    i.examples >= 1
      ? T('示例 1：资料“读书会，11 月 20 日”。输出 {"event":"读书会","date":"11-20"}。')
      : '',
    i.examples >= 2
      ? T('示例 2：资料“徒步活动，时间待定”。输出 {"event":"徒步活动","date":null}。')
      : '',
    i.evidence
      ? T('<资料>科学讲座安排在 12 月 8 日。</资料>')
      : T('<资料>科学讲座，日期未提供。</资料>'),
    i.format ? T('输出约束：只输出一个 JSON 对象，键为 event 与 date；不要添加其他文字。') : '',
  ].filter(Boolean);
  return (
    <section className="note-tool" aria-label={T('提示结构拼装器')}>
      <h3>{T('动手拼一份可核对的提示')}</h3>
      <div className="note-tool-controls">
        {(
          [
            ['role', T('加入角色')],
            ['evidence', T('提供明确日期资料')],
            ['format', T('约定 JSON 输出')],
          ] as const
        ).map(([key, label]) => (
          <label key={key}>
            <input
              type="checkbox"
              checked={i[key]}
              onChange={(e) => onChange({ ...state, input: { ...i, [key]: e.target.checked } })}
            />
            {label}
          </label>
        ))}
        <label>
          {T('示例数量')}
          <select
            value={i.examples}
            onChange={(e) =>
              onChange({ ...state, input: { ...i, examples: Number(e.target.value) } })
            }
          >
            <option value="0">Zero-shot</option>
            <option value="1">One-shot</option>
            <option value="2">Few-shot</option>
          </select>
        </label>
      </div>
      <pre className="note-prompt" data-testid="composed-prompt">
        {parts.join('\n\n')}
      </pre>
      <p className="note-tool-caption">
        {T(
          '这是实际拼出的输入，未调用模型。新增示例或格式约束能否改善回答，需要在真实任务集上验证。',
        )}
      </p>
      <button
        onClick={() =>
          onChange({
            ...state,
            input: { ...i, examples: 0, role: false, evidence: false, format: false },
          })
        }
      >
        {T('恢复预设示例')}
      </button>
    </section>
  );
}

export function VoteTool({ state, onChange }: ToolProps) {
  const [choice, setChoice] = useState<string | null>(null);
  const [shown, setShown] = useState(false);
  const result = tallyAnswers(state.input.answers, state.config.samples);
  useEffect(() => {
    setChoice(null);
    setShown(false);
  }, [state.input.answers, state.config.samples]);
  const preset = (answers: NotebookState['input']['answers']) =>
    onChange({
      ...state,
      input: { ...state.input, answers },
      config: { ...state.config, samples: 5 },
    });
  return (
    <section className="note-tool" aria-label={T('Self-Consistency 投票实验')}>
      <h3>{T('同一题：17 − 3 × 4 = ?')}</h3>
      <p>
        {T(
          '候选为手工给定答案，null 表示提取失败。实际手算正确答案是 5；先看候选，再预测聚合结果。',
        )}
      </p>
      <div className="chapter-preset-buttons">
        <button onClick={() => preset([5, 5, 3, 5, 3])}>{T('恢复预设示例')}</button>
        <button onClick={() => preset([3, 3, 5, 3, 5])}>{T('对照：多数答错')}</button>
        <button onClick={() => preset([5, 3, null, null, null])}>
          {T('对照：平局与提取失败')}
        </button>
      </div>
      <div className="note-votes">
        {state.input.answers.map((answer, index) => (
          <label key={index}>
            {Tx('候选 {#0}', index + 1)}
            <select
              aria-label={T('候选 {#0}', index + 1)}
              value={answer ?? 'null'}
              disabled={index >= state.config.samples}
              onChange={(e) =>
                onChange({
                  ...state,
                  input: {
                    ...state.input,
                    answers: state.input.answers.map((a, n) =>
                      n === index
                        ? e.target.value === 'null'
                          ? null
                          : (Number(e.target.value) as 3 | 5 | 7)
                        : a,
                    ),
                  },
                })
              }
            >
              {[3, 5, 7].map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
              <option value="null">{T('提取失败')}</option>
            </select>
          </label>
        ))}
      </div>
      <label className="note-slider">
        {Tx('使用前 {#0} 个候选', state.config.samples)}
        <input
          aria-label={T('投票候选数')}
          type="range"
          min="1"
          max="5"
          step="1"
          value={state.config.samples}
          onChange={(e) =>
            onChange({ ...state, config: { ...state.config, samples: Number(e.target.value) } })
          }
        />
      </label>
      <Prediction
        question={T('哪个答案获得最多票？')}
        choice={choice}
        onChoose={setChoice}
        options={[
          { value: '3', label: '3' },
          { value: '5', label: '5' },
          { value: '7', label: '7' },
          { value: 'tie', label: T('无法选出唯一答案') },
        ]}
      />
      <button disabled={!choice} onClick={() => setShown(true)}>
        {T('计算投票')}
      </button>
      {shown && (
        <div className="note-result" role="status" data-testid="vote-result">
          <strong>
            {result.winner === null ? T('无法选出唯一答案') : T('投票答案：{#0}', result.winner)}
          </strong>
          <p>
            {Tx(
              '有效答案 {#0}，提取失败 {#1}；最高票数占有效答案 {#2}。',
              result.valid,
              result.invalid,
              `${fmt(result.share * 100, 1)}%`,
            )}
          </p>
          <p>{result.ranked.map(([answer, count]) => `${answer}: ${count}`).join(' · ')}</p>
          <p>
            {choice === (result.winner === null ? 'tie' : String(result.winner))
              ? T('预测与计算一致。')
              : T('预测与计算不同，请重新数一遍有效答案。')}
          </p>
          <p>{T('票数占比不是正确率；这道题可用独立算术核对。')}</p>
        </div>
      )}
    </section>
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

export function ScalingTool({ state, onChange }: ToolProps) {
  const budget = trainingBudget(state.config.parametersB, state.config.tokensB);
  return (
    <section className="note-tool" aria-label={T('训练计算量估算器')}>
      <h3>{T('只计算明确口径下的训练账本')}</h3>
      <Formula
        expression={String.raw`C\approx6ND`}
        symbols={[
          [String.raw`C`, T('估算训练浮点运算次数，单位 FLOPs')],
          ['N', T('稠密模型参数数量')],
          ['D', T('训练中处理的 token 总数（含重复）')],
        ]}
      />
      <div className="note-tool-controls">
        <label>
          {T('参数 N（十亿）')}
          <input
            type="number"
            min="0.1"
            max="1000"
            step="0.1"
            value={state.config.parametersB}
            onChange={(e) => {
              if (e.currentTarget.validity.valid && e.target.value !== '')
                onChange({
                  ...state,
                  config: { ...state.config, parametersB: Number(e.target.value) },
                });
            }}
          />
        </label>
        <label>
          {T('训练 token D（十亿）')}
          <input
            type="number"
            min="1"
            max="100000"
            step="1"
            value={state.config.tokensB}
            onChange={(e) => {
              if (e.currentTarget.validity.valid && e.target.value !== '')
                onChange({
                  ...state,
                  config: { ...state.config, tokensB: Number(e.target.value) },
                });
            }}
          />
        </label>
      </div>
      <div className="note-result" aria-live="polite" data-testid="training-budget">
        <strong>C ≈ {budget.flops.toExponential(3)} FLOPs</strong>
        <p>D/N = {fmt(budget.tokensPerParameter, 2)}</p>
      </div>
      <p>
        {T('只统计稠密模型主要训练矩阵运算的粗估，不含硬件效率、通信或费用；不据此预测 loss。')}
      </p>
      <button
        onClick={() =>
          onChange({ ...state, config: { ...state.config, parametersB: 7, tokensB: 140 } })
        }
      >
        {T('恢复预设示例')}
      </button>
    </section>
  );
}
