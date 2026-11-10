import { useState } from 'react';
import { Prediction, PredictionResult, Segment } from '../../components/Controls';
import { Matrix } from '../../components/Matrix';
import { formatNumber as fmt } from '../../math/attention';
import { computeFlash, type FlashResult } from '../../math/flash';
import type { FlashState } from '../../state/flash';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

export interface FlashProps {
  state: FlashState;
  result: FlashResult;
  onChange: (state: FlashState, preservePrediction?: boolean) => void;
}
export const vec = (values: number[]) => '[' + values.map((value) => fmt(value)).join(', ') + ']';
export const errorText = (value: number) => (value === 0 ? '0' : value.toExponential(2));
const sizes = [1, 2, 3, 4] as const;

export function FlashInputs({ state, onChange }: FlashProps) {
  return (
    <div className="flash-inputs">
      <Segment
        label={T('输入预设')}
        options={['rising', 'book', 'mixed'] as const}
        value={state.input.preset}
        onSelect={(preset) =>
          onChange({ ...state, input: { ...state.input, preset, keyDelta: 0, block: 0 } })
        }
        render={(preset) =>
          preset === 'rising'
            ? T('递增分数')
            : preset === 'book'
              ? T('书中 21.4.4 一行分数')
              : T('混合输入')
        }
      />
      <div className="flash-input-row">
        <Segment
          label={T('观察 Query')}
          options={[0, 1, 2, 3] as const}
          value={state.input.query}
          onSelect={(query) => onChange({ ...state, input: { ...state.input, query } })}
          render={(query) => 't' + query}
        />
        <label className="flash-field">
          {T('混合输入种子')}
          <input
            aria-label={T('混合输入种子')}
            disabled={state.input.preset !== 'mixed'}
            type="number"
            min={1}
            max={9999}
            value={state.seed}
            onChange={(e) => {
              const seed = Number(e.target.value);
              if (Number.isInteger(seed) && seed >= 1 && seed <= 9999) onChange({ ...state, seed });
            }}
          />
        </label>
      </div>
      <div className="flash-input-row">
        <Segment
          label={T('每块 Key 数 B')}
          options={sizes}
          value={state.config.blockSize}
          onSelect={(blockSize) =>
            onChange({
              ...state,
              config: { ...state.config, blockSize },
              input: { ...state.input, block: 0 },
            })
          }
          render={(size) => 'B = ' + size}
        />
        <label className="flash-mask">
          <input
            type="checkbox"
            checked={state.config.causal}
            onChange={(e) =>
              onChange({ ...state, config: { ...state.config, causal: e.target.checked } })
            }
          />
          {T('因果 Mask')}
        </label>
      </div>
      <p className="exp-caption">
        {Tx(
          '{#0} 当前 ΔK[3,0] = {#1}；在“重缩放”中先预测再修改。切块会从第 1 块重新查看。',
          state.config.causal
            ? T('每行只能读取自己与之前的位置。')
            : T('双向读取对照：每行可读取全部位置。'),
          fmt(state.input.keyDelta),
        )}
      </p>
    </div>
  );
}

function Baseline({ state, result }: FlashProps) {
  const t = state.input.query;
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('完整行 Attention 数值')}>
          <caption>{Tx('Query t{#0} · q = {#1} · Key 轴归一化', t, vec(result.q[t]))}</caption>
          <thead>
            <tr>
              <th scope="col">Key</th>
              <th scope="col">{T('分数 + Mask')}</th>
              <th scope="col">P</th>
              <th scope="col">V</th>
            </tr>
          </thead>
          <tbody>
            {result.k.map((_, j) => (
              <tr key={j}>
                <th scope="row">t{j}</th>
                <td>{fmt(result.reference.masked[t][j])}</td>
                <td>{fmt(result.reference.probabilities[t][j], 5)}</td>
                <td>{vec(result.v[j])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok">
        {Tx(
          '完整行输出 O[t{#0}] = {#1}。P 的每一项乘对应的 V，再逐维相加。',
          t,
          vec(result.reference.output[t]),
        )}
      </p>
      <details className="lesson-formula">
        <summary>{T('展开同一份 Q、K、V 激活')}</summary>
        <div className="flash-matrices">
          <Matrix name="Q" data={result.q} tone="query" row={t} />
          <Matrix name="K" data={result.k} tone="key" />
          <Matrix name="V" data={result.v} tone="value" />
        </div>
      </details>
    </div>
  );
}

function TileValues({ state, result }: FlashProps) {
  const block = result.rows[state.input.query][state.input.block];
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('当前 K V 块数值')}>
          <caption>
            {Tx(
              '第 {#0} 块 · K 与 V 按相同位置读取 · 实际 {#1} 行',
              block.index + 1,
              block.keys.length,
            )}
          </caption>
          <thead>
            <tr>
              <th scope="col">{T('位置')}</th>
              <th scope="col">K [1, 2]</th>
              <th scope="col">V [1, 2]</th>
              <th scope="col">{T('分数 + Mask')}</th>
            </tr>
          </thead>
          <tbody>
            {block.keys.map((j, offset) => (
              <tr key={j}>
                <th scope="row">t{j}</th>
                <td>{vec(result.k[j])}</td>
                <td>{vec(result.v[j])}</td>
                <td>{fmt(block.scores[offset])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict">
        {Tx(
          '当前有 {#0} 个有效 Key。{#1}',
          block.validCount,
          block.validCount === 0
            ? T('这一块全部被屏蔽，直接保留旧状态。')
            : T('局部分数仍需与此前块共用归一化尺度，不能把各块的 Softmax 直接平均。'),
        )}
      </p>
    </div>
  );
}

function Accumulator({ state, result }: FlashProps) {
  const b = result.rows[state.input.query][state.input.block];
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('在线累积量更新')}>
          <caption>{Tx('第 {#0} 块后的状态 · a 是未归一化的 2 维向量', b.index + 1)}</caption>
          <thead>
            <tr>
              <th scope="col">{T('状态')}</th>
              <th scope="col">{T('旧值')}</th>
              <th scope="col">{T('旧值 × α')}</th>
              <th scope="col">{T('本块贡献')}</th>
              <th scope="col">{T('新值')}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">{T('l 分母')}</th>
              <td>{fmt(b.previousL, 5)}</td>
              <td>{fmt(b.scaledPreviousL, 5)}</td>
              <td>{fmt(b.blockL, 5)}</td>
              <td>{fmt(b.l, 5)}</td>
            </tr>
            <tr>
              <th scope="row">{T('a 分子')}</th>
              <td>{vec(b.previousA)}</td>
              <td>{vec(b.scaledPreviousA)}</td>
              <td>{vec(b.blockA)}</td>
              <td>{vec(b.a)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="flash-equation">
        {Tx(
          'm：{#0} → {#1}；本块最大值 {#2}；α = {#3}',
          fmt(b.previousM),
          fmt(b.m),
          fmt(b.blockM),
          fmt(b.alpha, 6),
        )}
      </p>
      <p className="exp-caption">
        {Tx(
          '本块 exp(s − m′) = {#0}；已经读到 t{#1}。显示的小数不参与后续计算。',
          vec(b.exponentials),
          b.keys.at(-1),
        )}
      </p>
      <p className="exp-verdict" role="status">
        {Tx(
          '{#0} 当前 a / l = {#1}{#2}',
          b.validCount === 0
            ? T('全屏蔽块：α = 1，旧 m、l、a 保持不变。')
            : b.rescaled
              ? T('新最大值出现：旧 l 与旧 a 都已重缩放。')
              : b.previousL === 0
                ? T('首个有效块：旧累积为零，α = 0。')
                : T('最大值没有增大：α = 1，沿用原尺度。'),
          vec(b.output),
          b.index === result.rows[state.input.query].length - 1
            ? T('，已读完全部块。')
            : T('，这是前缀暂时输出。'),
        )}
      </p>
    </div>
  );
}

function RescaleExperiment(props: FlashProps) {
  const { state, result, onChange } = props;
  const [target, setTarget] = useState(state.input.keyDelta === 0 ? 2 : 0);
  const [choice, setChoice] = useState<'up' | 'same' | 'down' | null>(null);
  const [verdict, setVerdict] = useState<{
    before: number;
    after: number;
    alpha: number;
    correct: boolean;
  } | null>(null);
  const labels = { up: T('m 变大'), same: T('m 不变'), down: T('m 变小') };
  return (
    <>
      <Accumulator {...props} />
      <div className="exp-experiment">
        <strong>{T('只修改最后一个 Key，先判断最终最大值。')}</strong>
        <p className="exp-caption">
          {Tx(
            '当前 ΔK[3,0] = {#0}；Q、V 保持不变。被因果 Mask 排除的 Key 不参与最大值。',
            fmt(state.input.keyDelta),
          )}
        </p>
        <label className="exp-range">
          {T('待执行 ΔK[3,0]')}
          <input
            type="range"
            aria-label={T('待执行 Key 偏移')}
            min={-2}
            max={2}
            step={0.25}
            value={target}
            onChange={(e) => {
              onChange(state, true);
              setTarget(Number(e.target.value));
              setVerdict(null);
            }}
          />
          <output>{fmt(target)}</output>
        </label>
        <Prediction
          question={T('读完全部块后，m 会怎样变化？')}
          options={[
            { value: 'up', label: labels.up },
            { value: 'same', label: labels.same },
            { value: 'down', label: labels.down },
          ]}
          choice={choice}
          onChoose={(next) => {
            onChange(state, true);
            setChoice(next);
            setVerdict(null);
          }}
        />
        <button
          className="button secondary"
          disabled={!choice}
          onClick={() => {
            const next = {
              ...state,
              input: {
                ...state.input,
                keyDelta: target,
                block: result.rows[state.input.query].length - 1,
              },
            };
            const computed = computeFlash(next);
            const before = result.rows[state.input.query].at(-1)!.m;
            const after = computed.rows[state.input.query].at(-1)!;
            const direction =
              after.m > before + 1e-12 ? 'up' : after.m < before - 1e-12 ? 'down' : 'same';
            setVerdict({
              before,
              after: after.m,
              alpha: after.alpha,
              correct: choice === direction,
            });
            onChange(next, true);
          }}
        >
          {T('验证预测并执行 →')}
        </button>
        {verdict && (
          <p className={`exp-verdict ${verdict.correct ? 'ok' : ''}`} role="status">
            {Tx(
              '最终 m：{#0} → {#1}；最后一块 α = {#2}。',
              fmt(verdict.before),
              fmt(verdict.after),
              fmt(verdict.alpha, 6),
            )}
            <PredictionResult label={choice ? labels[choice] : null} holds={verdict.correct} />
          </p>
        )}
      </div>
    </>
  );
}

function Equality(props: FlashProps) {
  const { state, result, onChange } = props;
  const [target, setTarget] = useState<FlashState['config']['blockSize']>(
    state.config.blockSize === 1 ? 3 : 1,
  );
  const [choice, setChoice] = useState<'same' | 'different' | null>(null);
  const [verdict, setVerdict] = useState<{ error: number; correct: boolean } | null>(null);
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('分块与全量输出核对')}>
          <caption>{T('全部 Query、全部块完成后的输出 · [4, 2]')}</caption>
          <thead>
            <tr>
              <th scope="col">Query</th>
              <th scope="col">{T('全量 O')}</th>
              <th scope="col">{T('分块 O')}</th>
            </tr>
          </thead>
          <tbody>
            {result.output.map((output, t) => (
              <tr key={t} className={t === state.input.query ? 'exp-changed' : ''}>
                <th scope="row">t{t}</th>
                <td>{vec(result.reference.output[t])}</td>
                <td>{vec(output)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok">
        {Tx(
          '全部行最大绝对误差 = {#0}，容差 1e-12。数学等价，浮点结果无需逐比特相同。',
          errorText(result.maxError),
        )}
      </p>
      <div className="exp-experiment">
        <strong>{T('同一份输入，换一种切块。')}</strong>
        <Segment
          label={T('待执行块大小')}
          options={sizes}
          value={target}
          onSelect={(size) => {
            onChange(state, true);
            setTarget(size);
            setVerdict(null);
          }}
          render={(size) => T('切为 {#0}', size)}
        />
        <Prediction
          question={T('只改变块大小，最终 O 会变吗？')}
          options={[
            { value: 'same', label: T('容差内不变') },
            { value: 'different', label: T('输出会改变') },
          ]}
          choice={choice}
          onChoose={(next) => {
            onChange(state, true);
            setChoice(next);
            setVerdict(null);
          }}
        />
        <button
          className="button secondary"
          disabled={!choice}
          onClick={() => {
            const next = {
              ...state,
              config: { ...state.config, blockSize: target },
              input: { ...state.input, block: 0 },
            };
            const computed = computeFlash(next);
            const error = Math.max(
              ...computed.output.flatMap((row, t) =>
                row.map((value, d) => Math.abs(value - result.output[t][d])),
              ),
            );
            setVerdict({ error, correct: choice === (error <= 1e-12 ? 'same' : 'different') });
            onChange(next, true);
          }}
        >
          {T('验证分块预测 →')}
        </button>
        {verdict && (
          <p className={`exp-verdict ${verdict.correct ? 'ok' : ''}`} role="status">
            {Tx('切块前后最大差 = {#0}。', errorText(verdict.error))}
            <PredictionResult
              label={choice === 'same' ? T('容差内不变') : T('输出会改变')}
              holds={verdict.correct}
            />
          </p>
        )}
      </div>
    </div>
  );
}

function Ledger({ result }: FlashProps) {
  const l = result.ledger;
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('中间存储与 I O 账本')}>
          <caption>{Tx('教学调度 · T = 4，B = {#0} · 假设每元素 4 B', l.blockSize)}</caption>
          <thead>
            <tr>
              <th scope="col">{T('计数范围')}</th>
              <th scope="col">{T('朴素全量')}</th>
              <th scope="col">{T('逐行分块')}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">{T('完整 S / P 中间元素')}</th>
              <td>{l.naiveIntermediateElements}</td>
              <td>0</td>
            </tr>
            <tr>
              <th scope="row">{T('分块临时槽位')}</th>
              <td>{T('另计')}</td>
              <td>{l.tiledScratchElements}</td>
            </tr>
            <tr>
              <th scope="row">{T('仅 S / P 往返载荷')}</th>
              <td>{l.naiveIntermediateTransfers * l.bytesPerElement} B</td>
              <td>{l.tiledIntermediateTransfers} B</td>
            </tr>
            <tr>
              <th scope="row">{T('可见配对数')}</th>
              <td>{l.validPairs}</td>
              <td>{l.validPairs}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="exp-caption">
        {Tx(
          '矩形网格共 {#0} 对，Mask 后有 {#1} 对有效。这里数逻辑配对，不保证具体 GPU kernel 会跳过所有屏蔽算术。',
          l.pairCount,
          l.validPairs,
        )}
      </p>
      <p className="exp-verdict">
        {T(
          '0 B 只指完整 S / P 的写回与读回。Q / K / V / O 仍要搬运；这份计数不能推算总 HBM 流量、KV Cache 或运行时间。',
        )}
      </p>
      <details className="lesson-formula">
        <summary>{T('展开调度与计数口径')}</summary>
        <p className="exp-caption">
          {Tx(
            '朴素路径保存两张 T × T 中间矩阵，各写一次、读一次，共 4T² 个元素。分块路径固定一个 Query 行，保留 K/V 块、指数块、一个 Query、两份向量累积与三个标量，合计 B(d_k + d_v + 1) + d_k + 2d_v + 3 = {#0} 个临时槽位。此小例子不保证工作区更小。两列不含各自完整输入输出；实际内核会采用不同的寄存器复用与调度。本演示为核对还额外保存参考矩阵和过程快照。',
            l.tiledScratchElements,
          )}
        </p>
      </details>
    </div>
  );
}

export function FlashWidget(props: FlashProps) {
  if (props.state.step === 0) return <Baseline {...props} />;
  if (props.state.step === 1) return <TileValues {...props} />;
  if (props.state.step === 2) return <Accumulator {...props} />;
  if (props.state.step === 3) return <RescaleExperiment {...props} />;
  if (props.state.step === 4) return <Equality {...props} />;
  return <Ledger {...props} />;
}
