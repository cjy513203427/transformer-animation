import { useState } from 'react';
import { Prediction, PredictionResult, Segment } from '../../components/Controls';
import { formatNumber as fmt, type Matrix } from '../../math/attention';
import { computeMoE, directReference, D_MODEL, D_FF, EXPERTS } from '../../math/moe';
import type { MoEState } from '../../state/moe';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

type Result = ReturnType<typeof computeMoE>;
interface Props {
  state: MoEState;
  result: Result;
  onChange: (state: MoEState, preservePrediction?: boolean) => void;
}
const tokens = [0, 1, 2, 3] as const;
const vec = (values: number[]) => '[' + values.map((v) => fmt(v)).join(', ') + ']';
const pct = (value: number) => fmt(value * 100, 1) + '%';
const difference = (a: Matrix, b: Matrix) =>
  Math.max(...a.flatMap((row, i) => row.map((value, j) => Math.abs(value - b[i][j]))));
const errorText = (value: number) => (value === 0 ? '0' : value.toExponential(1));

function Tensor({
  name,
  data,
  rowPrefix,
  colPrefix,
  selected,
}: {
  name: string;
  data: Matrix;
  rowPrefix: string;
  colPrefix: string;
  selected?: number;
}) {
  return (
    <div className="exp-table-wrap">
      <table className="exp-table" aria-label={name}>
        <caption>{Tx('{#0} · [{#1}, {#2}]，行 ↓ 列 →', name, data.length, data[0].length)}</caption>
        <thead>
          <tr>
            <th scope="col">{T('轴')}</th>
            {data[0].map((_, i) => (
              <th scope="col" key={i}>
                {colPrefix}
                {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className={selected === i ? 'exp-changed' : ''}>
              <th scope="row">
                {rowPrefix}
                {i}
              </th>
              {row.map((value, j) => (
                <td key={j}>{fmt(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MoEInputs({ state, result, onChange }: Props) {
  const setInput = (patch: Partial<MoEState['input']>) =>
    onChange({ ...state, input: { ...state.input, ...patch } });
  return (
    <div className="moe-inputs">
      <Segment
        label={T('输入预设')}
        options={['balanced', 'concentrated'] as const}
        value={state.input.preset}
        onSelect={(preset) => setInput({ preset, delta: 0, expertScale: 1 })}
        render={(value) => (value === 'balanced' ? T('分散输入') : T('集中输入'))}
      />
      <div className="moe-input-row">
        <Segment
          label={T('观察 token')}
          options={tokens}
          value={state.input.token}
          onSelect={(token) => setInput({ token, delta: 0 })}
          render={(t) => 't' + t}
        />
        <label className="moe-field">
          {T('权重种子')}
          <input
            type="number"
            aria-label={T('权重种子')}
            min={1}
            max={9999}
            step={1}
            value={state.seed}
            onChange={(event) => {
              const seed = Number(event.target.value);
              if (Number.isInteger(seed) && seed >= 1 && seed <= 9999) onChange({ ...state, seed });
            }}
          />
        </label>
      </div>
      <label className="exp-range">
        {Tx('修改 t{#0} 的 x₀：偏移 Δ', state.input.token)}
        <input
          aria-label={T('输入 x0 偏移')}
          type="range"
          min={-2}
          max={2}
          step={0.25}
          value={state.input.delta}
          onChange={(e) => setInput({ delta: Number(e.target.value) })}
        />
        <output>{fmt(state.input.delta)}</output>
        <span>{Tx('当前 x = {#0}', vec(result.x[state.input.token]))}</span>
      </label>
      <p className="exp-caption">
        {T('范围 −2 到 2，步长 0.25，便于手算；切换观察 token 会清除偏移。所有步骤共用这份输入。')}
      </p>
    </div>
  );
}

function RouterWidget({ state, result }: Props) {
  const t = state.input.token;
  return (
    <div className="exp-widget">
      <p className="moe-equation">
        x_{t} = {vec(result.x[t])} → z_{t} = {vec(result.logits[t])}
      </p>
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('Router 打分与概率')}>
          <caption>{Tx('观察 t{#0}，Softmax 沿专家轴计算，尚未做 Top-K', t)}</caption>
          <thead>
            <tr>
              <th scope="col">{T('专家')}</th>
              <th scope="col">{T('W_R 的列')}</th>
              <th scope="col">{T('分数 z')}</th>
              <th scope="col">{T('概率 p')}</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((e) => (
              <tr key={e}>
                <th scope="row">E{e}</th>
                <td>{vec(result.weights.router.map((row) => row[e]))}</td>
                <td>{fmt(result.logits[t][e])}</td>
                <td>{pct(result.probabilities[t][e])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict">
        {T('每个分数是 x 与 W_R 对应列的点积；概率是路由信号，还不是专家的计算结果。')}
      </p>
      <details className="lesson-formula">
        <summary>{T('展开全部输入与 Router 参数')}</summary>
        <Tensor name={T('输入 X')} data={result.x} rowPrefix="t" colPrefix="d" selected={t} />
        <Tensor
          name={T('Router 权重 W_R')}
          data={result.weights.router}
          rowPrefix="d"
          colPrefix="E"
        />
      </details>
    </div>
  );
}

function GateWidget({ state, result, onChange }: Props) {
  const t = state.input.token;
  return (
    <div className="exp-widget">
      <Segment
        label={T('选择专家数 K')}
        options={[1, 2] as const}
        value={state.config.topK}
        onSelect={(topK) => onChange({ ...state, config: { ...state.config, topK } })}
        render={(k) => 'Top-' + k}
      />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('Top-K 与归一化门控')}>
          <caption>
            {Tx(
              't{#0} 选中：{#1}（同分按专家编号）',
              t,
              result.selected[t].map((e) => 'E' + e).join(T('、')),
            )}
          </caption>
          <thead>
            <tr>
              <th scope="col">{T('专家')}</th>
              <th scope="col">{T('原概率 p')}</th>
              <th scope="col">{T('选中')}</th>
              <th scope="col">{T('门值 g')}</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((e) => (
              <tr key={e} className={result.selected[t].includes(e) ? 'exp-changed' : ''}>
                <th scope="row">E{e}</th>
                <td>{pct(result.probabilities[t][e])}</td>
                <td>{result.selected[t].includes(e) ? T('是') : T('否')}</td>
                <td>{fmt(result.gates[t][e])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok">
        {Tx(
          '门值之和 = {#0}。{#1}',
          fmt(
            result.gates[t].reduce((a, b) => a + b, 0),
            12,
          ),
          state.config.topK === 1
            ? T(' 本例 Top-1 重新归一化后门值为 1。')
            : T(' 未选中的门值为 0；两个选中专家共同参与输出。'),
        )}
      </p>
      <p className="exp-caption">
        {T(
          '本例使用确定性 Top-K，无噪声与容量限制。选中后重新归一化是一种具体门控规则，不是所有 MoE 的统一做法。',
        )}
      </p>
    </div>
  );
}

type Guess = 'same' | 'change';
const options: { value: Guess; label: string }[] = [
  { value: 'same', label: T('路由不变') },
  { value: 'change', label: T('路由改变') },
];

function ExpertWidget({ state, result, onChange }: Props) {
  const t = state.input.token;
  const [expert, setExpert] = useState(result.selected[t][0]);
  const packet = result.packets.find((p) => p.token === t && p.expert === expert)!;
  const [pending, setPending] = useState(state.input.expertScale === 1 ? 2 : 1);
  const [choice, setChoice] = useState<Guess | null>(null);
  const [verdict, setVerdict] = useState<{
    gates: number;
    output: number;
    choice: Guess | null;
  } | null>(null);
  return (
    <div className="exp-widget">
      <Segment
        label={T('查看选中专家')}
        options={result.selected[t]}
        value={expert}
        onSelect={(next) => {
          onChange(state, true);
          setExpert(next);
        }}
        render={(e) => 'E' + e}
      />
      <p className="moe-equation">
        {Tx(
          't{#0} → E{#1}：{#2} → {#3} → ReLU → {#4} → {#5}',
          t,
          expert,
          vec(packet.input),
          vec(packet.preactivation),
          vec(packet.hidden),
          vec(packet.output),
        )}
      </p>
      <p className="exp-caption">
        {Tx(
          '形状 [1, 2] × [2, 3] → [1, 3]，ReLU 把负数截为 0；再乘 [3, 2] 得到 [1, 2]。 当前 E0 输出倍率为 {#0}，作用在第二层 FFN 之后。',
          fmt(state.input.expertScale),
        )}
      </p>
      <details className="lesson-formula">
        <summary>{Tx('展开专家 E{#0} 的两份权重', expert)}</summary>
        <Tensor
          name={'E' + expert + ' W1'}
          data={result.weights.w1[expert]}
          rowPrefix="d"
          colPrefix="f"
        />
        <Tensor
          name={'E' + expert + ' W2'}
          data={result.weights.w2[expert]}
          rowPrefix="f"
          colPrefix="d"
        />
      </details>
      <div className="exp-experiment">
        <strong>{T('对照：只改变 E0 的输出')}</strong>
        <label className="exp-range">
          {T('待执行倍率 α')}
          <input
            type="range"
            aria-label={T('E0 待执行输出倍率')}
            min={0}
            max={2}
            step={0.25}
            value={pending}
            onChange={(e) => {
              onChange(state, true);
              setPending(Number(e.target.value));
              setVerdict(null);
            }}
          />
          <output>{fmt(pending)}</output>
        </label>
        <Prediction
          question={T('只改 E0 输出倍率，这次前向的路由会变吗？')}
          options={options}
          choice={choice}
          onChoose={(guess) => {
            onChange(state, true);
            setChoice(guess);
            setVerdict(null);
          }}
        />
        <button
          className="button secondary"
          disabled={!choice}
          onClick={() => {
            const next = { ...state, input: { ...state.input, expertScale: pending } };
            const after = computeMoE(next);
            setVerdict({
              gates: difference(result.gates, after.gates),
              output: difference(result.output, after.output),
              choice,
            });
            onChange(next, true);
          }}
        >
          {T('验证预测并执行 →')}
        </button>
        {verdict && (
          <p className="exp-verdict ok" role="status">
            {Tx(
              '全部 token 的门控最大差 = {#0}；输出最大差 = {#1}。',
              errorText(verdict.gates),
              errorText(verdict.output),
            )}
            <PredictionResult
              label={options.find((o) => o.value === verdict.choice)?.label ?? null}
              holds={verdict.choice === 'same'}
            />
            {T('没选中 E0 的 token 不受这次干预影响。')}
          </p>
        )}
        <p className="exp-caption">
          {T('这是对专家输出激活的人为干预，范围 0–2；不更新 Router 或专家参数。')}
        </p>
      </div>
    </div>
  );
}

function CombineWidget({ state, result }: Props) {
  const t = state.input.token;
  const ref = directReference(state);
  return (
    <div className="exp-widget">
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('专家输出的加权合并')}>
          <caption>{Tx('保留 t{#0} 的身份，逐项相加', t)}</caption>
          <thead>
            <tr>
              <th scope="col">{T('专家')}</th>
              <th scope="col">{T('输出 f_e(x)')}</th>
              <th scope="col">g</th>
              <th scope="col">{T('贡献 g·f_e(x)')}</th>
            </tr>
          </thead>
          <tbody>
            {result.selected[t].map((e) => (
              <tr key={e}>
                <th scope="row">E{e}</th>
                <td>{vec(result.expertOutputs[t][e]!)}</td>
                <td>{fmt(result.gates[t][e])}</td>
                <td>{vec(result.contributions[t][e]!)}</td>
              </tr>
            ))}
            <tr className="exp-total">
              <th scope="row">{Tx('合并 y_{#0}', t)}</th>
              <td colSpan={3}>{vec(result.output[t])}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <Tensor
        name={T('按原 token 顺序返回的 Y')}
        data={result.output}
        rowPrefix="t"
        colPrefix="d"
        selected={t}
      />
      <p className="exp-verdict ok">
        {Tx(
          '按专家分组计算再收集，与直接逐 token 参考计算的输出最大差为 {#0}（容差 1e-12）。',
          errorText(difference(result.output, ref)),
        )}
      </p>
      <p className="exp-caption">
        {T('合并的是专家输出向量，不是把专家权重相加。图上同一 t 的各条支路最终只回到该 t。')}
      </p>
    </div>
  );
}

function LoadWidget({ state, result, onChange }: Props) {
  const top1 = computeMoE({ ...state, config: { ...state.config, topK: 1 } });
  const top2 = computeMoE({ ...state, config: { ...state.config, topK: 2 } });
  const c = result.counts;
  return (
    <div className="exp-widget">
      <Segment
        label={T('选择专家数 K')}
        options={[1, 2] as const}
        value={state.config.topK}
        onSelect={(topK) => onChange({ ...state, config: { ...state.config, topK } })}
        render={(k) => 'Top-' + k}
      />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('Top-1 与 Top-2 参数和计算对照')}>
          <caption>{T('同一份权重与输入，仅改变 K；账本只覆盖本 MoE 子层')}</caption>
          <thead>
            <tr>
              <th scope="col">{T('计数口径')}</th>
              <th scope="col">Top-1</th>
              <th scope="col">Top-2</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                [T('全部参数'), 'totalParams'],
                [T('每 token 激活参数（含 Router）'), 'activeParamsPerToken'],
                [T('每 token 专家 MAC'), 'expertMACsPerToken'],
                [T('每 token Router MAC'), 'routerMACsPerToken'],
              ] as const
            ).map(([name, field]) => (
              <tr key={field}>
                <th scope="row">{name}</th>
                <td>{top1.counts[field]}</td>
                <td>{top2.counts[field]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="moe-ledger" aria-label={T('当前 MoE 参数账本')}>
        <div>
          <span>{T('全部权重存储 · FP32')}</span>
          <strong>{c.totalParams * 4} B</strong>
        </div>
        <div>
          <span>{T('本批专家计算')}</span>
          <strong>{c.totalExpertMACs} MAC</strong>
        </div>
        <div>
          <span>{T('本批 Router 计算')}</span>
          <strong>{c.totalRouterMACs} MAC</strong>
        </div>
      </div>
      <p className="exp-caption">
        {Tx(
          '每专家 {#0}×{#1} + {#2}×{#3} = {#4} 个参数； Router {#5}×{#6} = {#7} 个参数。激活参数统计每个 token 使用的权重个数，不是激活张量的大小。 MAC 为所选路径的理论矩阵乘加，忽略 ReLU、Softmax、Top-K 和门控合并。',
          D_MODEL,
          D_FF,
          D_FF,
          D_MODEL,
          2 * D_MODEL * D_FF,
          D_MODEL,
          EXPERTS,
          c.routerParams,
        )}
      </p>
      <p className="exp-verdict">
        {T(
          '切换上方「集中输入」：总分配数仍为 4×K，但可能集中到少数专家。 未激活的专家权重仍需存储；这张账本没有卸载机制，也没有用负载均衡损失调整路由。',
        )}
      </p>
    </div>
  );
}

function ParallelWidget({ state, result, onChange }: Props) {
  const t = result.traffic;
  const ref = directReference(state);
  return (
    <div className="exp-widget">
      <Segment
        label={T('设备布局')}
        options={['single', 'split'] as const}
        value={state.config.topology}
        onSelect={(topology) => onChange({ ...state, config: { ...state.config, topology } })}
        render={(layout) => (layout === 'single' ? T('单设备') : T('双设备专家并行'))}
      />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('专家分发与返回记录')}>
          <caption>{T('按专家聚合的激活包，始终携带原 token 身份')}</caption>
          <thead>
            <tr>
              <th scope="col">{T('token → 专家')}</th>
              <th scope="col">{T('原设备 → 专家设备')}</th>
              <th scope="col">{T('路径')}</th>
            </tr>
          </thead>
          <tbody>
            {result.packets.map((p) => (
              <tr
                key={p.token + ':' + p.expert}
                className={p.token === state.input.token ? 'exp-changed' : ''}
              >
                <th scope="row">
                  t{p.token} → E{p.expert} → t{p.token}
                </th>
                <td>
                  {p.homeDevice} → {p.expertDevice} → {p.homeDevice}
                </td>
                <td>{p.remote ? T('跨设备往返') : T('本地')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="moe-ledger" aria-label={T('跨设备通信账本')}>
        <div>
          <span>{T('远端 token–专家分配')}</span>
          <strong>{Tx('{#0} 份', t.remoteAssignments)}</strong>
        </div>
        <div>
          <span>{T('Dispatch 输入')}</span>
          <strong>{t.dispatchBytes} B</strong>
        </div>
        <div>
          <span>{T('Return 输出')}</span>
          <strong>{t.returnBytes} B</strong>
        </div>
        <div>
          <span>{T('总通信载荷')}</span>
          <strong>{t.totalBytes} B</strong>
        </div>
      </div>
      <p className="exp-caption">
        {T(
          '每份远端分配：发送 D×4 = 8 B 输入，再返回 8 B 输出；本地不计网络流量。 双设备时 t0/t2 原属设备 0，t1/t3 原属设备 1；E0/E1 在设备 0，E2/E3 在设备 1。 单设备全部在设备 0。每条箭头代表激活向量，不发送专家权重。 忽略元数据与协议，只算载荷，不据此推断运行时间。',
        )}
      </p>
      <p className="exp-verdict ok">
        {Tx(
          '分发与返回后，仍按 t0、t1、t2、t3 合并。对照直接逐 token 计算，最大差 {#0}（容差 1e-12）。',
          errorText(difference(result.output, ref)),
        )}
      </p>
    </div>
  );
}

export function MoEWidget(props: Props) {
  const Widget = [
    RouterWidget,
    GateWidget,
    ExpertWidget,
    CombineWidget,
    LoadWidget,
    ParallelWidget,
  ][props.state.step];
  return <Widget {...props} />;
}
