import { formatNumber as fmt } from '../../math/attention';
import { computeMoE, D_MODEL, EXPERTS, TOKENS } from '../../math/moe';
import type { MoEState } from '../../state/moe';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import { FitText } from '../../components/FitText';

type Result = ReturnType<typeof computeMoE>;
const vec = (values: number[]) => `[${values.map((value) => fmt(value)).join(', ')}]`;
const experts = Array.from({ length: EXPERTS }, (_, expert) => expert);
const tokens = Array.from({ length: TOKENS }, (_, token) => token);

function RouterDiagram({ state, result }: { state: MoEState; result: Result }) {
  const token = state.input.token;
  return (
    <svg
      className="moe-svg"
      viewBox="0 0 440 330"
      role="img"
      aria-label={T(
        't{#0} 的 Router：输入 {#1}，得到四个专家的分数和概率',
        token,
        vec(result.x[token]),
      )}
    >
      <title>{T('同一个 token 给四个专家打分，Softmax 沿专家轴归一化')}</title>
      <text x="220" y="19" textAnchor="middle" className="moe-svg-label">
        {Tx('输入行向量 · x_{#0} [1, {#1}]', token, D_MODEL)}
      </text>
      <g className="moe-node moe-token moe-node-on">
        <rect x="130" y="32" width="180" height="43" rx="4" />
        <text x="220" y="58" textAnchor="middle">
          t{token} · {vec(result.x[token])}
        </text>
      </g>
      <path className="moe-wire moe-wire-on" d="M220 75 V108 M58 108 H382" />
      {experts.map((expert) => (
        <path
          key={`score-wire-${expert}`}
          className="moe-wire moe-wire-on"
          d={`M${58 + expert * 108} 108 V139`}
        />
      ))}
      <text x="236" y="96" className="moe-svg-label">
        × W_R [2, 4]
      </text>
      <text x="10" y="125" className="moe-svg-label">
        {T('z · 分数')}
      </text>
      {experts.map((expert) => (
        <g key={expert} className="moe-node moe-expert">
          <rect x={10 + expert * 108} y="139" width="96" height="55" rx="4" />
          <text x={58 + expert * 108} y="159" textAnchor="middle">
            E{expert}
          </text>
          <text x={58 + expert * 108} y="180" textAnchor="middle" className="moe-svg-number">
            {fmt(result.logits[token][expert])}
          </text>
          <path className="moe-wire" d={`M${58 + expert * 108} 194 V207`} />
        </g>
      ))}
      <g className="moe-node moe-probability">
        <rect x="10" y="207" width="420" height="32" rx="4" />
        <text x="220" y="228" textAnchor="middle">
          {T('Softmax · 整行沿专家轴归一化')}
        </text>
      </g>
      {experts.map((expert) => (
        <g key={expert} className="moe-node moe-probability">
          <path className="moe-wire" d={`M${58 + expert * 108} 239 V254`} />
          <rect x={10 + expert * 108} y="254" width="96" height="40" rx="4" />
          <text x={58 + expert * 108} y="279" textAnchor="middle">
            {(result.probabilities[token][expert] * 100).toFixed(1)}%
          </text>
        </g>
      ))}
      <text x="10" y="320" className="moe-svg-label">
        {T('p · 每行四个概率之和 = 1')}
      </text>
    </svg>
  );
}

function RoutingDiagram({ state, result }: { state: MoEState; result: Result }) {
  const selectedToken = state.input.token;
  const y = (index: number) => 63 + index * 70;
  const showReturn = state.step >= 3;
  const selection = result.selected[selectedToken].map((expert) => `E${expert}`).join(T('、'));
  return (
    <svg
      className="moe-svg"
      viewBox="0 0 440 320"
      role="img"
      aria-label={T(
        't{#0} 选中 {#1}；{#2}',
        selectedToken,
        selection,
        showReturn ? T('专家结果按 token 身份合并回原顺序') : T('只向选中的专家分发输入'),
      )}
    >
      <title>{T('输入 token、编号专家与按原顺序返回的输出；橙色线追踪当前 token')}</title>
      <text x="42" y="18" textAnchor="middle" className="moe-svg-label">
        X [4, 2]
      </text>
      <text x="220" y="18" textAnchor="middle" className="moe-svg-label">
        {state.step === 1 ? T('Top-K 门控') : T('专家 FFN')}
      </text>
      <FitText x="398" y="18" textAnchor="middle" className="moe-svg-label" width={80} maxLines={1}>
        {showReturn ? 'Y [4, 2]' : T('稍后合并')}
      </FitText>
      {tokens.flatMap((token) =>
        result.selected[token].map((expert) => (
          <g key={`${token}-${expert}`}>
            <path
              className={`moe-wire ${token === selectedToken ? 'moe-wire-on' : 'moe-wire-muted'}`}
              d={`M76 ${y(token)} C128 ${y(token)},132 ${y(expert)},180 ${y(expert)}`}
            />
            {showReturn && (
              <path
                className={`moe-wire moe-return ${token === selectedToken ? 'moe-wire-on' : 'moe-wire-muted'}`}
                d={`M260 ${y(expert)} C310 ${y(expert)},312 ${y(token)},364 ${y(token)}`}
              />
            )}
          </g>
        )),
      )}
      {tokens.map((token) => (
        <g
          key={`token-${token}`}
          className={`moe-node moe-token ${token === selectedToken ? 'moe-node-on' : ''}`}
        >
          <rect x="8" y={y(token) - 21} width="68" height="42" rx="4" />
          <text x="42" y={y(token) + 5} textAnchor="middle">
            t{token}
          </text>
        </g>
      ))}
      {experts.map((expert) => {
        const on = result.selected[selectedToken].includes(expert);
        return (
          <g key={`expert-${expert}`} className={`moe-node moe-expert ${on ? 'moe-node-on' : ''}`}>
            <rect x="180" y={y(expert) - 26} width="80" height="52" rx="4" />
            <text x="220" y={y(expert) - 4} textAnchor="middle">
              E{expert}
            </text>
            <text x="220" y={y(expert) + 15} textAnchor="middle" className="moe-svg-number">
              {on ? `g=${fmt(result.gates[selectedToken][expert])}` : T('未选中')}
            </text>
          </g>
        );
      })}
      {tokens.map((token) => (
        <g
          key={`output-${token}`}
          className={`moe-node moe-output ${token === selectedToken && showReturn ? 'moe-node-on' : ''} ${showReturn ? '' : 'moe-node-pending'}`}
        >
          <rect x="364" y={y(token) - 21} width="68" height="42" rx="4" />
          <text x="398" y={y(token) + 5} textAnchor="middle">
            y{token}
          </text>
        </g>
      ))}
      <text x="220" y="311" textAnchor="middle" className="moe-svg-label">
        {showReturn
          ? T('沿 token ID 返回，按门控权重相加')
          : T('每个 token 发送给 {#0} 个专家', state.config.topK)}
      </text>
    </svg>
  );
}

function LoadDiagram({ state, result }: { state: MoEState; result: Result }) {
  const assignments = TOKENS * state.config.topK;
  return (
    <div className="moe-load-diagram">
      <ul className="moe-load-bars" aria-label={T('四个专家的 token 分配负载')}>
        {experts.map((expert) => (
          <li key={expert}>
            <span>E{expert}</span>
            <span className="moe-load-track">
              <i style={{ width: `${(result.load[expert] / TOKENS) * 100}%` }} />
            </span>
            <strong>{Tx('{#0} 次', result.load[expert])}</strong>
            <small>
              {result.packets
                .filter((packet) => packet.expert === expert)
                .map((packet) => `t${packet.token}`)
                .join(' · ') || T('空闲')}
            </small>
          </li>
        ))}
      </ul>
      <div className="moe-readout">
        <strong>
          {Tx('{#0} token × Top-{#1} = {#2} 次专家调用', TOKENS, state.config.topK, assignments)}
        </strong>
        <span>
          {Tx(
            '总参数 {#0}；每 token 激活参数 {#1}。',
            result.counts.totalParams,
            result.counts.activeParamsPerToken,
          )}
        </span>
        <span>{Tx('负载统计调用次数，每次输入仍是同一个 D = {#0} 维向量。', D_MODEL)}</span>
      </div>
    </div>
  );
}

function ParallelDiagram({ state, result }: { state: MoEState; result: Result }) {
  const split = state.config.topology === 'split';
  const remote = result.packets.filter((packet) => packet.remote);
  const destinations = [0, 1].map((device) =>
    remote.filter((packet) => packet.homeDevice === device),
  );
  const devices = split ? [0, 1] : [0];
  return (
    <div className="moe-parallel-diagram">
      <svg
        className="moe-svg"
        viewBox={`0 0 440 ${split ? 290 : 190}`}
        role="img"
        aria-label={
          split
            ? T(
                '两个设备直接交换激活，远端分配 {#0} 次，总有效载荷 {#1} 字节',
                result.traffic.remoteAssignments,
                result.traffic.totalBytes,
              )
            : T('全部 token 和专家位于设备 0，跨设备通信为 0 字节')
        }
      >
        <title>{T('专家权重驻留在各自设备，只交换 token 激活与专家输出')}</title>
        {devices.map((device) => {
          const x = split ? 8 + device * 222 : 44;
          const width = split ? 202 : 352;
          const residentTokens = tokens.filter((token) => !split || token % 2 === device);
          const residentExperts = experts.filter(
            (expert) => !split || Math.floor(expert / 2) === device,
          );
          return (
            <g key={device}>
              <rect className="moe-device" x={x} y="18" width={width} height="145" rx="5" />
              <text className="moe-device-title" x={x + 14} y="43">
                {Tx('设备 {#0}', device)}
              </text>
              <FitText className="moe-svg-label" x={x + 14} y="68" width={width - 28} maxLines={1}>
                {T('原 token：{#0}', residentTokens.map((token) => `t${token}`).join(T('、')))}
              </FitText>
              <FitText className="moe-svg-label" x={x + 14} y="94" width={width - 28} maxLines={1}>
                {T('驻留专家：{#0}', residentExperts.map((expert) => `E${expert}`).join(T('、')))}
              </FitText>
              <FitText className="moe-svg-label" x={x + 14} y="124" width={width - 28} maxLines={1}>
                {T(
                  '本地完成：{#0} 次',
                  result.packets.filter((packet) => packet.homeDevice === device && !packet.remote)
                    .length,
                )}
              </FitText>
              <FitText className="moe-svg-label" x={x + 14} y="148" width={width - 28} maxLines={1}>
                {T('返回后按原 token 合并')}
              </FitText>
            </g>
          );
        })}
        {split && (
          <>
            <path className="moe-wire moe-wire-on" d="M70 164 V190 H370 V164" />
            <path className="moe-arrow-head" d="M365 172 L370 164 L375 172" />
            <text className="moe-svg-label" x="220" y="211" textAnchor="middle">
              {Tx(
                '0 → 1：{#0} 份输入 + {#1} 份返回',
                destinations[0].length,
                destinations[1].length,
              )}
            </text>
            <path className="moe-wire moe-wire-on" d="M370 166 V238 H70 V166" />
            <path className="moe-arrow-head" d="M65 174 L70 166 L75 174" />
            <text className="moe-svg-label" x="220" y="259" textAnchor="middle">
              {Tx(
                '1 → 0：{#0} 份输入 + {#1} 份返回',
                destinations[1].length,
                destinations[0].length,
              )}
            </text>
            <text className="moe-svg-label" x="220" y="284" textAnchor="middle">
              {T('直接设备间交换，无中央节点')}
            </text>
          </>
        )}
      </svg>
      <div className="moe-readout">
        <strong>
          Dispatch {result.traffic.dispatchBytes} B + Return {result.traffic.returnBytes} B ={' '}
          {result.traffic.totalBytes} B
        </strong>
        <span>
          {Tx(
            '只算远端 D = {#0} 维向量，float32 每个数 4 B；每份输入与返回各 {#1} B。',
            D_MODEL,
            D_MODEL * 4,
          )}
        </span>
      </div>
      {split && (
        <ul className="moe-packet-list" aria-label={T('跨设备激活包')}>
          {remote.map((packet) => (
            <li key={`${packet.token}-${packet.expert}`}>
              {Tx(
                't{#0}：设备 {#1} → E{#2}（设备 {#3}）→ 原设备 {#4}',
                packet.token,
                packet.homeDevice,
                packet.expert,
                packet.expertDevice,
                packet.homeDevice,
              )}
            </li>
          ))}
          {remote.length === 0 && <li>{T('本配置没有远端分配。')}</li>}
        </ul>
      )}
    </div>
  );
}

export function MoEDiagram({ state, result }: { state: MoEState; result: Result }) {
  const token = state.input.token;
  const titles = [
    T('同一个输入，给四个专家打分'),
    T('只向 Top-K 专家分发'),
    T('不同专家，各自完成 FFN'),
    T('带着 token 身份返回并加权合并'),
    T('负载取决于路由，参数仍需存储'),
    T('专家驻留设备，激活直接往返'),
  ];
  return (
    <div className="moe-diagram-panel">
      <div className="diagram-caption">
        <strong>{titles[state.step]}</strong>
      </div>
      {state.step === 0 ? (
        <RouterDiagram state={state} result={result} />
      ) : state.step < 4 ? (
        <RoutingDiagram state={state} result={result} />
      ) : state.step === 4 ? (
        <LoadDiagram state={state} result={result} />
      ) : (
        <ParallelDiagram state={state} result={result} />
      )}
      {state.step < 4 && (
        <div className="moe-readout">
          <strong>{Tx('追踪 t{#0} · 输入 {#1}', token, vec(result.x[token]))}</strong>
          {state.step === 0 ? (
            <span>
              {Tx(
                'z = {#0}；p = {#1}。',
                vec(result.logits[token]),
                vec(result.probabilities[token]),
              )}
            </span>
          ) : (
            <span>
              {Tx(
                '选中 {#0}。',
                result.selected[token]
                  .map((expert) => T('E{#0}（g={#1}）', expert, fmt(result.gates[token][expert])))
                  .join(T('、')),
              )}
            </span>
          )}
          {state.step === 2 &&
            result.selected[token].map((expert) => (
              <span key={expert}>
                E{expert}(x{token}) = {vec(result.expertOutputs[token][expert] ?? [])}
              </span>
            ))}
          {state.step === 3 && (
            <span>
              {Tx(
                'y{#0} = {#1}；输出行顺序仍是 t0、t1、t2、t3。',
                token,
                vec(result.output[token]),
              )}
            </span>
          )}
        </div>
      )}
      <p className="exp-caption">
        {T('编号只是专家身份，未预设语义分工。教学小矩阵，图线表示数据去向。')}
      </p>
    </div>
  );
}
