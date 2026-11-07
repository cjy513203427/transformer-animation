import { formatNumber as fmt } from '../../math/attention';
import { computeMoE, D_MODEL, EXPERTS, TOKENS } from '../../math/moe';
import type { MoEState } from '../../state/moe';

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
      aria-label={`t${token} 的 Router：输入 ${vec(result.x[token])}，得到四个专家的分数和概率`}
    >
      <title>同一个 token 给四个专家打分，Softmax 沿专家轴归一化</title>
      <text x="220" y="19" textAnchor="middle" className="moe-svg-label">
        输入行向量 · x_{token} [1, {D_MODEL}]
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
        z · 分数
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
          Softmax · 整行沿专家轴归一化
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
        p · 每行四个概率之和 = 1
      </text>
    </svg>
  );
}

function RoutingDiagram({ state, result }: { state: MoEState; result: Result }) {
  const selectedToken = state.input.token;
  const y = (index: number) => 63 + index * 70;
  const showReturn = state.step >= 3;
  const selection = result.selected[selectedToken].map((expert) => `E${expert}`).join('、');
  return (
    <svg
      className="moe-svg"
      viewBox="0 0 440 320"
      role="img"
      aria-label={`t${selectedToken} 选中 ${selection}；${showReturn ? '专家结果按 token 身份合并回原顺序' : '只向选中的专家分发输入'}`}
    >
      <title>输入 token、编号专家与按原顺序返回的输出；橙色线追踪当前 token</title>
      <text x="42" y="18" textAnchor="middle" className="moe-svg-label">
        X [4, 2]
      </text>
      <text x="220" y="18" textAnchor="middle" className="moe-svg-label">
        {state.step === 1 ? 'Top-K 门控' : '专家 FFN'}
      </text>
      <text x="398" y="18" textAnchor="middle" className="moe-svg-label">
        {showReturn ? 'Y [4, 2]' : '稍后合并'}
      </text>
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
              {on ? `g=${fmt(result.gates[selectedToken][expert])}` : '未选中'}
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
          ? '沿 token ID 返回，按门控权重相加'
          : `每个 token 发送给 ${state.config.topK} 个专家`}
      </text>
    </svg>
  );
}

function LoadDiagram({ state, result }: { state: MoEState; result: Result }) {
  const assignments = TOKENS * state.config.topK;
  return (
    <div className="moe-load-diagram">
      <ul className="moe-load-bars" aria-label="四个专家的 token 分配负载">
        {experts.map((expert) => (
          <li key={expert}>
            <span>E{expert}</span>
            <span className="moe-load-track">
              <i style={{ width: `${(result.load[expert] / TOKENS) * 100}%` }} />
            </span>
            <strong>{result.load[expert]} 次</strong>
            <small>
              {result.packets
                .filter((packet) => packet.expert === expert)
                .map((packet) => `t${packet.token}`)
                .join(' · ') || '空闲'}
            </small>
          </li>
        ))}
      </ul>
      <div className="moe-readout">
        <strong>
          {TOKENS} token × Top-{state.config.topK} = {assignments} 次专家调用
        </strong>
        <span>
          总参数 {result.counts.totalParams}；每 token 激活参数 {result.counts.activeParamsPerToken}
          。
        </span>
        <span>负载统计调用次数，每次输入仍是同一个 D = {D_MODEL} 维向量。</span>
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
            ? `两个设备直接交换激活，远端分配 ${result.traffic.remoteAssignments} 次，总有效载荷 ${result.traffic.totalBytes} 字节`
            : '全部 token 和专家位于设备 0，跨设备通信为 0 字节'
        }
      >
        <title>专家权重驻留在各自设备，只交换 token 激活与专家输出</title>
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
                设备 {device}
              </text>
              <text className="moe-svg-label" x={x + 14} y="68">
                原 token：{residentTokens.map((token) => `t${token}`).join('、')}
              </text>
              <text className="moe-svg-label" x={x + 14} y="94">
                驻留专家：{residentExperts.map((expert) => `E${expert}`).join('、')}
              </text>
              <text className="moe-svg-label" x={x + 14} y="124">
                本地完成：
                {
                  result.packets.filter((packet) => packet.homeDevice === device && !packet.remote)
                    .length
                }{' '}
                次
              </text>
              <text className="moe-svg-label" x={x + 14} y="148">
                返回后按原 token 合并
              </text>
            </g>
          );
        })}
        {split && (
          <>
            <path className="moe-wire moe-wire-on" d="M70 164 V190 H370 V164" />
            <path className="moe-arrow-head" d="M365 172 L370 164 L375 172" />
            <text className="moe-svg-label" x="220" y="211" textAnchor="middle">
              0 → 1：{destinations[0].length} 份输入 + {destinations[1].length} 份返回
            </text>
            <path className="moe-wire moe-wire-on" d="M370 166 V238 H70 V166" />
            <path className="moe-arrow-head" d="M65 174 L70 166 L75 174" />
            <text className="moe-svg-label" x="220" y="259" textAnchor="middle">
              1 → 0：{destinations[1].length} 份输入 + {destinations[0].length} 份返回
            </text>
            <text className="moe-svg-label" x="220" y="284" textAnchor="middle">
              直接设备间交换，无中央节点
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
          只算远端 D = {D_MODEL} 维向量，float32 每个数 4 B；每份输入与返回各 {D_MODEL * 4} B。
        </span>
      </div>
      {split && (
        <ul className="moe-packet-list" aria-label="跨设备激活包">
          {remote.map((packet) => (
            <li key={`${packet.token}-${packet.expert}`}>
              t{packet.token}：设备 {packet.homeDevice} → E{packet.expert}（设备{' '}
              {packet.expertDevice}）→ 原设备 {packet.homeDevice}
            </li>
          ))}
          {remote.length === 0 && <li>本配置没有远端分配。</li>}
        </ul>
      )}
    </div>
  );
}

export function MoEDiagram({ state, result }: { state: MoEState; result: Result }) {
  const token = state.input.token;
  const titles = [
    '同一个输入，给四个专家打分',
    '只向 Top-K 专家分发',
    '不同专家，各自完成 FFN',
    '带着 token 身份返回并加权合并',
    '负载取决于路由，参数仍需存储',
    '专家驻留设备，激活直接往返',
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
          <strong>
            追踪 t{token} · 输入 {vec(result.x[token])}
          </strong>
          {state.step === 0 ? (
            <span>
              z = {vec(result.logits[token])}；p = {vec(result.probabilities[token])}。
            </span>
          ) : (
            <span>
              选中{' '}
              {result.selected[token]
                .map((expert) => `E${expert}（g=${fmt(result.gates[token][expert])}）`)
                .join('、')}
              。
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
              y{token} = {vec(result.output[token])}；输出行顺序仍是 t0、t1、t2、t3。
            </span>
          )}
        </div>
      )}
      <p className="exp-caption">
        编号只是专家身份，未预设语义分工。教学小矩阵，图线表示数据去向。
      </p>
    </div>
  );
}
