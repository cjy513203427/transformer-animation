import { Tex } from '../../components/Tex';
import {
  D_HEAD,
  D_LATENT,
  D_ROPE,
  DEEPSEEK_V2,
  formatBytes,
  groupOf,
  kvCacheBytes,
  mlaCacheBytes,
  Q_HEADS,
} from '../../math/kv';
import { kvHeadChoices, type KVState } from '../../state/kv';

const kindOf = (g: number, h: number) => (g === h ? 'MHA' : g === 1 ? 'MQA' : `GQA-${g}`);

function Bars({
  label,
  rows,
}: {
  label: string;
  rows: { name: string; value: number; text: string; current?: boolean; parts?: number[] }[];
}) {
  const max = Math.max(...rows.map((row) => row.value));
  return (
    <ul className="kv-bars" aria-label={label}>
      {rows.map((row) => (
        <li key={row.name} className={row.current ? 'kv-bar-current' : undefined}>
          <span className="kv-bar-name">
            {row.name}
            {row.current && <small>当前</small>}
          </span>
          <span className="kv-bar-track">
            {(row.parts ?? [row.value]).map((part, i) => (
              <i
                key={i}
                className={`kv-bar-part kv-bar-part-${i}`}
                style={{ width: `${Math.max(0.6, (part / max) * 100)}%` }}
              />
            ))}
          </span>
          <span className="kv-bar-value">{row.text}</span>
        </li>
      ))}
    </ul>
  );
}

function HeadMap({ kvHeads, head }: { kvHeads: number; head: number }) {
  const width = 360;
  const qx = (h: number) => 45 + h * 90;
  const gx = (g: number) => width / 2 + (g - (kvHeads - 1) / 2) * 90;
  const selectedGroup = groupOf(head, Q_HEADS, kvHeads);
  return (
    <svg
      className="kv-headmap"
      viewBox={`0 0 ${width} 230`}
      role="img"
      aria-label={`${Q_HEADS} 个 Query 头读 ${kvHeads} 组 K/V：${Array.from(
        { length: Q_HEADS },
        (_, h) => `Q${h} 读第 ${groupOf(h, Q_HEADS, kvHeads)} 组`,
      ).join('，')}`}
    >
      {Array.from({ length: Q_HEADS }, (_, h) => {
        const g = groupOf(h, Q_HEADS, kvHeads);
        const on = h === head;
        return (
          <line
            key={`wire-${h}`}
            x1={qx(h)}
            y1={62}
            x2={gx(g)}
            y2={160}
            className={on ? 'kv-wire kv-wire-on' : 'kv-wire'}
          />
        );
      })}
      {Array.from({ length: Q_HEADS }, (_, h) => (
        <g key={`q-${h}`} className={h === head ? 'kv-node kv-q kv-node-on' : 'kv-node kv-q'}>
          <rect x={qx(h) - 32} y={22} width={64} height={40} rx={4} />
          <text x={qx(h)} y={47} textAnchor="middle">
            Q{h}
          </text>
        </g>
      ))}
      {Array.from({ length: kvHeads }, (_, g) => (
        <g
          key={`kv-${g}`}
          className={g === selectedGroup ? 'kv-node kv-kv kv-node-on' : 'kv-node kv-kv'}
        >
          <rect x={gx(g) - 36} y={160} width={72} height={40} rx={4} />
          <text x={gx(g)} y={185} textAnchor="middle">
            K/V {g}
          </text>
        </g>
      ))}
      <text x={8} y={14} className="kv-axis">
        Query 头（数量不变）
      </text>
      <text x={8} y={222} className="kv-axis">
        K/V 头：{kvHeads} 组（{kindOf(kvHeads, Q_HEADS)}）
      </text>
    </svg>
  );
}

/** One token's cache footprint in one layer: a cell per stored number. */
function CacheCells() {
  const rows: { name: string; cells: { text: string; kind: string }[] }[] = [
    ...([4, 2, 1] as const).map((g) => ({
      name: `${kindOf(g, Q_HEADS)}（${g} 组 K/V）`,
      cells: Array.from({ length: g }, (_, h) => [
        ...Array.from({ length: D_HEAD }, () => ({ text: `K${h}`, kind: 'k' })),
        ...Array.from({ length: D_HEAD }, () => ({ text: `V${h}`, kind: 'v' })),
      ]).flat(),
    })),
    {
      name: 'MLA（latent + RoPE Key）',
      cells: [
        ...Array.from({ length: D_LATENT }, () => ({ text: 'c', kind: 'latent' })),
        ...Array.from({ length: D_ROPE }, () => ({ text: 'kᴿ', kind: 'rope' })),
      ],
    },
  ];
  return (
    <div className="kv-cells" role="table" aria-label="一个 token 在一层里缓存的数">
      {rows.map((row) => (
        <div key={row.name} className="kv-cells-row" role="row">
          <span role="rowheader">
            {row.name}
            <small>{row.cells.length} 个数</small>
          </span>
          <span className="kv-cells-track" role="cell">
            {row.cells.map((cell, i) => (
              <i key={i} className={`kv-cell kv-cell-${cell.kind}`}>
                {cell.text}
              </i>
            ))}
          </span>
        </div>
      ))}
    </div>
  );
}

function Flow({
  title,
  active,
  steps,
  note,
}: {
  title: string;
  active: boolean;
  steps: { tex?: string; text: string; kind?: 'cache' | 'warn' | 'plain' }[];
  note: string;
}) {
  return (
    <section className={active ? 'kv-flow kv-flow-on' : 'kv-flow'} aria-label={title}>
      <strong>
        {title}
        {active && <small>当前</small>}
      </strong>
      <ol>
        {steps.map((step, i) => (
          <li key={i} className={`kv-flow-step kv-flow-${step.kind ?? 'plain'}`}>
            {step.tex && <Tex tex={step.tex} />}
            <span>{step.text}</span>
          </li>
        ))}
      </ol>
      <p>{note}</p>
    </section>
  );
}

export function KVDiagram({ state }: { state: KVState }) {
  const { step } = state;
  let title = '';
  let body = null;

  if (step === 0) {
    const c = state.calc;
    title = '同一配置下，不同 KV 头数的缓存';
    body = (
      <Bars
        label="不同 KV 头数的 KV 缓存字节数"
        rows={kvHeadChoices(c.qHeads)
          .slice(0, 8)
          .map((g) => {
            const bytes = kvCacheBytes({
              batch: c.batch,
              layers: c.layers,
              tokens: c.tokens,
              kvHeads: g,
              dK: c.dHead,
              dV: c.dHead,
              bytesPerElement: c.bytes,
            });
            return {
              name: `${kindOf(g, c.qHeads)} · ${g} 个 KV 头`,
              value: bytes,
              text: formatBytes(bytes),
              current: g === c.kvHeads,
            };
          })}
      />
    );
  } else if (step === 1) {
    title = 'Query 头不变，K/V 头分组共享';
    body = <HeadMap kvHeads={state.group.kvHeads} head={state.group.head} />;
  } else if (step === 2) {
    title = '一个 token 在一层里要缓存多少个数';
    body = <CacheCells />;
  } else if (step === 3) {
    const coupled = state.mla.coupled;
    title = 'RoPE 放在哪里，决定了能不能吸收';
    body = (
      <div className="kv-flows">
        <Flow
          title="不解耦：RoPE 加在还原出的 K 上"
          active={coupled}
          steps={[
            { tex: 'c_j', text: '读缓存', kind: 'cache' },
            { tex: 'k_j=c_jW_{UK}', text: '还原内容 Key' },
            { tex: '\\operatorname{RoPE}_j(k_j)', text: '旋转依赖位置 j', kind: 'warn' },
            { tex: 'q\\,R_tR_j^{\\top}W_{UK}^{\\top}c_j^{\\top}', text: '中间矩阵随 t − j 变化' },
          ]}
          note="没有一个固定矩阵能把 W_UK 并进 Query，只能为每个历史位置重新还原 K。"
        />
        <Flow
          title="解耦：位置信息单独走一支"
          active={!coupled}
          steps={[
            { tex: 'c_j', text: '内容：读缓存', kind: 'cache' },
            { tex: '\\left(qW_{UK}^{\\top}\\right)c_j^{\\top}', text: '内容分数可以吸收' },
            { tex: 'k^R_j', text: '位置：读缓存（所有头共享）', kind: 'cache' },
            { tex: 'q^R\\cdot k^R_j', text: '位置分数，两部分相加' },
          ]}
          note="缓存里多存一个共享的 RoPE Key，换来内容部分可以吸收。"
        />
      </div>
    );
  } else if (step === 4) {
    title = '同一个结果，两种计算顺序';
    body = (
      <div className="kv-flows">
        <Flow
          title="展开：先还原 K、V"
          active={false}
          steps={[
            { tex: 'c_j', text: '读缓存', kind: 'cache' },
            { tex: 'k^C_{j,i}=c_jW_{UK}^{(i)},\\ v^C_{j,i}=c_jW_{UV}^{(i)}', text: '每个头还原' },
            { tex: '\\sum_j p_j\\,v^C_{j,i}', text: '加权求和' },
            { tex: '[o_1;\\dots;o_H]\\,W_O', text: '拼接后输出投影' },
          ]}
          note="教学上直观：每个头先拿到自己的 K、V。"
        />
        <Flow
          title="吸收：只读缓存，不还原"
          active
          steps={[
            { tex: '\\tilde q=h_t\\tilde W_Q^{(i)}', text: `Query 变成 ${D_LATENT} 维` },
            { tex: '\\tilde q\\cdot c_j+q^R\\cdot k^R_j', text: '直接与缓存打分', kind: 'cache' },
            { tex: '\\sum_j p_j\\,c_j', text: '加权求和停在 latent 空间' },
            { tex: '\\sum_i(\\cdot)\\,\\tilde W_O^{(i)}', text: '吸收后的输出投影' },
          ]}
          note="结合律保证结果相同；推理时每个历史 token 只读缓存里的数。"
        />
      </div>
    );
  } else {
    const tokens = state.compare.tokens;
    const { layers, heads, dHead, latent, ropeDim } = DEEPSEEK_V2;
    const kv = (g: number) =>
      kvCacheBytes({
        batch: 1,
        layers,
        tokens,
        kvHeads: g,
        dK: dHead,
        dV: dHead,
        bytesPerElement: 2,
      });
    const mla = mlaCacheBytes({ batch: 1, layers, tokens, latent, ropeDim, bytesPerElement: 2 });
    const latentOnly = (mla * latent) / (latent + ropeDim);
    title = `DeepSeek-V2 的头数与头维，${tokens.toLocaleString('en-US')} 个 token`;
    body = (
      <>
        <Bars
          label="同样头数与头维时四种设计的 KV 缓存"
          rows={[
            { name: `MHA · ${heads} 个 KV 头`, value: kv(heads), text: formatBytes(kv(heads)) },
            { name: 'GQA-8 · 8 个 KV 头', value: kv(8), text: formatBytes(kv(8)) },
            { name: 'MQA · 1 个 KV 头', value: kv(1), text: formatBytes(kv(1)) },
            {
              name: 'MLA · latent + RoPE Key',
              value: mla,
              text: formatBytes(mla),
              parts: [latentOnly, mla - latentOnly],
            },
          ]}
        />
        <p className="exp-caption">
          MLA 一栏分两段：深色是 latent，浅色是共享的 RoPE Key。BF16、batch = 1，只统计 KV 缓存。
        </p>
      </>
    );
  }

  return (
    <div className="kv-diagram-panel">
      <div className="diagram-caption">
        <strong>{title}</strong>
      </div>
      {body}
    </div>
  );
}
