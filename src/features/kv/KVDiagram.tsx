import { Cuboid } from '../../components/Oblique';
import { Tex } from '../../components/Tex';
import {
  D_HEAD,
  D_LATENT,
  D_ROPE,
  DEEPSEEK_V2,
  formatBytes,
  groupOf,
  kvCacheBytes,
  kvCacheElements,
  KV_OPTIONS,
  mlaCacheBytes,
  mlaCacheElements,
  Q_HEADS,
  TOKENS,
} from '../../math/kv';
import { kvHeadChoices, type KVState } from '../../state/kv';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

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
            {row.current && <small>{T('当前')}</small>}
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
      aria-label={T(
        '{#0} 个 Query 头读 {#1} 组 K/V：{#2}',
        Q_HEADS,
        kvHeads,
        Array.from({ length: Q_HEADS }, (_, h) =>
          T('Q{#0} 读第 {#1} 组', h, groupOf(h, Q_HEADS, kvHeads)),
        ).join(T('，')),
      )}
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
        {T('Query 头（数量不变）')}
      </text>
      <text x={8} y={222} className="kv-axis">
        {Tx('K/V 头：{#0} 组（{#1}）', kvHeads, kindOf(kvHeads, Q_HEADS))}
      </text>
    </svg>
  );
}

/** One design's cache in one layer, drawn from the same counts as the cache formula. */
interface CacheDesign {
  name: string;
  sub: string;
  rows: number;
  columns: { kind: 'k' | 'v' | 'latent' | 'rope'; label: string }[];
  elements: number;
}
function cacheDesigns(): CacheDesign[] {
  const kv = (g: number): CacheDesign => ({
    name: kindOf(g, Q_HEADS),
    sub: T('{#0} 组 K/V', g),
    rows: g,
    columns: [
      ...Array.from({ length: D_HEAD }, () => ({ kind: 'k' as const, label: 'k' })),
      ...Array.from({ length: D_HEAD }, () => ({ kind: 'v' as const, label: 'v' })),
    ],
    elements: kvCacheElements({
      batch: 1,
      layers: 1,
      tokens: TOKENS,
      kvHeads: g,
      dK: D_HEAD,
      dV: D_HEAD,
      bytesPerElement: 1,
    }),
  });
  return [
    ...KV_OPTIONS.map(kv),
    {
      name: 'MLA',
      sub: 'latent + RoPE Key',
      rows: 1,
      columns: [
        ...Array.from({ length: D_LATENT }, () => ({ kind: 'latent' as const, label: 'c' })),
        ...Array.from({ length: D_ROPE }, () => ({ kind: 'rope' as const, label: 'kᴿ' })),
      ],
      elements: mlaCacheElements({
        batch: 1,
        layers: 1,
        tokens: TOKENS,
        latent: D_LATENT,
        ropeDim: D_ROPE,
        bytesPerElement: 1,
      }),
    },
  ];
}
const BLOCK_CELL: [number, number] = [15, 15];
const BLOCK_STEP: [number, number] = [8, 6];
const BLOCK_LEFT = 34;
const BLOCK_BOTTOM = 98;

/** [tokens, KV heads, d_k + d_v] as a block of cells: rows are KV heads, depth is tokens. */
function CacheBlock({ design }: { design: CacheDesign }) {
  const cols = design.columns.length,
    { rows } = design,
    [cw, ch] = BLOCK_CELL,
    [sx, sy] = BLOCK_STEP,
    top = BLOCK_BOTTOM - rows * ch,
    /** Each block keeps the same width and scale; only the empty space above it is cropped. */
    viewTop = top - TOKENS * sy - 12,
    perToken = rows * cols;
  return (
    <figure className="kv-block">
      <svg
        viewBox={`0 ${viewTop} 176 ${BLOCK_BOTTOM + 16 - viewTop}`}
        role="img"
        aria-label={T(
          '{#0}：{#1} 个 token × {#2}{#3} 个数 = {#4} 个数',
          design.name,
          TOKENS,
          design.name === 'MLA' ? '' : T('{#0} 组 K/V × ', rows),
          cols,
          design.elements,
        )}
      >
        <Cuboid
          x={BLOCK_LEFT}
          y={top}
          cols={cols}
          rows={rows}
          depth={TOKENS}
          cell={BLOCK_CELL}
          step={BLOCK_STEP}
          fill={(c) => `kv-blk kv-blk-${design.columns[c].kind}`}
        />
        {design.name !== 'MLA' && (
          <text
            x={BLOCK_LEFT - 5}
            y={top + (rows * ch) / 2 + 4}
            textAnchor="end"
            className="kv-blk-axis"
          >
            {T('{#0}组', rows)}
          </text>
        )}
        {design.columns.map((col, c) => (
          <text
            key={c}
            x={BLOCK_LEFT + c * cw + cw / 2}
            y={BLOCK_BOTTOM + 12}
            textAnchor="middle"
            className="kv-blk-col"
          >
            {col.label}
          </text>
        ))}
        <text
          x={BLOCK_LEFT + cols * cw + TOKENS * sx + 3}
          y={top - TOKENS * sy + 6}
          className="kv-blk-axis"
        >
          T={TOKENS}
        </text>
      </svg>
      <figcaption>
        <strong>{design.name}</strong>
        <span>{design.sub}</span>
        <span>{Tx('每 token {#0} 个 × {#1} = {#2}', perToken, TOKENS, design.elements)}</span>
      </figcaption>
    </figure>
  );
}

function CacheBlocks() {
  return (
    <>
      <div
        className="kv-blocks"
        role="group"
        aria-label={T('一层里 {#0} 个 token 的 KV 缓存块', TOKENS)}
      >
        {cacheDesigns().map((design) => (
          <CacheBlock key={design.name} design={design} />
        ))}
      </div>
      <p className="exp-caption">
        {Tx(
          '每个小方块是缓存里的一个数。块的正面是一个 token 的一片：行是 KV 头，列是 k、v 各 {#0} 维（MLA 没有 KV 头这一维，每片是 latent c 的 {#1} 维和 RoPE Key 的 {#2} 维）；往后每多一个 token 多一片。这是一层、一个请求的缓存，L 层、batch B 时再乘 L × B。',
          D_HEAD,
          D_LATENT,
          D_ROPE,
        )}
      </p>
    </>
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
        {active && <small>{T('当前')}</small>}
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
    title = T('同一配置下，不同 KV 头数的缓存');
    body = (
      <Bars
        label={T('不同 KV 头数的 KV 缓存字节数')}
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
              name: T('{#0} · {#1} 个 KV 头', kindOf(g, c.qHeads), g),
              value: bytes,
              text: formatBytes(bytes),
              current: g === c.kvHeads,
            };
          })}
      />
    );
  } else if (step === 1) {
    title = T('Query 头不变，K/V 头分组共享');
    body = <HeadMap kvHeads={state.group.kvHeads} head={state.group.head} />;
  } else if (step === 2) {
    title = T('一层里 {#0} 个 token 的缓存：一个小方块是一个数', TOKENS);
    body = <CacheBlocks />;
  } else if (step === 3) {
    const coupled = state.mla.coupled;
    title = T('RoPE 放在哪里，决定了能不能吸收');
    body = (
      <div className="kv-flows">
        <Flow
          title={T('不解耦：RoPE 加在还原出的 K 上')}
          active={coupled}
          steps={[
            { tex: 'c_j', text: T('读缓存'), kind: 'cache' },
            { tex: 'k_j=c_jW_{UK}', text: T('还原内容 Key') },
            { tex: '\\operatorname{RoPE}_j(k_j)', text: T('旋转依赖位置 j'), kind: 'warn' },
            {
              tex: 'q\\,R_tR_j^{\\top}W_{UK}^{\\top}c_j^{\\top}',
              text: T('中间矩阵随 t − j 变化'),
            },
          ]}
          note={T('没有一个固定矩阵能把 W_UK 并进 Query，只能为每个历史位置重新还原 K。')}
        />
        <Flow
          title={T('解耦：位置信息单独走一支')}
          active={!coupled}
          steps={[
            { tex: 'c_j', text: T('内容：读缓存'), kind: 'cache' },
            { tex: '\\left(qW_{UK}^{\\top}\\right)c_j^{\\top}', text: T('内容分数可以吸收') },
            { tex: 'k^R_j', text: T('位置：读缓存（所有头共享）'), kind: 'cache' },
            { tex: 'q^R\\cdot k^R_j', text: T('位置分数，两部分相加') },
          ]}
          note={T('缓存里多存一个共享的 RoPE Key，换来内容部分可以吸收。')}
        />
      </div>
    );
  } else if (step === 4) {
    title = T('同一个结果，两种计算顺序');
    body = (
      <div className="kv-flows">
        <Flow
          title={T('展开：先还原 K、V')}
          active={false}
          steps={[
            { tex: 'c_j', text: T('读缓存'), kind: 'cache' },
            {
              tex: 'k^C_{j,i}=c_jW_{UK}^{(i)},\\ v^C_{j,i}=c_jW_{UV}^{(i)}',
              text: T('每个头还原'),
            },
            { tex: '\\sum_j p_j\\,v^C_{j,i}', text: T('加权求和') },
            { tex: '[o_1;\\dots;o_H]\\,W_O', text: T('拼接后输出投影') },
          ]}
          note={T('教学上直观：每个头先拿到自己的 K、V。')}
        />
        <Flow
          title={T('吸收：只读缓存，不还原')}
          active
          steps={[
            { tex: '\\tilde q=h_t\\tilde W_Q^{(i)}', text: T('Query 变成 {#0} 维', D_LATENT) },
            {
              tex: '\\tilde q\\cdot c_j+q^R\\cdot k^R_j',
              text: T('直接与缓存打分'),
              kind: 'cache',
            },
            { tex: '\\sum_j p_j\\,c_j', text: T('加权求和停在 latent 空间') },
            { tex: '\\sum_i(\\cdot)\\,\\tilde W_O^{(i)}', text: T('吸收后的输出投影') },
          ]}
          note={T('结合律保证结果相同；推理时每个历史 token 只读缓存里的数。')}
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
    title = T('DeepSeek-V2 的头数与头维，{#0} 个 token', tokens.toLocaleString('en-US'));
    body = (
      <>
        <Bars
          label={T('同样头数与头维时四种设计的 KV 缓存')}
          rows={[
            {
              name: T('MHA · {#0} 个 KV 头', heads),
              value: kv(heads),
              text: formatBytes(kv(heads)),
            },
            { name: T('GQA-8 · 8 个 KV 头'), value: kv(8), text: formatBytes(kv(8)) },
            { name: T('MQA · 1 个 KV 头'), value: kv(1), text: formatBytes(kv(1)) },
            {
              name: 'MLA · latent + RoPE Key',
              value: mla,
              text: formatBytes(mla),
              parts: [latentOnly, mla - latentOnly],
            },
          ]}
        />
        <p className="exp-caption">
          {T(
            'MLA 一栏分两段：深色是 latent，浅色是共享的 RoPE Key。BF16、batch = 1，只统计 KV 缓存。',
          )}
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
