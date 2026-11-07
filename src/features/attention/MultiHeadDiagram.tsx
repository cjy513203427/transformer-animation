import { SvgRich } from '../../components/Tex';
import { formatNumber as fmt, type Matrix } from '../../math/attention';
import { MH_WEIGHTS, MH_X, type MultiHeadResult } from '../../math/multihead';

export const HEAD_COLORS = ['#4d622f', '#c0507a', '#5262c4', '#a8741f'];
const CW = 34;
const CH = 22;
const WIDTH = 480;
const HEIGHT = 450;
const ROW_Y = [62, 200, 338];

type Tone = 'neutral' | 'query' | 'key' | 'value' | 'probability' | 'output' | 'weight';
const TINTS: Record<Tone, string> = {
  neutral: '#d9d7d2',
  query: '#dbd8d0',
  key: '#e3d5c3',
  value: '#d7cde1',
  probability: '#d2dac6',
  output: '#cadad5',
  weight: '#dddbd7',
};
const INKS: Record<Tone, string> = {
  neutral: '#504d46',
  query: '#3558a4',
  key: '#8d4b13',
  value: '#6f4a92',
  probability: '#354818',
  output: '#1f5d4c',
  weight: '#55524c',
};

interface Placement {
  x: number;
  y: number;
  visible: boolean;
}
type Layout = Record<string, Placement>;

function pieceKeys(numHeads: number) {
  const keys = ['X', 'Wo', 'OUT'];
  for (let h = 0; h < numHeads; h++) keys.push(`Q${h}`, `K${h}`, `V${h}`, `P${h}`, `A${h}`);
  return keys;
}

/** Where every piece sits in a given step; hidden pieces are positioned later. */
function rawLayout(step: number, numHeads: number): Partial<Layout> {
  const dKey = 4 / numHeads;
  const slice = dKey * CW;
  const out: Partial<Layout> = {};
  const qkv = (x0: number, gap: number) => {
    for (let h = 0; h < numHeads; h++)
      ['Q', 'K', 'V'].forEach((name, i) => {
        out[`${name}${h}`] = { x: x0 + h * (slice + gap), y: ROW_Y[i], visible: true };
      });
  };
  if (step === 0) {
    out.X = { x: 16, y: ROW_Y[1], visible: true };
    qkv(320, 0);
  } else if (step === 1) {
    const gap = numHeads === 1 ? 0 : 14;
    qkv((WIDTH - (4 * CW + (numHeads - 1) * gap)) / 2, gap);
  } else if (step === 2) {
    const gap = numHeads === 1 ? 0 : numHeads === 2 ? 64 : 34;
    qkv((WIDTH - (4 * CW + (numHeads - 1) * gap)) / 2, gap);
  } else if (step === 3) {
    for (let h = 0; h < numHeads; h++) {
      const y = rowTop(h, numHeads);
      out[`P${h}`] = { x: 20, y, visible: true };
      out[`V${h}`] = { x: 160, y, visible: true };
      out[`A${h}`] = { x: 160 + slice + 50, y, visible: true };
    }
  } else if (step === 4) {
    for (let h = 0; h < numHeads; h++)
      out[`A${h}`] = { x: (WIDTH - 4 * CW) / 2 + h * slice, y: ROW_Y[1], visible: true };
  } else {
    for (let h = 0; h < numHeads; h++) out[`A${h}`] = { x: 16 + h * slice, y: 186, visible: true };
    out.Wo = { x: 172, y: 175, visible: true };
    out.OUT = { x: 330, y: 186, visible: true };
  }
  return out;
}

function rowTop(h: number, numHeads: number) {
  if (numHeads === 1) return 180;
  if (numHeads === 2) return 92 + h * 170;
  return 58 + h * 100;
}

/** Hidden pieces wait where they will next appear (or where they were last), so fades never fly. */
function layoutFor(step: number, numHeads: number): Layout {
  const all = Array.from({ length: 6 }, (_, s) => rawLayout(s, numHeads));
  const result: Layout = {};
  for (const key of pieceKeys(numHeads)) {
    const current = all[step][key];
    if (current) {
      result[key] = current;
      continue;
    }
    const order = [...Array(6).keys()].sort(
      (a, b) => Math.abs(a - step) - Math.abs(b - step) || b - a,
    );
    const found = order.map((s) => all[s][key]).find(Boolean) ?? { x: 0, y: 0, visible: false };
    result[key] = { ...found, visible: false };
  }
  return result;
}

function Grid({
  data,
  tone,
  head,
  query,
  changed,
  heat,
  masked,
  showValues,
  rowHeads,
}: {
  data: Matrix;
  tone: Tone;
  head?: number;
  query?: number;
  changed?: boolean[][];
  heat?: boolean;
  masked?: boolean[][];
  showValues: boolean;
  rowHeads?: number[];
}) {
  const width = data[0].length * CW;
  return (
    <>
      {head !== undefined && (
        <rect x="0" y="-6" width={width} height="4" rx="1.5" fill={HEAD_COLORS[head]} />
      )}
      {data.map((row, i) =>
        row.map((value, j) => {
          const fill = heat ? `rgba(69, 94, 32, ${0.06 + Math.max(0, value) * 0.62})` : TINTS[tone];
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={j * CW + 1}
                y={i * CH + 1}
                width={CW - 2}
                height={CH - 2}
                rx="3"
                fill={masked?.[i]?.[j] ? 'url(#mh-hatch)' : fill}
                stroke={changed?.[i]?.[j] ? '#c2410c' : 'none'}
                strokeWidth={changed?.[i]?.[j] ? 2 : 0}
              />
              {showValues && (
                <text
                  className="mh-cell-text"
                  x={j * CW + CW / 2}
                  y={i * CH + 15}
                  textAnchor="middle"
                  fill={heat && value > 0.55 ? '#e5e2dc' : INKS[tone]}
                >
                  {fmt(value, 2)}
                </text>
              )}
              {changed?.[i]?.[j] && (
                <path
                  d={`M${(j + 1) * CW - 9} ${i * CH + 2} h7 v7 Z`}
                  fill="#c2410c"
                  aria-hidden="true"
                />
              )}
            </g>
          );
        }),
      )}
      {rowHeads?.map((h, i) => (
        <rect key={i} x="-6" y={i * CH + 2} width="3.5" height={CH - 4} fill={HEAD_COLORS[h]} />
      ))}
      {query !== undefined && (
        <rect
          className="mh-query-row"
          x="-1"
          y={query * CH - 1}
          width={width + 2}
          height={CH + 2}
          rx="4"
        />
      )}
    </>
  );
}

function Label({
  x,
  y,
  children,
  tone = '#5b5852',
  anchor = 'start',
}: {
  x: number;
  y: number;
  children: string;
  tone?: string;
  anchor?: 'start' | 'middle' | 'end';
}) {
  if (children.includes('$')) {
    return (
      <SvgRich
        x={x}
        y={y}
        text={children}
        anchor={anchor}
        size={11}
        width={340}
        style={{ color: tone }}
      />
    );
  }
  return (
    <text className="mh-label" x={x} y={y} fill={tone} textAnchor={anchor}>
      {children}
    </text>
  );
}

export function MultiHeadDiagram({
  result,
  step,
  numHeads,
  query,
  changed,
  showValues,
}: {
  result: MultiHeadResult;
  step: number;
  numHeads: number;
  query: number;
  changed: {
    probabilities: boolean[][][];
    concat: boolean[][];
    output: boolean[][];
    key: boolean[][];
  };
  showValues: boolean;
}) {
  const layout = layoutFor(step, numHeads);
  const dKey = 4 / numHeads;
  const slice = dKey * CW;
  const heads = Array.from({ length: numHeads }, (_, h) => h);
  const place = (key: string) => ({
    transform: `translate(${layout[key].x}px, ${layout[key].y}px)`,
    opacity: layout[key].visible ? 1 : 0,
  });
  const sliceOf = (matrix: Matrix, h: number) =>
    matrix.map((row) => row.slice(h * dKey, (h + 1) * dKey));
  const shape = (dims: number[]) => `[${dims.join(', ')}]`;
  const titles = [
    `一次投影：X ${shape([3, 4])} × W ${shape([4, 4])}`,
    `切分多头：${shape([3, 4])} → ${shape([3, numHeads, dKey])}`,
    `矩阵变形：→ ${shape([numHeads, 3, dKey])}`,
    `每个头独立计算：P_h ${shape([3, 3])}，A_h ${shape([3, dKey])}`,
    `Concatenate：→ A ${shape([3, 4])}`,
    `输出投影：A × W_O → Output ${shape([3, 4])}`,
  ];
  return (
    <svg
      className="mh-diagram"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`多头注意力第 ${step + 1} 步，${titles[step]}`}
    >
      <defs>
        <pattern
          id="mh-hatch"
          width="5"
          height="5"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width="5" height="5" fill="#d7d5d2" />
          <line x1="0" y1="0" x2="0" y2="5" stroke="#bcb9b3" strokeWidth="2" />
        </pattern>
        <marker
          id="mh-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0 0 L10 5 L0 10Z" fill="#79756c" />
        </marker>
      </defs>
      <text className="mh-title" x={WIDTH / 2} y="18" textAnchor="middle">
        {titles[step]}
      </text>

      {/* Static overlays per step: arrows, frames and labels. They fade but never fake numbers. */}
      <g className="mh-overlay" style={{ opacity: step === 0 ? 1 : 0 }} aria-hidden="true">
        <Label x={16} y={ROW_Y[1] - 12}>
          X [3, 4]
        </Label>
        {ROW_Y.map((y, i) => (
          <g key={y}>
            <path
              d={`M156 ${ROW_Y[1] + 33} C230 ${ROW_Y[1] + 33} 240 ${y + 33} 312 ${y + 33}`}
              className="mh-wire"
              markerEnd="url(#mh-arrow)"
            />
            <Label x={300} y={y + 24} anchor="end" tone="#615f58">
              {['$\\times W_Q$', '$\\times W_K$', '$\\times W_V$'][i]}
            </Label>
            <Label x={320} y={y - 12} tone={['#3558a4', '#8d4b13', '#6f4a92'][i]}>
              {['Q [3, 4]', 'K [3, 4]', 'V [3, 4]'][i]}
            </Label>
          </g>
        ))}
      </g>
      <g
        className="mh-overlay"
        style={{ opacity: step === 1 || step === 2 ? 1 : 0 }}
        aria-hidden="true"
      >
        {step === 1 &&
          ROW_Y.map((y, i) => (
            <Label
              key={y}
              x={WIDTH / 2}
              y={y - 12}
              anchor="middle"
              tone={['#3558a4', '#8d4b13', '#6f4a92'][i]}
            >
              {`${['Q', 'K', 'V'][i]} → [3, ${numHeads}, ${dKey}]`}
            </Label>
          ))}
        {step === 2 &&
          heads.map((h) => {
            const x = layout[`Q${h}`].x;
            return (
              <g key={h}>
                <rect
                  x={x - 9}
                  y={ROW_Y[0] - 30}
                  width={slice + 18}
                  height={ROW_Y[2] + 3 * CH - ROW_Y[0] + 40}
                  rx="8"
                  className="mh-card"
                  stroke={HEAD_COLORS[h]}
                />
                <Label x={x + slice / 2} y={ROW_Y[0] - 14} anchor="middle" tone={HEAD_COLORS[h]}>
                  {`head ${h + 1}`}
                </Label>
              </g>
            );
          })}
        {step === 2 &&
          ['Q', 'K', 'V'].map((name, i) => (
            <Label key={name} x={14} y={ROW_Y[i] + 37} tone={['#3558a4', '#8d4b13', '#6f4a92'][i]}>
              {`$${name}_h$`}
            </Label>
          ))}
      </g>
      <g className="mh-overlay" style={{ opacity: step === 3 ? 1 : 0 }} aria-hidden="true">
        {step === 3 &&
          heads.map((h) => {
            const y = rowTop(h, numHeads);
            return (
              <g key={h}>
                <Label x={20} y={y - 11} tone={HEAD_COLORS[h]}>
                  {`head ${h + 1} · $P_{${h + 1}}=\\operatorname{softmax}\\!\\left(Q_{${h + 1}}K_{${h + 1}}^{\\top}/\\sqrt{${dKey}}\\right)$`}
                </Label>
                <text className="mh-operator" x={140} y={y + 38} textAnchor="middle">
                  ×
                </text>
                <text className="mh-operator" x={160 + slice + 25} y={y + 38} textAnchor="middle">
                  =
                </text>
                <Label x={160} y={y + 3 * CH + 14} tone="#6f4a92">{`$V_{${h + 1}}$`}</Label>
                <Label x={160 + slice + 50} y={y + 3 * CH + 14} tone="#1f5d4c">
                  {`$A_{${h + 1}}$`}
                </Label>
              </g>
            );
          })}
      </g>
      <g className="mh-overlay" style={{ opacity: step === 4 ? 1 : 0 }} aria-hidden="true">
        <Label x={WIDTH / 2} y={ROW_Y[1] - 14} anchor="middle" tone="#1f5d4c">
          {`$A=\\operatorname{Concat}(${heads.map((h) => `A_{${h + 1}}`).join(',\\,')})$  [3, 4]`}
        </Label>
        <Label x={WIDTH / 2} y={ROW_Y[1] + 3 * CH + 30} anchor="middle">
          {`[${numHeads}, 3, ${dKey}] → 转置 [3, ${numHeads}, ${dKey}] → view [3, 4]`}
        </Label>
        <Label x={WIDTH / 2} y={ROW_Y[1] + 3 * CH + 52} anchor="middle" tone="#6f6c65">
          拼接只是并排摆放，头与头之间还没有交流
        </Label>
      </g>
      <g className="mh-overlay" style={{ opacity: step === 5 ? 1 : 0 }} aria-hidden="true">
        <Label x={16} y={172}>
          A [3, 4]
        </Label>
        <Label x={172} y={161}>
          {'$\\times W_O$ [4, 4]'}
        </Label>
        <Label x={330} y={172}>
          = Output [3, 4]
        </Label>
        <Label x={WIDTH / 2} y={310} anchor="middle" tone="#6f6c65">
          {'$W_O$ 的每一列都读取 A 的全部列：每个输出数都混合了所有头'}
        </Label>
        <Label x={WIDTH / 2} y={332} anchor="middle" tone="#6f6c65">
          {'$W_O$ 左侧色条：这几行只与对应头的 A 列相乘'}
        </Label>
      </g>

      {/* Pieces: one element per tensor slice, moved between steps by CSS transitions. */}
      <g className="mh-piece" style={place('X')}>
        <Grid data={MH_X} tone="neutral" query={query} showValues={showValues} />
      </g>
      {heads.map((h) => (
        <g key={`q${h}`}>
          <g className="mh-piece" style={place(`Q${h}`)}>
            <Grid
              data={sliceOf(result.q, h)}
              tone="query"
              head={numHeads > 1 ? h : undefined}
              query={query}
              showValues={showValues}
            />
          </g>
          <g className="mh-piece" style={place(`K${h}`)}>
            <Grid
              data={sliceOf(result.k, h)}
              tone="key"
              head={numHeads > 1 ? h : undefined}
              changed={changed.key.map((row) => row.slice(h * dKey, (h + 1) * dKey))}
              showValues={showValues}
            />
          </g>
          <g className="mh-piece" style={place(`V${h}`)}>
            <Grid
              data={sliceOf(result.v, h)}
              tone="value"
              head={numHeads > 1 ? h : undefined}
              showValues={showValues}
            />
          </g>
          <g className="mh-piece" style={place(`P${h}`)}>
            <Grid
              data={result.heads[h].probabilities}
              tone="probability"
              heat
              masked={result.heads[h].masked.map((row) => row.map((x) => x === -Infinity))}
              query={query}
              changed={changed.probabilities[h]}
              showValues={showValues}
            />
          </g>
          <g className="mh-piece" style={place(`A${h}`)}>
            <Grid
              data={result.heads[h].output}
              tone="output"
              head={numHeads > 1 ? h : undefined}
              query={step === 5 || step === 4 ? undefined : query}
              changed={changed.concat.map((row) => row.slice(h * dKey, (h + 1) * dKey))}
              showValues={showValues}
            />
          </g>
        </g>
      ))}
      {(step === 4 || step === 5) && (
        <rect
          className="mh-query-row"
          x={layout.A0.x - 1}
          y={layout.A0.y + query * CH - 1}
          width={4 * CW + 2}
          height={CH + 2}
          rx="4"
        />
      )}
      <g className="mh-piece" style={place('Wo')}>
        <Grid
          data={MH_WEIGHTS.wo}
          tone="weight"
          rowHeads={[0, 1, 2, 3].map((r) => Math.floor(r / dKey))}
          showValues={showValues}
        />
      </g>
      <g className="mh-piece" style={place('OUT')}>
        <Grid
          data={result.output}
          tone="output"
          query={query}
          changed={changed.output}
          showValues={showValues}
        />
      </g>
    </svg>
  );
}
