import type { ReactNode } from 'react';
import { NUM_LAYERS, VOCAB, type Generation } from '../../math/inference';
import { PROMPT } from '../../state/inference';

type Kind =
  | 'visible'
  | 'masked'
  | 'repeat'
  | 'fresh'
  | 'write'
  | 'read'
  | 'new'
  | 'empty'
  | 'stale'
  | 'keep'
  | 'dropped'
  | 'unknown'
  | 'target'
  | 'count';

interface Cell {
  text: string;
  kind: Kind;
}
interface Row {
  label: string;
  note?: string;
  changed?: boolean;
  cells: Cell[];
}

const LEGEND: Partial<Record<Kind, string>> = {
  visible: '可见：这一行能读到的输入',
  masked: '屏蔽：未来位置，不参与计算',
  repeat: '重算：上一步已经算过，结果相同',
  fresh: '新：这一步真正需要的行',
  write: '写入：Prefill 算出并存进缓存',
  read: '读取：从缓存直接取用',
  new: '新增：本步为新 token 算出并追加',
  empty: '空：之后的 Decode 才会写入',
  target: '目标：这一行要预测的 token（来自同一条序列）',
  stale: '失效：数值变了，必须重算',
  keep: '可用：数值不变，可以直接复用',
  dropped: '丢弃：移出窗口',
  unknown: '？先预测，再核对',
  count: '数字：这个位置的 K、V 一共被算了几次',
};

function Grid({
  caption,
  columns,
  rows,
  corner,
}: {
  caption: string;
  columns: { label: string; sub?: string }[];
  rows: Row[];
  corner: string;
}) {
  const kinds = [...new Set(rows.flatMap((row) => row.cells.map((cell) => cell.kind)))];
  return (
    <div className="inf-diagram">
      <div className="inf-grid-scroll">
        <table className="inf-grid" aria-label={caption}>
          <thead>
            <tr>
              <th scope="col">{corner}</th>
              {columns.map((column, i) => (
                <th key={i} scope="col">
                  <span className="inf-token">{column.label}</span>
                  {column.sub && <small>{column.sub}</small>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className={row.changed ? 'inf-row-changed' : undefined}>
                <th scope="row">
                  {row.label}
                  {row.note && <small>{row.note}</small>}
                </th>
                {row.cells.map((cell, i) => (
                  <td key={i} className={`inf-cell inf-${cell.kind}`}>
                    {cell.text}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="inf-legend">
        {kinds
          .filter((kind) => LEGEND[kind])
          .map((kind) => (
            <li key={kind}>
              <i className={`inf-cell inf-${kind}`} aria-hidden="true" />
              {LEGEND[kind]}
            </li>
          ))}
      </ul>
    </div>
  );
}

const name = (id: number) => VOCAB[id];
const layerLabel = (layer: number) => `第 ${layer + 1} 层`;

export function CacheDiagram({
  step,
  generation,
  decodeSteps,
  training,
  invalidation,
}: {
  step: number;
  generation: Generation;
  decodeSteps: number;
  training: { inputs: number[]; targets: number[]; changedRows: boolean[]; applied: boolean };
  invalidation: {
    mode: 'edit' | 'slide';
    ids: number[];
    flags: (boolean | null)[][];
    revealed: boolean;
  };
}) {
  const cached = generation.tokens.slice(0, PROMPT.length + decodeSteps);
  const positions = cached.map((id, p) => ({ label: name(id), sub: `pos ${p}` }));
  const layers = Array.from({ length: NUM_LAYERS }, (_, layer) => layer);

  let title = '';
  let content: ReactNode = null;

  if (step === 0) {
    title = '训练：4 行在一次前向里同时算出';
    content = (
      <Grid
        caption="训练时的因果 Mask：行是输出位置，列是输入"
        corner="输出行 ↓ 输入 →"
        columns={[
          ...training.inputs.map((id, j) => ({ label: name(id), sub: `x${j}` })),
          { label: '目标', sub: 'y_t' },
        ]}
        rows={training.inputs.map((_, t) => ({
          label: `第 ${t} 行`,
          note: training.applied ? (training.changedRows[t] ? '已变' : '不变') : undefined,
          changed: training.applied && training.changedRows[t],
          cells: [
            ...training.inputs.map((_, j): Cell =>
              j <= t ? { text: '可见', kind: 'visible' } : { text: '屏蔽', kind: 'masked' },
            ),
            { text: name(training.targets[t]), kind: 'target' },
          ],
        }))}
      />
    );
  } else if (step === 1) {
    title = '无缓存：每一步都把整段重新算一遍';
    const passes = [PROMPT.length, ...generation.steps.map((_, s) => PROMPT.length + s + 1)];
    content = (
      <Grid
        caption="无缓存时每一步重算的行"
        corner="步骤 ↓ 位置 →"
        columns={positions}
        rows={passes.map((n, s) => ({
          label: s === 0 ? '第一次前向' : `第 ${s} 步`,
          note: `长度 ${n}`,
          cells: positions.map((_, p): Cell => {
            if (p >= n) return { text: '', kind: 'empty' };
            if (s === 0) return { text: '新', kind: 'fresh' };
            return p === n - 1 ? { text: '新', kind: 'fresh' } : { text: '重算', kind: 'repeat' };
          }),
        }))}
      />
    );
  } else if (step === 2) {
    title = 'Prefill：提示的 K、V 写入每一层的缓存';
    content = (
      <Grid
        caption="Prefill 之后的 KV 缓存"
        corner="层 ↓ 位置 →"
        // Only the prompt is known here; later tokens do not exist yet, so they are not named.
        columns={[...positions.slice(0, PROMPT.length), { label: '…', sub: '之后' }]}
        rows={layers.map((layer) => ({
          label: layerLabel(layer),
          cells: [
            ...PROMPT.map((): Cell => ({ text: 'K, V', kind: 'write' })),
            { text: '', kind: 'empty' },
          ],
        }))}
      />
    );
  } else if (step === 3) {
    title = `第 ${decodeSteps} 次 Decode：读旧条目，只追加一个新位置`;
    const newest = cached.length - 1;
    content = (
      <Grid
        caption={`第 ${decodeSteps} 次 Decode 时的 KV 缓存`}
        corner="层 ↓ 位置 →"
        columns={positions}
        rows={layers.map((layer) => ({
          label: layerLabel(layer),
          cells: positions.map((_, p): Cell =>
            p === newest ? { text: '新增', kind: 'new' } : { text: '读取', kind: 'read' },
          ),
        }))}
      />
    );
  } else if (step === 4) {
    title = '每个位置的 K、V 一共被算了几次（每层相同）';
    // Without a cache, position p is projected in the first pass (if it is in the prompt) and
    // again in every later pass whose sequence reaches it.
    const passes = [PROMPT.length, ...generation.steps.map((_, s) => PROMPT.length + s + 1)];
    content = (
      <Grid
        caption="无缓存与有缓存时每个位置被投影的次数"
        corner="方式 ↓ 位置 →"
        columns={positions}
        rows={[
          {
            label: '无缓存',
            cells: positions.map((_, p) => ({
              text: `×${passes.filter((n) => n > p).length}`,
              kind: 'count',
            })),
          },
          {
            label: '有缓存',
            cells: positions.map(() => ({ text: '×1', kind: 'count' })),
          },
        ]}
      />
    );
  } else {
    const { mode, ids, flags, revealed } = invalidation;
    title =
      mode === 'edit'
        ? '改一个提示 token 之后，哪些缓存条目还能用？'
        : '窗口右移一格之后，哪些旧条目还能用？';
    content = (
      <Grid
        caption={mode === 'edit' ? '改动前缀后的缓存有效性' : '滑动窗口后的缓存有效性'}
        corner="层 ↓ 位置 →"
        columns={ids.map((id, p) => ({ label: name(id), sub: `pos ${p}` }))}
        rows={layers.map((layer) => ({
          label: layerLabel(layer),
          cells: flags[layer].map((flag): Cell => {
            if (flag === null) return { text: '丢弃', kind: 'dropped' };
            if (!revealed) return { text: '?', kind: 'unknown' };
            return flag ? { text: '失效', kind: 'stale' } : { text: '可用', kind: 'keep' };
          }),
        }))}
      />
    );
  }

  return (
    <div className="inf-diagram-panel">
      <div className="diagram-caption">
        <strong>{title}</strong>
      </div>
      {content}
    </div>
  );
}
