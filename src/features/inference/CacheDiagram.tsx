import type { ReactNode } from 'react';
import { NUM_LAYERS, VOCAB, type Generation } from '../../math/inference';
import { PROMPT } from '../../state/inference';
import { T } from '../../i18n';

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
  visible: T('可见：这一行能读到的输入'),
  masked: T('屏蔽：未来位置，不参与计算'),
  repeat: T('重算：上一步已经算过，结果相同'),
  fresh: T('新：这一步真正需要的行'),
  write: T('写入：Prefill 算出并存进缓存'),
  read: T('读取：从缓存直接取用'),
  new: T('新增：本步为新 token 算出并追加'),
  empty: T('空：之后的 Decode 才会写入'),
  target: T('目标：这一行要预测的 token（来自同一条序列）'),
  stale: T('失效：数值变了，必须重算'),
  keep: T('可用：数值不变，可以直接复用'),
  dropped: T('丢弃：移出窗口'),
  unknown: T('？先预测，再核对'),
  count: T('数字：这个位置的 K、V 一共被算了几次'),
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
const layerLabel = (layer: number) => T('第 {#0} 层', layer + 1);

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
    title = T('训练：4 行在一次前向里同时算出');
    content = (
      <Grid
        caption={T('训练时的因果 Mask：行是输出位置，列是输入')}
        corner={T('输出行 ↓ 输入 →')}
        columns={[
          ...training.inputs.map((id, j) => ({ label: name(id), sub: `x${j}` })),
          { label: T('目标'), sub: 'y_t' },
        ]}
        rows={training.inputs.map((_, t) => ({
          label: T('第 {#0} 行', t),
          note: training.applied ? (training.changedRows[t] ? T('已变') : T('不变')) : undefined,
          changed: training.applied && training.changedRows[t],
          cells: [
            ...training.inputs.map((_, j): Cell =>
              j <= t ? { text: T('可见'), kind: 'visible' } : { text: T('屏蔽'), kind: 'masked' },
            ),
            { text: name(training.targets[t]), kind: 'target' },
          ],
        }))}
      />
    );
  } else if (step === 1) {
    title = T('无缓存：每一步都把整段重新算一遍');
    const passes = [PROMPT.length, ...generation.steps.map((_, s) => PROMPT.length + s + 1)];
    content = (
      <Grid
        caption={T('无缓存时每一步重算的行')}
        corner={T('步骤 ↓ 位置 →')}
        columns={positions}
        rows={passes.map((n, s) => ({
          label: s === 0 ? T('第一次前向') : T('第 {#0} 步', s),
          note: T('长度 {#0}', n),
          cells: positions.map((_, p): Cell => {
            if (p >= n) return { text: '', kind: 'empty' };
            if (s === 0) return { text: T('新'), kind: 'fresh' };
            return p === n - 1
              ? { text: T('新'), kind: 'fresh' }
              : { text: T('重算'), kind: 'repeat' };
          }),
        }))}
      />
    );
  } else if (step === 2) {
    title = T('Prefill：提示的 K、V 写入每一层的缓存');
    content = (
      <Grid
        caption={T('Prefill 之后的 KV 缓存')}
        corner={T('层 ↓ 位置 →')}
        // Only the prompt is known here; later tokens do not exist yet, so they are not named.
        columns={[...positions.slice(0, PROMPT.length), { label: '…', sub: T('之后') }]}
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
    title = T('第 {#0} 次 Decode：读旧条目，只追加一个新位置', decodeSteps);
    const newest = cached.length - 1;
    content = (
      <Grid
        caption={T('第 {#0} 次 Decode 时的 KV 缓存', decodeSteps)}
        corner={T('层 ↓ 位置 →')}
        columns={positions}
        rows={layers.map((layer) => ({
          label: layerLabel(layer),
          cells: positions.map((_, p): Cell =>
            p === newest ? { text: T('新增'), kind: 'new' } : { text: T('读取'), kind: 'read' },
          ),
        }))}
      />
    );
  } else if (step === 4) {
    title = T('每个位置的 K、V 一共被算了几次（每层相同）');
    // Without a cache, position p is projected in the first pass (if it is in the prompt) and
    // again in every later pass whose sequence reaches it.
    const passes = [PROMPT.length, ...generation.steps.map((_, s) => PROMPT.length + s + 1)];
    content = (
      <Grid
        caption={T('无缓存与有缓存时每个位置被投影的次数')}
        corner={T('方式 ↓ 位置 →')}
        columns={positions}
        rows={[
          {
            label: T('无缓存'),
            cells: positions.map((_, p) => ({
              text: `×${passes.filter((n) => n > p).length}`,
              kind: 'count',
            })),
          },
          {
            label: T('有缓存'),
            cells: positions.map(() => ({ text: '×1', kind: 'count' })),
          },
        ]}
      />
    );
  } else {
    const { mode, ids, flags, revealed } = invalidation;
    title =
      mode === 'edit'
        ? T('改一个提示 token 之后，哪些缓存条目还能用？')
        : T('窗口右移一格之后，哪些旧条目还能用？');
    content = (
      <Grid
        caption={mode === 'edit' ? T('改动前缀后的缓存有效性') : T('滑动窗口后的缓存有效性')}
        corner={T('层 ↓ 位置 →')}
        columns={ids.map((id, p) => ({ label: name(id), sub: `pos ${p}` }))}
        rows={layers.map((layer) => ({
          label: layerLabel(layer),
          cells: flags[layer].map((flag): Cell => {
            if (flag === null) return { text: T('丢弃'), kind: 'dropped' };
            if (!revealed) return { text: '?', kind: 'unknown' };
            return flag ? { text: T('失效'), kind: 'stale' } : { text: T('可用'), kind: 'keep' };
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
