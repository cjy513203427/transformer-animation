import { useId, type KeyboardEvent } from 'react';
import { SvgTex } from '../../components/Tex';
import type { Insertion } from '../../content/position';

export const INSERTION_COLORS: Record<Insertion, string> = {
  add: '#b45309',
  rope: '#be185d',
  alibi: '#4d7c0f',
};
const INSERTION_STEP: Record<Insertion, number> = { add: 1, rope: 3, alibi: 5 };
const INSERTION_LABEL: Record<Insertion, string> = {
  add: '① 加法位置编码：加到 X 上',
  rope: '② RoPE：旋转 Q 和 K',
  alibi: '③ ALiBi：分数加距离偏置',
};

function Box({
  x,
  y,
  width,
  label,
  tex,
  sub,
  tone = 'neutral',
}: {
  x: number;
  y: number;
  width: number;
  label: string;
  /** When given, shown instead of `label` as typeset math. */
  tex?: string;
  sub?: string;
  tone?: 'neutral' | 'query' | 'key' | 'value' | 'probability';
}) {
  return (
    <g className={`pp-box pp-${tone}`}>
      <rect x={x - width / 2} y={y - 17} width={width} height="34" rx="6" />
      {tex ? (
        <SvgTex x={x} y={sub ? y - 2 : y + 5} tex={tex} size={14} width={width} />
      ) : (
        <text x={x} y={sub ? y - 2 : y + 5} textAnchor="middle">
          {label}
        </text>
      )}
      {sub && (
        <text className="pp-sub" x={x} y={y + 11} textAnchor="middle">
          {sub}
        </text>
      )}
    </g>
  );
}

export function PositionPipeline({
  active,
  onSelect,
}: {
  active: Insertion[];
  onSelect: (step: number) => void;
}) {
  const id = useId().replace(/:/g, '') + '-pp';
  const on = (kind: Insertion) => active.includes(kind);
  const wire = (kind: Insertion | null) =>
    `pp-wire ${kind && on(kind) ? `pp-lit pp-lit-${kind}` : ''}`;
  function insertionProps(kind: Insertion) {
    const select = () => onSelect(INSERTION_STEP[kind]);
    return {
      className: `pp-insert ${on(kind) ? 'active' : 'idle'}`,
      role: 'button',
      tabIndex: 0,
      'aria-label': `${INSERTION_LABEL[kind]}，跳到对应步骤`,
      'aria-pressed': on(kind),
      onClick: select,
      onKeyDown: (e: KeyboardEvent) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      },
      style: { ['--insert' as string]: INSERTION_COLORS[kind] },
    };
  }
  return (
    <div className="pp-panel">
      <div className="diagram-caption">
        <strong>顺序信息从哪里进入？</strong>
        <span>↑ 从下往上读</span>
      </div>
      <svg
        viewBox="0 0 460 640"
        role="group"
        aria-label="单头 Attention 计算流程，标出三种位置编码加入的位置"
      >
        <defs>
          {(['base', 'add', 'rope', 'alibi'] as const).map((tone) => (
            <marker
              key={tone}
              id={`${id}-${tone}`}
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path
                d="M0 0 L10 5 L0 10Z"
                fill={tone === 'base' ? '#817c73' : INSERTION_COLORS[tone]}
              />
            </marker>
          ))}
        </defs>
        {/* Wires: a lit wire means position information is already present in that tensor. */}
        <path
          className={wire('add')}
          d="M230 592 V572"
          markerEnd={`url(#${id}-${on('add') ? 'add' : 'base'})`}
        />
        <path
          className={wire('add')}
          d="M230 542 V522"
          markerEnd={`url(#${id}-${on('add') ? 'add' : 'base'})`}
        />
        {[110, 230, 350].map((x) => (
          <path
            key={x}
            className={wire('add')}
            d={`M230 486 V474 H${x} V464`}
            markerEnd={`url(#${id}-${on('add') ? 'add' : 'base'})`}
          />
        ))}
        {[110, 230, 350].map((x) => (
          <path
            key={x}
            className={wire('add')}
            d={`M${x} 428 V406`}
            markerEnd={`url(#${id}-${on('add') ? 'add' : 'base'})`}
          />
        ))}
        {[110, 230].map((x) => (
          <path
            key={x}
            className={on('rope') ? wire('rope') : wire('add')}
            d={`M${x} 370 V348`}
            markerEnd={`url(#${id}-${on('rope') ? 'rope' : on('add') ? 'add' : 'base'})`}
          />
        ))}
        <path
          className={on('rope') ? wire('rope') : wire('add')}
          d="M110 312 V298 H150 V290"
          markerEnd={`url(#${id}-${on('rope') ? 'rope' : on('add') ? 'add' : 'base'})`}
        />
        <path
          className={on('rope') ? wire('rope') : wire('add')}
          d="M230 312 V298 H190 V290"
          markerEnd={`url(#${id}-${on('rope') ? 'rope' : on('add') ? 'add' : 'base'})`}
        />
        <path
          className={
            on('rope') || on('add')
              ? `pp-wire pp-lit pp-lit-${on('rope') ? 'rope' : 'add'}`
              : 'pp-wire'
          }
          d="M170 254 V232"
          markerEnd={`url(#${id}-${on('rope') ? 'rope' : on('add') ? 'add' : 'base'})`}
        />
        {['M170 204 V185', 'M170 151 V135', 'M170 101 V82'].map((d) => (
          <path
            key={d}
            className={
              active.length
                ? `pp-wire pp-lit pp-lit-${on('alibi') ? 'alibi' : on('rope') ? 'rope' : 'add'}`
                : 'pp-wire'
            }
            d={d}
            markerEnd={`url(#${id}-${on('alibi') ? 'alibi' : on('rope') ? 'rope' : on('add') ? 'add' : 'base'})`}
          />
        ))}
        <path
          className={wire('add')}
          d="M350 370 V64 H322"
          markerEnd={`url(#${id}-${on('add') ? 'add' : 'base'})`}
          data-testid="pp-value-path"
        />
        <path className="pp-wire" d="M230 47 V30" markerEnd={`url(#${id}-base)`} />
        <text className="pp-output" x="230" y="22" textAnchor="middle">
          输出 O
        </text>

        <Box
          x={230}
          y={610}
          width={170}
          label="E[token]"
          tex="E[\text{token}]"
          sub="查嵌入表：只有内容"
        />
        <Box x={230} y={504} width={120} label="X" sub="进入 Attention 的表示" />
        <Box x={110} y={446} width={86} label="× W_Q" tex="\times W_Q" tone="query" />
        <Box x={230} y={446} width={86} label="× W_K" tex="\times W_K" tone="key" />
        <Box x={350} y={446} width={86} label="× W_V" tex="\times W_V" tone="value" />
        <Box x={110} y={388} width={64} label="Q" tone="query" />
        <Box x={230} y={388} width={64} label="K" tone="key" />
        <Box x={350} y={388} width={64} label="V" tone="value" />
        <Box
          x={170}
          y={272}
          width={170}
          label="Q·Kᵀ / √d"
          tex="QK^{\top}/\sqrt{d}"
          sub="逐对打分"
        />
        <Box x={170} y={168} width={150} label="Mask" sub="挡住未来" />
        <Box x={170} y={118} width={150} label="Softmax" tone="probability" />
        <Box x={230} y={64} width={180} label="P · V" sub="按比例汇总 V" tone="probability" />
        <text className="pp-note" x={362} y={300} textAnchor="start">
          {active.length > 1 ? '' : on('rope') ? 'V 不旋转' : on('add') ? 'V 也带位置' : ''}
        </text>

        {/* ① additive PE */}
        <g {...insertionProps('add')}>
          <circle cx="230" cy="556" r="15" />
          <text x="230" y="563" textAnchor="middle" className="pp-plus">
            ⊕
          </text>
          <rect x="300" y="540" width="140" height="32" rx="6" className="pp-side" />
          <SvgTex
            x={370}
            y={561}
            width={140}
            size={12}
            prefix="① "
            tex="\mathrm{PE}[\mathrm{pos}]"
          />
          <path d="M300 556 H249" className="pp-insert-wire" />
        </g>
        {/* ② RoPE on Q and K */}
        <g {...insertionProps('rope')}>
          {[110, 230].map((x) => (
            <g key={x}>
              <rect x={x - 54} y="313" width="108" height="34" rx="6" className="pp-side" />
              <SvgTex
                x={x}
                y={335}
                width={108}
                size={12}
                prefix="② ↻ "
                tex={x === 110 ? 'R_m\\,q' : 'R_n\\,k'}
              />
            </g>
          ))}
        </g>
        {/* ③ ALiBi on the scores */}
        <g {...insertionProps('alibi')}>
          <circle cx="170" cy="218" r="15" />
          <text x="170" y="225" textAnchor="middle" className="pp-plus">
            ⊕
          </text>
          <rect x="16" y="202" width="118" height="32" rx="6" className="pp-side" />
          <SvgTex x={75} y={223} width={118} size={12} prefix="③ " tex="-m\cdot\text{距离}" />
          <path d="M134 218 H151" className="pp-insert-wire" />
        </g>
        {!active.length && (
          <g className="pp-empty" aria-hidden="true">
            <rect x="322" y="588" width="130" height="46" rx="8" />
            <text x="387" y="607" textAnchor="middle">
              三个入口都未启用
            </text>
            <text x="387" y="625" textAnchor="middle">
              X 只含内容
            </text>
          </g>
        )}
      </svg>
      <ul className="pp-legend">
        {(['add', 'rope', 'alibi'] as const).map((kind) => (
          <li
            key={kind}
            className={on(kind) ? 'active' : ''}
            style={{ ['--insert' as string]: INSERTION_COLORS[kind] }}
          >
            {INSERTION_LABEL[kind]}
          </li>
        ))}
      </ul>
    </div>
  );
}
