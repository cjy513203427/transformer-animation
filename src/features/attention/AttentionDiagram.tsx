import { useId } from 'react';
import { SvgTex, Tex } from '../../components/Tex';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/attention-diagram.css';
import '../../styles/attention-interactions.css';
import { FitText } from '../../components/FitText';

export function AttentionDiagram({
  step,
  onStep,
  causal = true,
  scaled = true,
  onOpenHeads,
}: {
  step: number;
  onStep: (step: number) => void;
  causal?: boolean;
  scaled?: boolean;
  onOpenHeads?: () => void;
}) {
  const id = useId().replace(/:/g, '') + '-attention';
  function node(
    x: number,
    y: number,
    width: number,
    label: string,
    sublabel: string,
    phase: number,
    tone = 'neutral',
    tex?: string,
  ) {
    return (
      <g
        className={`diagram-node diagram-${tone} ${step === phase ? 'active' : ''}`}
        role="button"
        tabIndex={0}
        aria-label={T('图中步骤：{#0}，{#1}', label, sublabel)}
        aria-pressed={step === phase}
        onClick={() => onStep(phase)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onStep(phase);
          }
        }}
      >
        <rect x={x - width / 2} y={y - 23} width={width} height="46" rx="7" />
        {tex ? (
          <SvgTex x={x} y={y - 3} tex={tex} size={15} width={width} />
        ) : (
          <FitText x={x} y={y - 3} textAnchor="middle" width={width - 10} maxLines={1}>
            {label}
          </FitText>
        )}
        <FitText
          className="diagram-node-sub"
          x={x}
          y={y + 14}
          textAnchor="middle"
          width={width - 10}
          maxLines={1}
        >
          {sublabel}
        </FitText>
      </g>
    );
  }
  return (
    <div className="diagram-panel">
      <div className="diagram-caption">
        <strong>{T('放大：一个头的内部')}</strong>
        <span>{T('↑ 沿箭头从下往上读')}</span>
      </div>
      <p className="figure-interaction-hint">{T('点选计算节点，跳到对应步骤')}</p>
      <svg
        viewBox="0 0 540 665"
        role="group"
        aria-label={T('Attention 内部架构图：Q 和 K 打分，V 绕过打分链进入最后的加权求和')}
      >
        <defs>
          {['base', 'q', 'k', 'v', 'p'].map((tone, i) => (
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
                fill={['#817c73', '#4267c7', '#a85a17', '#865dad', '#455e20'][i]}
              />
            </marker>
          ))}
        </defs>
        {['M270 588 V572 H85 V553', 'M270 572 H245 V553', 'M270 572 H425 V553'].map((d) => (
          <path
            key={d}
            className={`diagram-wire ${step === 0 ? 'lit' : ''}`}
            d={d}
            markerEnd={`url(#${id}-base)`}
          />
        ))}
        {['M85 507 V483', 'M85 437 V419 H150 V408'].map((d) => (
          <path key={d} className="diagram-wire q-wire" d={d} markerEnd={`url(#${id}-q)`} />
        ))}
        {['M245 507 V483', 'M245 437 V419 H220 V408'].map((d) => (
          <path key={d} className="diagram-wire k-wire" d={d} markerEnd={`url(#${id}-k)`} />
        ))}
        <path className="diagram-wire v-wire" d="M425 507 V483" markerEnd={`url(#${id}-v)`} />
        <path
          className={`diagram-wire v-wire ${step === 4 ? 'lit' : ''}`}
          data-testid="value-bypass"
          d="M425 437 V130 H284"
          markerEnd={`url(#${id}-v)`}
        />
        {['M185 362 V345', 'M185 299 V285', 'M185 239 V225', 'M185 179 V155', 'M185 107 V91'].map(
          (d, index) => (
            <path
              key={d}
              className={`diagram-wire main-wire ${step >= [1, 2, 3, 4, 4][index] ? 'lit' : ''}`}
              d={d}
              markerEnd={`url(#${id}-p)`}
            />
          ),
        )}
        {/* Between the Mask box (right edge 284) and the V wire (x 425); longer translations
            wrap away from each other: the label upwards, the note downwards. */}
        <FitText
          x="355"
          y="303"
          textAnchor="middle"
          className="diagram-bypass-label"
          width={128}
          grow="up"
        >
          {T('V 不参与打分')}
        </FitText>
        <FitText
          x="350"
          y="326"
          textAnchor="middle"
          className="diagram-bypass-note"
          width={118}
          grow="down"
        >
          {T('留到最后，按比例取内容')}
        </FitText>
        {node(270, 611, 194, 'X', T('各位置当前的表示'), 0)}
        {node(85, 530, 119, '× W_Q', T('生成 Query'), 0, 'query', '\\times W_Q')}
        {node(245, 530, 119, '× W_K', T('生成 Key'), 0, 'key', '\\times W_K')}
        {node(425, 530, 119, '× W_V', T('生成 Value'), 0, 'value', '\\times W_V')}
        {node(85, 460, 100, 'Q', T('要找什么'), 0, 'query')}
        {node(245, 460, 100, 'K', T('用什么来匹配'), 0, 'key')}
        {node(425, 460, 100, 'V', T('实际取出的内容'), 0, 'value')}
        {node(
          185,
          385,
          198,
          'MatMul · QKᵀ',
          T('逐对打分'),
          1,
          'neutral',
          '\\mathrm{MatMul}\\cdot QK^{\\top}',
        )}
        {node(
          185,
          322,
          198,
          scaled ? 'Scale · ÷ √d_k' : T('Scale · 对照关闭'),
          scaled ? T('缩放分数') : T('保留未缩放分数'),
          1,
          'neutral',
          scaled ? '\\mathrm{Scale}\\cdot\\div\\sqrt{d_k}' : undefined,
        )}
        {node(
          185,
          262,
          198,
          causal ? 'Mask' : T('Mask · 当前关闭'),
          causal ? T('挡住未来位置') : T('双向读取对照'),
          2,
        )}
        {node(185, 202, 198, 'Softmax → P', T('分配取用比例'), 3, 'probability')}
        {node(
          185,
          130,
          198,
          'MatMul · P × V',
          T('按比例混合内容'),
          4,
          'probability',
          '\\mathrm{MatMul}\\cdot P\\times V',
        )}
        {node(185, 65, 198, T('单头输出 O'), T('新的上下文向量'), 4, 'probability')}
      </svg>
      <div className="diagram-multihead">
        <strong>{T('放回完整 MHA')}</strong>
        <p>
          {Tx(
            '投影 → 拆成多个头 → {#0} → Concat 拼接 → {#1} 输出投影 → 返回残差连接',
            <em>{T('每头执行上图')}</em>,
            <Tex tex="W_O" />,
          )}
        </p>
        <small>
          {Tx('上图只演算一个头；拆头、拼接、{#0} 的数值演算在“多头注意力”。', <Tex tex="W_O" />)}
        </small>
        {onOpenHeads && (
          <button className="diagram-heads-link" onClick={onOpenHeads}>
            {T('看多头怎样切分与合并 →')}
          </button>
        )}
      </div>
    </div>
  );
}
