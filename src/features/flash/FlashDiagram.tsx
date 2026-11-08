import { formatNumber as fmt } from '../../math/attention';
import { Segment } from '../../components/Controls';
import { vec, type FlashProps } from './FlashWidgets';

export function FlashDiagram({ state, result, onChange }: FlashProps) {
  const t = state.input.query;
  const blocks = result.rows[t];
  const b = blocks[state.input.block];
  const full = state.step === 0;
  return (
    <div className="flash-diagram">
      <div className="flash-diagram-heading">
        <strong>
          {state.step === 5
            ? '数据留在哪里？'
            : full
              ? '完整行作为核对基线'
              : `Query t${t} · 读到第 ${b.index + 1} 块`}
        </strong>
        <span>行 Query ↓ · 列 Key →</span>
      </div>
      {state.step === 5 ? (
        <MemoryDiagram />
      ) : (
        <svg
          className="flash-svg"
          viewBox="0 0 440 370"
          role="img"
          aria-label={
            full
              ? '完整分数与概率矩阵，沿 Key 轴做 Softmax'
              : `分块图：第${b.index + 1}块含Key ${b.keys.map((j) => 't' + j).join('、')}，当前最大值${fmt(b.m)}，归一化总量${fmt(b.l)}`
          }
        >
          <title>
            {full
              ? '完整 S / P 热力图，选中 Query 行'
              : '教学调度：从 HBM 读 K / V 块，在片上更新 m、l、a'}
          </title>
          <text x="122" y="22" className="flash-svg-title" textAnchor="middle">
            S + M · [4, 4]
          </text>
          {full && (
            <text x="330" y="22" className="flash-svg-title" textAnchor="middle">
              P · [4, 4]
            </text>
          )}
          {[0, 1, 2, 3].map((i) => (
            <g key={i}>
              <text x={65 + i * 38} y="43" textAnchor="middle" className="flash-svg-small">
                t{i}
              </text>
              <text x="28" y={72 + i * 37} textAnchor="middle" className="flash-svg-small">
                t{i}
              </text>
              {[0, 1, 2, 3].map((j) => {
                const masked = result.reference.masked[i][j] === -Infinity;
                const active = i === t && (full || b.keys.includes(j));
                return (
                  <g key={j}>
                    <rect
                      x={47 + j * 38}
                      y={52 + i * 37}
                      width="36"
                      height="34"
                      rx="2"
                      className={`flash-score ${masked ? 'flash-masked' : ''} ${active ? 'flash-active' : ''}`}
                    />
                    <text
                      x={65 + j * 38}
                      y={73 + i * 37}
                      textAnchor="middle"
                      className={`flash-svg-number ${active ? 'flash-svg-strong' : ''}`}
                    >
                      {fmt(result.reference.masked[i][j], 2)}
                    </text>
                    {full && (
                      <>
                        <rect
                          x={255 + j * 38}
                          y={52 + i * 37}
                          width="36"
                          height="34"
                          rx="2"
                          className={`flash-probability ${i === t ? 'flash-active' : ''}`}
                          fillOpacity={0.2 + 0.8 * result.reference.probabilities[i][j]}
                        />
                        <text
                          x={273 + j * 38}
                          y={73 + i * 37}
                          textAnchor="middle"
                          className="flash-svg-number"
                        >
                          {fmt(result.reference.probabilities[i][j], 2)}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
              {full && (
                <text x={273 + i * 38} y="43" textAnchor="middle" className="flash-svg-small">
                  t{i}
                </text>
              )}
            </g>
          ))}
          {!full && (
            <>
              <text x="325" y="43" textAnchor="middle" className="flash-svg-small">
                HBM · 完整 K / V
              </text>
              {[0, 1, 2, 3].map((j) => (
                <g key={j}>
                  <rect
                    x="245"
                    y={52 + j * 37}
                    width="170"
                    height="34"
                    rx="2"
                    className={`flash-kv ${b.keys.includes(j) ? 'flash-active' : ''}`}
                  />
                  <text x="253" y={65 + j * 37} className="flash-svg-number">
                    t{j} · K {vec(result.k[j])}
                  </text>
                  <text x="277" y={79 + j * 37} className="flash-svg-number">
                    V {vec(result.v[j])}
                  </text>
                </g>
              ))}
              <path className="flash-wire" d="M330 202 V228 H220 V244" />
              <text x="338" y="222" className="flash-svg-small">
                读取 {b.keys.length} 行
              </text>
            </>
          )}
          {full ? (
            <>
              <path className="flash-wire" d="M202 120 H249" />
              <text x="224" y="109" textAnchor="middle" className="flash-svg-small">
                行归一化
              </text>
              <path className="flash-wire" d="M330 202 V240 H220 V261" />
              <text x="220" y="236" textAnchor="middle" className="flash-svg-small">
                每行 P × V
              </text>
              <rect x="42" y="263" width="356" height="64" rx="3" className="flash-output" />
              <text x="220" y="289" textAnchor="middle" className="flash-svg-title">
                O[t{t}] = {vec(result.reference.output[t])}
              </text>
              <text x="220" y="311" textAnchor="middle" className="flash-svg-small">
                基线保留完整 S 与 P · 每张 [4, 4]
              </text>
            </>
          ) : (
            <>
              <rect x="25" y="246" width="390" height="98" rx="3" className="flash-onchip" />
              <text x="220" y="266" textAnchor="middle" className="flash-svg-small">
                片上 SRAM / 寄存器 · 示意
              </text>
              {[
                ['m', fmt(b.m)],
                ['l', fmt(b.l, 4)],
                ['a [1, 2]', vec(b.a)],
              ].map(([label, value], i) => (
                <g key={label}>
                  <rect
                    x={38 + i * 126}
                    y="277"
                    width="115"
                    height="53"
                    rx="2"
                    className="flash-state-node"
                  />
                  <text x={95 + i * 126} y="294" textAnchor="middle" className="flash-svg-small">
                    {label}
                  </text>
                  <text
                    x={95 + i * 126}
                    y="317"
                    textAnchor="middle"
                    className="flash-svg-number flash-svg-strong"
                  >
                    {value}
                  </text>
                </g>
              ))}
              <text x="220" y="364" textAnchor="middle" className="flash-svg-small">
                完整 S / P 不写回 HBM；本图额外画出 S 便于核对
              </text>
            </>
          )}
        </svg>
      )}
      {!full && state.step !== 5 && (
        <>
          <Segment
            label="查看完成的块"
            options={blocks.map((block) => block.index)}
            value={b.index}
            onSelect={(block) => onChange({ ...state, input: { ...state.input, block } })}
            render={(index) =>
              `块 ${index + 1} · ${blocks[index].keys.map((j) => 't' + j).join('、')}`
            }
          />
          <div className="flash-block-controls">
            <button
              disabled={b.index === 0}
              onClick={() => onChange({ ...state, input: { ...state.input, block: b.index - 1 } })}
            >
              ← 前一块
            </button>
            <button
              disabled={b.index === blocks.length - 1}
              onClick={() => onChange({ ...state, input: { ...state.input, block: b.index + 1 } })}
            >
              读下一块 →
            </button>
          </div>
          <p className="flash-readout" aria-label="当前块状态">
            {b.validCount === 0
              ? '全屏蔽块 · 保留旧状态'
              : b.rescaled
                ? '更大最大值 · 旧累积量重缩放'
                : '当前块已累积'}
            <br />m = {fmt(b.m)} · l = {fmt(b.l, 5)}
            <br />
            {b.index === blocks.length - 1 ? '最终输出' : '前缀暂时输出'} a / l = {vec(b.output)}
          </p>
        </>
      )}
      <p className="exp-caption">
        {state.step === 5
          ? '箭头表示数据搬运，不表示耗时。真实 kernel 的块形状、复用与并行调度更复杂。'
          : '格子是实际数值，橙框标出当前 Query / Key 块。切换上方 Query 查看另一行；屏蔽项以 −∞ 标明。'}
      </p>
    </div>
  );
}

function MemoryDiagram() {
  return (
    <svg
      className="flash-svg"
      viewBox="0 0 440 390"
      role="img"
      aria-label="朴素路径将S和P写回HBM，分块路径在片上直接累积并写出O"
    >
      <title>两条计算路径的中间矩阵去向</title>
      <text x="220" y="24" textAnchor="middle" className="flash-svg-title">
        朴素调度：完整中间矩阵往返
      </text>
      <rect x="20" y="44" width="400" height="108" className="flash-kv" rx="3" />
      <text x="40" y="67" className="flash-svg-small">
        HBM
      </text>
      <text x="220" y="91" textAnchor="middle" className="flash-svg-title">
        Q、K、V　＋　S [T, T]、P [T, T]
      </text>
      <text x="220" y="123" textAnchor="middle" className="flash-svg-small">
        S 写入 → 读出归一化 → P 写入 → 读出汇总 V
      </text>
      <text x="220" y="199" textAnchor="middle" className="flash-svg-title">
        逐行分块：块用完即丢弃
      </text>
      <rect x="20" y="218" width="400" height="56" className="flash-kv" rx="3" />
      <text x="220" y="252" textAnchor="middle" className="flash-svg-title">
        HBM · Q、K、V 与最终 O
      </text>
      <path className="flash-wire" d="M110 274 V306 M330 306 V274" />
      <text x="123" y="296" className="flash-svg-small">
        读块
      </text>
      <text x="340" y="296" className="flash-svg-small">
        写 O
      </text>
      <rect x="20" y="308" width="400" height="65" className="flash-onchip" rx="3" />
      <text x="220" y="335" textAnchor="middle" className="flash-svg-title">
        片上 · 分数块 → 更新 m、l、a
      </text>
      <text x="220" y="358" textAnchor="middle" className="flash-svg-small">
        循环读块，最后 O = a / l；无需保存完整 S、P
      </text>
    </svg>
  );
}
