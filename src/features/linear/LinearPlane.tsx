import { useRef, useState, type PointerEvent } from 'react';
import { FitText } from '../../components/FitText';
import { formatNumber as fmt } from '../../math/attention';
import type { LinearResult, Vec2 } from '../../math/linear';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

const SIZE = 420;
const HALF = SIZE / 2;
const RANGE = 3.2;
const SCALE = (SIZE - 44) / (2 * RANGE);

const toSvgX = (x: number) => HALF + x * SCALE;
const toSvgY = (y: number) => HALF - y * SCALE;
const fromSvg = (px: number, py: number): Vec2 => {
  const snap = (v: number) => Math.max(-2, Math.min(2, Math.round(v * 4) / 4));
  return [snap((px - HALF) / SCALE), snap((HALF - py) / SCALE)];
};

function Arrow({
  from = [0, 0],
  to,
  className,
  label,
  offset = [8, -8],
}: {
  from?: Vec2;
  to: Vec2;
  className: string;
  label?: string;
  offset?: [number, number];
}) {
  const x1 = toSvgX(from[0]);
  const y1 = toSvgY(from[1]);
  const x2 = toSvgX(to[0]);
  const y2 = toSvgY(to[1]);
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  const ux = len > 1e-6 ? dx / len : 0;
  const uy = len > 1e-6 ? dy / len : 0;
  const head = 8;
  const ax = x2 - ux * head - uy * 4;
  const ay = y2 - uy * head + ux * 4;
  const bx = x2 - ux * head + uy * 4;
  const by = y2 - uy * head - ux * 4;
  return (
    <g className={className}>
      {len > 1 && (
        <>
          <line x1={x1} y1={y1} x2={x2} y2={y2} />
          <polygon points={`${x2},${y2} ${ax},${ay} ${bx},${by}`} />
        </>
      )}
      {label && (
        <text
          x={Math.max(18, Math.min(SIZE - 18, x2 + offset[0]))}
          y={Math.max(18, Math.min(SIZE - 12, y2 + offset[1]))}
          className="lin-vec-label"
        >
          {label}
        </text>
      )}
    </g>
  );
}

type HandleId = 'e0' | 'e1' | 'x' | 'k';

export function LinearPlane({
  step,
  result,
  onDrag,
}: {
  step: number;
  result: LinearResult;
  onDrag: (handle: HandleId, value: Vec2) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<HandleId | null>(null);

  function pointerToCoords(event: PointerEvent<SVGSVGElement>): Vec2 | null {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const px = ((event.clientX - rect.left) / rect.width) * SIZE;
    const py = ((event.clientY - rect.top) / rect.height) * SIZE;
    return fromSvg(px, py);
  }

  function startDrag(handle: HandleId, event: PointerEvent<SVGCircleElement>) {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(handle);
  }

  function moveDrag(event: PointerEvent<SVGSVGElement>) {
    if (!dragging) return;
    const next = pointerToCoords(event);
    if (next) onDrag(dragging, next);
  }

  function endDrag() {
    setDragging(null);
  }

  const gridTicks = [-3, -2, -1, 0, 1, 2, 3];
  const { e0, e1, x, y, k, unitSquare, dotInfo, step1, step2, reversedY } = result;
  const polyPoints = unitSquare.map(([px, py]) => `${toSvgX(px)},${toSvgY(py)}`).join(' ');

  return (
    <div className="lin-plane-wrap">
      <svg
        ref={svgRef}
        className="lin-svg"
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label={T(
          '二维线性变换平面：基向量 e₀\'=[{#0},{#1}]，e₁\'=[{#2},{#3}]，输出 y=[{#4},{#5}]',
          fmt(e0[0]),
          fmt(e0[1]),
          fmt(e1[0]),
          fmt(e1[1]),
          fmt(y[0]),
          fmt(y[1]),
        )}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <title>{T('二维坐标网格与线性变换结果；拖动圆点端点可交互改变向量或矩阵行')}</title>
        {/* Original faint coordinate grid */}
        {gridTicks.map((t) => (
          <g key={`orig-${t}`}>
            <line
              x1={toSvgX(t)}
              y1={16}
              x2={toSvgX(t)}
              y2={SIZE - 16}
              className={t === 0 ? 'lin-axis' : 'lin-grid-orig'}
            />
            <line
              x1={16}
              y1={toSvgY(t)}
              x2={SIZE - 16}
              y2={toSvgY(t)}
              className={t === 0 ? 'lin-axis' : 'lin-grid-orig'}
            />
          </g>
        ))}

        {/* Transformed grid lines on steps 0, 1, 4 */}
        {(step === 0 || step === 1 || step === 4) &&
          [-2, -1, 0, 1, 2].map((c) => {
            const p1: Vec2 = [c * e0[0] - 3 * e1[0], c * e0[1] - 3 * e1[1]];
            const p2: Vec2 = [c * e0[0] + 3 * e1[0], c * e0[1] + 3 * e1[1]];
            const q1: Vec2 = [-3 * e0[0] + c * e1[0], -3 * e0[1] + c * e1[1]];
            const q2: Vec2 = [3 * e0[0] + c * e1[0], 3 * e0[1] + c * e1[1]];
            return (
              <g key={`trans-${c}`}>
                <line
                  x1={toSvgX(p1[0])}
                  y1={toSvgY(p1[1])}
                  x2={toSvgX(p2[0])}
                  y2={toSvgY(p2[1])}
                  className={c === 0 ? 'lin-grid-trans-zero' : 'lin-grid-trans'}
                />
                <line
                  x1={toSvgX(q1[0])}
                  y1={toSvgY(q1[1])}
                  x2={toSvgX(q2[0])}
                  y2={toSvgY(q2[1])}
                  className={c === 0 ? 'lin-grid-trans-zero' : 'lin-grid-trans'}
                />
              </g>
            );
          })}

        {/* Unit square vs transformed parallelogram on step 1 (and subtle on step 0) */}
        {(step === 0 || step === 1) && (
          <>
            <rect
              x={toSvgX(0)}
              y={toSvgY(1)}
              width={SCALE}
              height={SCALE}
              className="lin-unit-orig"
            />
            <polygon
              points={polyPoints}
              className={
                result.orientation === 'flipped'
                  ? 'lin-unit-trans lin-unit-flipped'
                  : result.orientation === 'collapsed'
                    ? 'lin-unit-trans lin-unit-collapsed'
                    : 'lin-unit-trans'
              }
            />
          </>
        )}

        {/* Step 2: dot product projection geometry */}
        {step === 2 && (
          <>
            {dotInfo.normK > 0 && (
              <>
                <line
                  x1={toSvgX(-k[0] * 3)}
                  y1={toSvgY(-k[1] * 3)}
                  x2={toSvgX(k[0] * 3)}
                  y2={toSvgY(k[1] * 3)}
                  className="lin-proj-axis"
                />
                <line
                  x1={toSvgX(y[0])}
                  y1={toSvgY(y[1])}
                  x2={toSvgX(dotInfo.projOnK[0])}
                  y2={toSvgY(dotInfo.projOnK[1])}
                  className="lin-proj-drop"
                />
                <line
                  x1={toSvgX(0)}
                  y1={toSvgY(0)}
                  x2={toSvgX(dotInfo.projOnK[0])}
                  y2={toSvgY(dotInfo.projOnK[1])}
                  className="lin-proj-seg"
                />
              </>
            )}
            <Arrow to={k} className="lin-vec lin-vec-k" label="k" offset={[8, 12]} />
            <Arrow to={y} className="lin-vec lin-vec-y" label="q=xW" offset={[8, -8]} />
          </>
        )}

        {/* Step 3: composition path x -> xW1 -> (xW1)W2 vs x(W2W1) */}
        {step === 3 && (
          <>
            <Arrow to={x} className="lin-vec lin-vec-x" label="x" offset={[-16, -8]} />
            <Arrow to={step1} className="lin-vec lin-vec-e0" label="xW₁" offset={[8, -8]} />
            <Arrow to={step2} className="lin-vec lin-vec-y" label="xW₁W₂" offset={[8, 12]} />
            <Arrow to={reversedY} className="lin-vec lin-vec-k" label="xW₂W₁" offset={[-46, 12]} />
          </>
        )}

        {/* Steps 0, 1, 4: basis vectors e0', e1' and input x -> output y */}
        {(step === 0 || step === 1 || step === 4) && (
          <>
            <Arrow to={x} className="lin-vec lin-vec-x" label="x" offset={[-16, -8]} />
            <Arrow to={e0} className="lin-vec lin-vec-e0" label="e₀'" offset={[8, 12]} />
            <Arrow to={e1} className="lin-vec lin-vec-e1" label="e₁'" offset={[-24, -8]} />
            <Arrow to={y} className="lin-vec lin-vec-y" label="y=xW" offset={[8, -8]} />
          </>
        )}

        {/* Origin dot */}
        <circle cx={HALF} cy={HALF} r="3" className="lin-origin" />

        {/* Interactive drag handles */}
        {(step === 0 || step === 1 || step === 4) && (
          <>
            <circle
              cx={toSvgX(e0[0])}
              cy={toSvgY(e0[1])}
              r="7"
              className="lin-handle lin-handle-e0"
              onPointerDown={(ev) => startDrag('e0', ev)}
            />
            <circle
              cx={toSvgX(e1[0])}
              cy={toSvgY(e1[1])}
              r="7"
              className="lin-handle lin-handle-e1"
              onPointerDown={(ev) => startDrag('e1', ev)}
            />
            <circle
              cx={toSvgX(x[0])}
              cy={toSvgY(x[1])}
              r="7"
              className="lin-handle lin-handle-x"
              onPointerDown={(ev) => startDrag('x', ev)}
            />
          </>
        )}
        {step === 2 && (
          <circle
            cx={toSvgX(k[0])}
            cy={toSvgY(k[1])}
            r="7"
            className="lin-handle lin-handle-k"
            onPointerDown={(ev) => startDrag('k', ev)}
          />
        )}

        {/* Corner badge */}
        <FitText x="14" y="20" width={190} maxLines={1} className="lin-badge">
          {step === 2
            ? T('q·k = {#0}', fmt(dotInfo.dot, 3))
            : step === 3
              ? T('‖W₁W₂ − W₂W₁‖ = {#0}', fmt(result.commuteGap, 3))
              : T('det(W) = {#0}', fmt(result.det, 3))}
        </FitText>
      </svg>
      <p className="exp-caption">
        {step === 2
          ? Tx(
              '绿色向量为 q = xW，琥珀色为参考向量 k（可拖动端点），虚线为 q 向 k 的正交投影。',
            )
          : step === 3
            ? Tx('蓝色为先经 W₁ 的中间向量，绿色为 x(W₁W₂)，琥珀色为反序 x(W₂W₁)。')
            : Tx(
                '可直接拖动蓝色 e₀\'、橙色 e₁\' 或灰色 x 的端点（步长 0.25），或使用右侧滑杆微调。',
              )}
      </p>
    </div>
  );
}
