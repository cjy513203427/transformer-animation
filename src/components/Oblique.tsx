import '../styles/oblique.css';
/**
 * Oblique (斜二测) drawing of a block of unit cells, for tensors that really have a third axis.
 * The front face stays an upright grid that can carry numbers; depth runs up and to the right, so
 * no perspective changes the size of a cell. Only the three visible faces are drawn.
 */
export interface CuboidProps {
  /** Screen position of the front face's top-left corner. */
  x: number;
  y: number;
  cols: number;
  rows: number;
  depth: number;
  /** Width and height of one front cell. */
  cell: [number, number];
  /** Screen offset of one unit of depth: [right, up]. */
  step: [number, number];
  /** Class of a cell, by its column and row on the front face. Top and side faces reuse it. */
  fill: (col: number, row: number) => string;
  /** Text written on a front cell, e.g. its value. */
  text?: (col: number, row: number) => string;
  textClass?: string;
}
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
export function Cuboid({
  x,
  y,
  cols,
  rows,
  depth,
  cell: [cw, ch],
  step: [sx, sy],
  fill,
  text,
  textClass = 'blk-text',
}: CuboidProps) {
  const p = (c: number, r: number, k: number) => `${x + c * cw + k * sx},${y + r * ch - k * sy}`;
  const quad = (...corners: string[]) => corners.join(' ');
  return (
    <g>
      {range(cols).flatMap((c) =>
        range(depth).map((k) => (
          <polygon
            key={`t${c}-${k}`}
            points={quad(p(c, 0, k), p(c + 1, 0, k), p(c + 1, 0, k + 1), p(c, 0, k + 1))}
            className={fill(c, 0)}
          />
        )),
      )}
      <polygon
        points={quad(p(0, 0, 0), p(cols, 0, 0), p(cols, 0, depth), p(0, 0, depth))}
        className="blk-shade-top"
      />
      {range(rows).flatMap((r) =>
        range(depth).map((k) => (
          <polygon
            key={`s${r}-${k}`}
            points={quad(
              p(cols, r, k),
              p(cols, r, k + 1),
              p(cols, r + 1, k + 1),
              p(cols, r + 1, k),
            )}
            className={fill(cols - 1, r)}
          />
        )),
      )}
      <polygon
        points={quad(p(cols, 0, 0), p(cols, 0, depth), p(cols, rows, depth), p(cols, rows, 0))}
        className="blk-shade-side"
      />
      {range(rows).flatMap((r) =>
        range(cols).map((c) => (
          <g key={`f${r}-${c}`}>
            <rect x={x + c * cw} y={y + r * ch} width={cw} height={ch} className={fill(c, r)} />
            {text && (
              <text
                x={x + c * cw + cw / 2}
                y={y + r * ch + ch / 2 + 3.5}
                textAnchor="middle"
                className={textClass}
              >
                {text(c, r)}
              </text>
            )}
          </g>
        )),
      )}
    </g>
  );
}
