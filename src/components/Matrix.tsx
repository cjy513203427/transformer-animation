import { useId, type CSSProperties } from 'react';
import { formatNumber, type Matrix as MatrixData } from '../math/attention';
import { Tex } from './Tex';
import { T } from '../i18n';
import '../styles/attention-interactions.css';

/** Parameter names like W_Q are typeset as W with a subscript; aria-labels keep the plain name. */
const SUBSCRIPTED = /^[A-Za-z]+_[A-Za-z]+$/;

interface Props {
  name: string;
  data: MatrixData;
  tone?: 'query' | 'key' | 'value' | 'probability' | 'neutral';
  row?: number;
  column?: number;
  heatmap?: boolean;
  tokenColumns?: boolean;
  parameter?: boolean;
  onSelect?: (row: number, column: number) => void;
}

export function Matrix({
  name,
  data,
  tone = 'neutral',
  row,
  column,
  heatmap,
  tokenColumns,
  parameter,
  onSelect,
}: Props) {
  const hintId = useId();
  return (
    <div
      className={`matrix-wrap tone-${tone} ${onSelect ? 'matrix-selectable' : 'matrix-readonly'}`}
    >
      <div className="matrix-heading">
        <strong>
          {SUBSCRIPTED.test(name) ? <Tex tex={name.replace('_', '_{') + '}'} /> : name}
        </strong>
        <div className="matrix-meta">
          <span>
            {data.length} × {data[0].length}
          </span>
          <small>{onSelect ? T('可点选') : T('只读数值')}</small>
        </div>
      </div>
      {onSelect && (
        <p id={hintId} className="matrix-interaction-hint">
          {T('点选格子，核对这一项')}
        </p>
      )}
      <div className="matrix-scroll">
        <table
          className="matrix"
          aria-label={T('{#0} 矩阵', name)}
          aria-describedby={onSelect ? hintId : undefined}
        >
          <thead>
            <tr>
              <th scope="col">{parameter ? 'in ↓ out →' : tokenColumns ? 'Q ↓ K →' : 'token'}</th>
              {data[0].map((_, j) => (
                <th key={j} scope="col">
                  {tokenColumns ? `t${j}` : `d${j}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((values, i) => (
              <tr key={i} className={i === row ? 'selected-row' : ''}>
                <th scope="row">
                  {parameter ? 'd' : 't'}
                  {i}
                </th>
                {values.map((value, j) => (
                  <td key={j}>
                    {onSelect ? (
                      <button
                        className={`matrix-cell ${i === row && j === column ? 'selected-cell' : ''} ${value === -Infinity ? 'masked-cell' : ''}`}
                        style={
                          heatmap
                            ? ({ '--intensity': Math.max(0, value) } as CSSProperties)
                            : undefined
                        }
                        aria-label={`${name}[${i},${j}] = ${formatNumber(value)}`}
                        aria-pressed={i === row && j === column}
                        onClick={() => onSelect(i, j)}
                      >
                        {formatNumber(value)}
                      </button>
                    ) : (
                      <span className="matrix-cell">{formatNumber(value)}</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
