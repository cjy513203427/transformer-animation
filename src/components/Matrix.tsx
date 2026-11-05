import type { CSSProperties } from 'react';
import { formatNumber, type Matrix as MatrixData } from '../math/attention';

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
  return (
    <div className={`matrix-wrap tone-${tone}`}>
      <div className="matrix-heading">
        <strong>{name}</strong>
        <span>
          {data.length} × {data[0].length}
        </span>
      </div>
      <div className="matrix-scroll">
        <table className="matrix" aria-label={`${name} 矩阵`}>
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
