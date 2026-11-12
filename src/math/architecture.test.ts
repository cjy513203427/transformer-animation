import { expect, it } from 'vitest';
import { architectureLedger, structuredRecurrence } from './architecture';
import { initialArchitectureState } from '../state/architecture';

it('KV 随 token 增长，SSM 与卷积历史状态固定；MoE 独立于序列混合器', () => {
  const c = initialArchitectureState().config,
    a = architectureLedger(c);
  const doubled = architectureLedger({ ...c, tokens: c.tokens * 2 });
  expect(doubled.kvBytes).toBe(a.kvBytes * 2);
  expect(doubled.ssmBytes).toBe(a.ssmBytes);
  expect(doubled.convBytes).toBe(a.convBytes);
  expect(a.kvBytes).toBe(1 * 1 * 1024 * 2 * (2 + 2) * 2);
  expect(a.ssmBytes + a.convBytes).toBe(7 * 8 * (4 + 3) * 2);
  const moe = architectureLedger({ ...c, moe: c.moe.map(() => false) });
  expect(moe.historyBytes).toBe(a.historyBytes);
  expect(a.allFfnParameters).toBeGreaterThan(a.activeFfnParameters);
  expect(architectureLedger({ ...c, mixers: c.mixers.map(() => 'ssm') }).kvBytes).toBe(0);
});

it('结构化矩阵与递推相同，包含零保留、完全保留及因果边界', () => {
  for (const a of [0, 0.1, 0.5, 0.9, 1])
    for (const x of [
      [1, 0, 2, -1],
      [-4, 4, -2, 2],
    ]) {
      const r = structuredRecurrence(x, a);
      expect(r.error).toBeLessThanOrEqual(1e-12);
      r.matrix.forEach((row, t) =>
        row.forEach((v, j) => {
          if (j > t) expect(v).toBe(0);
          if (j === t) expect(v).toBe(1);
        }),
      );
      const changed = structuredRecurrence([...x.slice(0, 3), 4], a);
      expect(changed.recurrent.slice(0, 3)).toEqual(r.recurrent.slice(0, 3));
    }
  expect(structuredRecurrence([1, 0, 2, -1], 0.5).recurrent).toEqual([1, 0.5, 2.25, 0.125]);
});
