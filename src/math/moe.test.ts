import { describe, expect, it } from 'vitest';
import { computeMoE, directReference, makeMoEWeights, selectExperts, type MoEOptions } from './moe';

const TOL = 1e-12;
const base: MoEOptions = {
  seed: 42,
  config: { topK: 2, topology: 'split' },
  input: { preset: 'balanced', token: 0, delta: 0, expertScale: 1 },
};
const maxDiff = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((row, t) => row.map((v, d) => Math.abs(v - b[t][d]))));

describe('MoE Router 与门控', () => {
  it('稳定 Softmax、Top-K 和选中后归一化；分数平移不改变结果', () => {
    const r = selectExperts([1001, 1000, 999, -1000], 2);
    expect(r.selected).toEqual([0, 1]);
    expect(r.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(r.gates.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(r.gates[0]).toBeCloseTo(1 / (1 + Math.exp(-1)), 12);
    expect(r.gates.slice(2)).toEqual([0, 0]);
    expect(selectExperts([1, 0, -1, -2000], 2)).toEqual(r);
    expect(selectExperts([1001, 1000, 999, -1000], 1).gates).toEqual([1, 0, 0, 0]);
  });

  it('相同分数按编号选取；未选专家不是被归一化后的稠密输出', () => {
    expect(selectExperts([0, 0, 0, 0], 2)).toEqual({
      probabilities: [0.25, 0.25, 0.25, 0.25],
      selected: [0, 1],
      gates: [0.5, 0.5, 0, 0],
    });
    expect(() => selectExperts([0, NaN, 0, 0], 2)).toThrow();
    expect(() => selectExperts([0, 0], 2)).toThrow();
  });

  it('固定种子得到同一组权重；不同种子会变，修改返回值不污染后续实验', () => {
    const a = makeMoEWeights(42);
    expect(makeMoEWeights(42)).toEqual(a);
    expect(makeMoEWeights(43)).not.toEqual(a);
    a.router[0][0] = 999;
    expect(makeMoEWeights(42).router[0][0]).not.toBe(999);
  });
});

describe('MoE 分发、FFN 与按 token 收集', () => {
  it('专家优先 dispatch/combine 与独立逐 token 基线一致，覆盖参数、拓扑与负载边界', () => {
    for (const seed of [1, 42, 9999])
      for (const topK of [1, 2] as const)
        for (const topology of ['single', 'split'] as const)
          for (const preset of ['balanced', 'concentrated'] as const)
            for (const delta of [-2, 0, 2]) {
              const options: MoEOptions = {
                seed,
                config: { topK, topology },
                input: { preset, token: 3, delta, expertScale: 1.75 },
              };
              const result = computeMoE(options);
              expect(maxDiff(result.output, directReference(options))).toBeLessThan(TOL);
              expect(result.packets).toHaveLength(4 * topK);
              expect(result.load.reduce((a, b) => a + b, 0)).toBe(4 * topK);
              result.packets.forEach((packet) => {
                expect(packet.input).toEqual(result.x[packet.token]);
                expect(packet.hidden.every((v) => v >= 0)).toBe(true);
                expect(result.selected[packet.token]).toContain(packet.expert);
              });
              result.expertOutputs.forEach((row, t) =>
                row.forEach((out, e) => expect(out === null).toBe(!result.selected[t].includes(e))),
              );
            }
  });

  it('专家 E0 输出倍率只影响其贡献，完全不改变 Router 分数、选择或门控', () => {
    const before = computeMoE(base);
    const after = computeMoE({ ...base, input: { ...base.input, expertScale: 2 } });
    expect(after.logits).toEqual(before.logits);
    expect(after.probabilities).toEqual(before.probabilities);
    expect(after.selected).toEqual(before.selected);
    expect(after.gates).toEqual(before.gates);
    expect(after.load).toEqual(before.load);
    expect(
      before.contributions
        .flat()
        .filter((v) => v !== null)
        .flat()
        .some((v) => v !== 0),
    ).toBe(true);
    after.output.forEach((row, t) =>
      row.forEach((v, d) => {
        expect(v - before.output[t][d]).toBeCloseTo(before.contributions[t][0]?.[d] ?? 0, 12);
      }),
    );
  });

  it('只改 t1 输入，只改变该 token 的路由与输出；其他行原样保留', () => {
    const a = computeMoE(base);
    const b = computeMoE({ ...base, input: { ...base.input, token: 1, delta: 2 } });
    for (const t of [0, 2, 3]) {
      expect(b.logits[t]).toEqual(a.logits[t]);
      expect(b.output[t]).toEqual(a.output[t]);
    }
    expect(b.logits[1]).not.toEqual(a.logits[1]);
  });

  it('分散与集中输入具有不同负载，但总任务数相同；Top-K不强制均衡', () => {
    expect(computeMoE(base).load).toEqual([2, 2, 2, 2]);
    const concentrated = computeMoE({ ...base, input: { ...base.input, preset: 'concentrated' } });
    expect(concentrated.load).toEqual([4, 4, 0, 0]);
    expect(concentrated.load.reduce((a, b) => a + b, 0)).toBe(8);
  });
});

describe('参数、矩阵 MAC 与通信账本', () => {
  it('Top-K 只改变激活参数和专家矩阵运算，不改变全部存储参数与Router运算', () => {
    const one = computeMoE({ ...base, config: { ...base.config, topK: 1 } }).counts;
    const two = computeMoE(base).counts;
    expect(two.totalParams).toBe(8 + 4 * 12);
    expect(one.totalParams).toBe(two.totalParams);
    expect(one.activeParamsPerToken).toBe(8 + 12);
    expect(two.activeParamsPerToken).toBe(8 + 24);
    expect(two.totalExpertMACs).toBe(4 * 2 * 12);
    expect(one.totalExpertMACs).toBe(two.totalExpertMACs / 2);
    expect(two.totalRouterMACs).toBe(4 * 8);
    expect(one.totalRouterMACs).toBe(two.totalRouterMACs);
  });

  it('改变专家部署不改变输出；仅计算跨设备的两个D维激活向量', () => {
    const split = computeMoE(base);
    const single = computeMoE({ ...base, config: { ...base.config, topology: 'single' } });
    expect(split.output).toEqual(single.output);
    expect(split.traffic.remoteAssignments).toBe(4);
    expect(split.traffic.dispatchBytes).toBe(4 * 2 * 4);
    expect(split.traffic.returnBytes).toBe(4 * 2 * 4);
    expect(split.traffic.totalBytes).toBe(64);
    expect(single.traffic).toEqual({
      remoteAssignments: 0,
      dispatchBytes: 0,
      returnBytes: 0,
      totalBytes: 0,
    });
    split.packets.forEach((p) => {
      expect(p.homeDevice).toBe(p.token % 2);
      expect(p.expertDevice).toBe(Math.floor(p.expert / 2));
      expect(p.remote).toBe(p.homeDevice !== p.expertDevice);
    });
  });

  it('共享专家、容量上限丢弃与独立参考实现完全一致，并正确统计负载均衡辅助损失', () => {
    for (const sharedExpert of [false, true]) {
      for (const capacity of [0, 1, 2, 3] as const) {
        for (const preset of ['balanced', 'concentrated'] as const) {
          const options: MoEOptions = {
            seed: 42,
            config: { topK: 2, topology: 'split', sharedExpert, capacity },
            input: { preset, token: 0, delta: 0, expertScale: 1.5 },
          };
          const result = computeMoE(options);
          expect(maxDiff(result.output, directReference(options))).toBeLessThan(TOL);
          expect(result.packets.length + result.droppedPackets.length).toBe(8);
          if (capacity > 0) {
            result.load.forEach((n) => expect(n).toBeLessThanOrEqual(capacity));
          }
          if (sharedExpert) {
            expect(result.counts.sharedParams).toBe(12);
            expect(result.counts.totalParams).toBe(8 + 5 * 12);
          }
        }
      }
    }
    const balanced = computeMoE(base);
    const concentrated = computeMoE({ ...base, input: { ...base.input, preset: 'concentrated' } });
    expect(concentrated.balance.balanceFactor).toBeGreaterThan(balanced.balance.balanceFactor);
    expect(concentrated.balance.auxLoss).toBeCloseTo(
      concentrated.balance.auxAlpha * concentrated.balance.balanceFactor,
      12,
    );
    const capped = computeMoE({
      ...base,
      config: { topK: 2, topology: 'split', sharedExpert: false, capacity: 2 },
      input: { ...base.input, preset: 'concentrated' },
    });
    expect(capped.counts.droppedCount).toBe(4);
    expect(capped.load).toEqual([2, 2, 0, 0]);
  });
});
