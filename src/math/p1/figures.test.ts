import { describe, expect, it } from 'vitest';
import decoding from '../../content/p1/decoding';
import preferenceTopic from '../../content/p1/preference';
import ssmTopic from '../../content/p1/ssm';
import type { TopicDefinition } from '../../content/p1/types';
import { initialLabState } from '../../state/p1';
import { softmax } from '../attention';
import { ppoTerm, sigmoid } from './preference';

/* The figures only draw what the step computed; these checks tie each figure back to the math. */
const config = (topic: TopicDefinition) => initialLabState(topic).config;

describe('解码的专门图', () => {
  it('温度曲线在 τ = 1 处就是原始 Softmax，当前点落在本步的温度分布上', () => {
    const x = [2, 1, 0.5],
      run = decoding.compute(x, { ...config(decoding), temperature: 0.5 }, 42),
      plot = run.frames[1].plot!,
      raw = softmax(x),
      prob = run.frames[1].tensors[1].values[0];
    plot.curves.forEach((curve, i) => {
      const atOne = curve.points.find(([t]) => Math.abs(t - 1) < 1e-12)!;
      expect(atOne[1]).toBeCloseTo(raw[i], 12);
    });
    plot.points.forEach((p, i) => {
      expect(p.x).toBe(0.5);
      expect(p.y).toBe(prob[i]);
    });
  });

  it('Top-P 图：保留累计首次 ≥ P 的那一项及之前的项，保留项重新归一化', () => {
    for (const topP of [0.3, 0.6, 0.65, 0.9, 1]) {
      const run = decoding.compute([2, 1, 0.5], { ...config(decoding), topP }, 42),
        { items, threshold } = run.frames[3].ranking!;
      expect(threshold).toBe(topP);
      for (let i = 1; i < items.length; i++)
        expect(items[i - 1].before).toBeGreaterThanOrEqual(items[i].before);
      const cut = topP < 1 ? items.findIndex((item) => item.cumulative >= topP) : items.length - 1;
      items.forEach((item, i) => expect(item.kept, `P=${topP} 第${i}项`).toBe(i <= cut));
      const kept = items.filter((item) => item.kept),
        mass = kept.reduce((s, item) => s + item.before, 0);
      kept.forEach((item) => expect(item.after).toBeCloseTo(item.before / mass, 12));
      expect(kept.reduce((s, item) => s + item.after, 0)).toBeCloseTo(1, 12);
    }
  });

  it('Top-K 之后再做 Top-P：已删去的候选排在最后，不会被 Top-P 找回', () => {
    const run = decoding.compute([2, 1, 0.5], { ...config(decoding), k: 2, topP: 1 }, 42),
      { items } = run.frames[3].ranking!;
    expect(items.map((item) => item.name)).toEqual(['a', 'b', 'c']);
    expect(items[2]).toMatchObject({ before: 0, after: 0, kept: false });
  });

  it('抽样条：区间首尾相接、总长为 1，u 落在被选中的区间里', () => {
    for (const seed of [1, 7, 42, 999]) {
      const run = decoding.compute([2, 1, 0.5], { ...config(decoding), k: 2 }, seed),
        strip = run.frames[4].strip!;
      strip.segments.forEach((s, i) => expect(s.start).toBe(i ? strip.segments[i - 1].end : 0));
      expect(strip.segments[strip.segments.length - 1].end).toBeCloseTo(1, 12);
      const hit = strip.segments[strip.chosen];
      expect(strip.u >= hit.start && strip.u < hit.end).toBe(true);
      expect(hit.end - hit.start).toBeGreaterThan(0);
      expect(run.frames[4].metrics.find(([name]) => name === '本次 token')![1]).toBe(hit.name);
    }
  });
});

describe('偏好学习的专门图', () => {
  const base = config(preferenceTopic),
    book = [0.8, 0.3, 0.9, 0.5];

  it('偏好曲线：当前点只由奖励差决定，奖励整体平移时点不动', () => {
    const a = preferenceTopic.compute(book, base).frames[0].plot!,
      b = preferenceTopic.compute(book, { ...base, shift: 2.5 }).frames[0].plot!;
    expect(b.points[0]).toEqual(a.points[0]);
    expect(a.points[0].y).toBeCloseTo(sigmoid(0.8 - 0.3), 12);
    // 原书 29.3.2：r(A) = 5、r(B) = 3 → σ(2) ≈ 0.88
    expect(a.points[1].x).toBe(2);
    expect(a.points[1].y).toBeCloseTo(0.8808, 4);
  });

  it('PPO 图：A > 0 时 ρ ≥ 1 + ε 的部分是平的，A < 0 时 ρ ≤ 1 − ε 的部分是平的', () => {
    for (const advantage of [1.5, -1]) {
      const plot = preferenceTopic.compute(book, { ...base, advantage, clip: 0.2 }).frames[2].plot!,
        [objective, unclipped] = plot.curves;
      objective.points.forEach(([r, y]) => expect(y).toBeCloseTo(ppoTerm(r, advantage, 0.2), 12));
      unclipped.points.forEach(([r, y]) => expect(y).toBeCloseTo(r * advantage, 12));
      const flat = objective.points.filter(([r]) => (advantage > 0 ? r >= 1.2 : r <= 0.8));
      expect(flat.length).toBeGreaterThan(10);
      flat.forEach(([, y]) => expect(y).toBeCloseTo((advantage > 0 ? 1.2 : 0.8) * advantage, 12));
    }
  });

  it('GRPO 数轴：两行刻度对齐，每个回答的奖励点正好在它的优势点上方', () => {
    for (const input of [book, [0, 1, 2, 3], [0, 0, 0, 1], [2, 2, 2, 2]]) {
      const line = preferenceTopic.compute(input, base).frames[3].line!,
        [[low, high], [low2, high2]] = line.domains as [number, number][];
      if (input.every((v) => v === input[0])) {
        expect(line.rows[1].values).toEqual([0, 0, 0, 0]);
        continue;
      }
      line.rows[0].values.forEach((r, i) =>
        expect((line.rows[1].values[i] - low2) / (high2 - low2)).toBeCloseTo(
          (r - low) / (high - low),
          12,
        ),
      );
      for (const m of line.markers!) {
        const [l, h] = line.domains![m.row]!;
        expect(m.value >= l && m.value <= h, m.label).toBe(true);
      }
    }
  });

  it('平移数轴：前两行共用刻度，第三行的优势与平移量无关', () => {
    const a = preferenceTopic.compute(book, base).frames[5].line!,
      b = preferenceTopic.compute(book, { ...base, shift: -2 }).frames[5].line!;
    expect(b.domains![0]).toEqual(b.domains![1]);
    b.rows[1].values.forEach((v, i) => expect(v).toBeCloseTo(b.rows[0].values[i] - 2, 12));
    b.rows[2].values.forEach((v, i) => expect(v).toBeCloseTo(a.rows[2].values[i], 12));
  });
});

describe('SSM 的专门图', () => {
  const base = config(ssmTopic),
    x = [1, 0, 2, -1];

  it('状态链每一步满足 h_t = 保留 × h_{t−1} + 写入，并与扫描表一致', () => {
    for (const selective of [0, 1]) {
      const run = ssmTopic.compute(x, { ...base, selective, cursor: 2 }),
        chain = run.frames[3].chain!,
        row = chain.rows[0];
      let h = row.initial;
      row.steps.forEach((s) => {
        expect(s.state).toBeCloseTo(s.keep * h + s.drive, 12);
        h = s.state;
      });
      expect(chain.cursor).toBe(2);
      expect(row.steps.slice(0, 3).map((s) => s.state)).toEqual(run.frames[3].tensors[0].values[0]);
    }
  });

  it('碰撞图：两段不同的输入扫两步后状态相同（含 A = 0）', () => {
    for (const [A, delta] of [
      [-1, 0.5],
      [-2, 1.25],
      [0, 0.5],
    ]) {
      const chain = ssmTopic.compute(x, { ...base, A, delta }).frames[5].chain!,
        [a, b] = chain.rows.map((r) => r.steps[r.steps.length - 1].state);
      expect(a).toBeCloseTo(b, 12);
      expect(chain.rows[0].steps.map((s) => s.input)).not.toEqual(
        chain.rows[1].steps.map((s) => s.input),
      );
    }
  });

  it('参数曲线与离散化曲线：交点就是表中的 Δ_t、B_t、C_t 与保留系数', () => {
    const run = ssmTopic.compute(x, base),
      [dt, B, C] = run.frames[1].tensors.map((t) => t.values[0]);
    run.frames[1].plot!.points.forEach((p, k) => {
      const t = Math.floor(k / 3);
      expect(p.x).toBe(x[t]);
      expect(p.y).toBe([dt, B, C][k % 3][t]);
    });
    const keep = run.frames[2].tensors[0].values[0];
    run.frames[2]
      .plot!.points.filter((p) => p.series === 0)
      .forEach((p, t) => {
        expect(p.x).toBe(dt[t]);
        expect(p.y).toBe(keep[t]);
      });
    // A < 0：保留系数随 Δ 单调下降，每单位 B 的写入系数单调上升（第 32.3.4 节的修正）
    const [keepCurve, writeCurve] = run.frames[2].plot!.curves.map((c) =>
      c.points.map(([, y]) => y),
    );
    for (let i = 1; i < keepCurve.length; i++) {
      expect(keepCurve[i]).toBeLessThan(keepCurve[i - 1]);
      expect(writeCurve[i]).toBeGreaterThan(writeCurve[i - 1]);
    }
  });
});
