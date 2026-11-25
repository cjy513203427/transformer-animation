import { describe, it, expect } from 'vitest';
import { matmul, softmax } from '../attention';
import { normalize, normalization } from './normalization';
import { ffn, FFN_W1, FFN_W2 } from './ffn';
import { lora, BASE_WEIGHT } from './lora';
import { quantize, quantizeAWQ, quantizeGPTQ, QUANT_ACTIVATION } from './quantization';
import { filterTokens, sampleToken, searchTree } from './sampling';
import { groupAdvantages, ppoTerm, preference } from './preference';
import { memory, linearScan, linearReference, infiniSegments, phi } from './memory';
import { selectiveScan } from './ssm';
import { error, randomValues, uniforms } from './shared';
const near = (a: number[], b: number[], tol = 1e-10) =>
  expect(error(a, b)).toBeLessThanOrEqual(tol);
describe('归一化与概率的数学性质', () => {
  it('总体方差、中心化与 ε 引起的输出方差可独立核对', () => {
    const n = normalize([1, 2, 3, 4], 0.01);
    expect(n.mu).toBe(2.5);
    expect(n.variance).toBe(1.25);
    near(
      n.ln,
      [-1.5, -0.5, 0.5, 1.5].map((v) => v / Math.sqrt(1.26)),
    );
    expect(n.ln.reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(1.25 / 1.26, 12);
  });
  it('常量与零向量没有 NaN；RMS 不要求均值0', () => {
    near(normalize([2, 2, 2, 2], 1e-5).ln, [0, 0, 0, 0]);
    near(normalize([0, 0, 0, 0], 1e-5).rms, [0, 0, 0, 0]);
    expect(normalize([1, 2, 3, 4], 1e-5).rms.every((v) => v > 0)).toBe(true);
  });
  it('平移不变但固定 ε 下不能声称任意缩放不变', () => {
    const x = [0.001, 0.002, 0.003, 0.004];
    near(
      normalize(x, 1e-5).ln,
      normalize(
        x.map((v) => v + 3),
        1e-5,
      ).ln,
    );
    expect(
      error(
        normalize(x, 1).ln,
        normalize(
          x.map((v) => v * 2),
          1,
        ).ln,
      ),
    ).toBeGreaterThan(0.001);
  });
  it('Softmax 沿候选轴，屏蔽项0，全屏蔽约定0', () => {
    const p = { eps: 1e-5, gain: 1, bias: 0, temperature: 0.25, mask: 1, shift: 2 };
    const result = normalization([8, -8, 4, 1], p);
    const probability = result.frames[4].tensors[2].values[0];
    expect(probability[3]).toBe(0);
    expect(probability.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    near(
      normalization([1, 2, 3, 4], { ...p, mask: 2 }).frames[4].tensors[2].values[0],
      [0, 0, 0, 0],
    );
  });
});
describe('FFN、残差和 Dropout', () => {
  it('关闭非线性后可以合并矩阵，启用后存在反例', () => {
    const x = [1, 2, -1, 1, 2, -1, -2, -1],
      p = { nonlinear: 0, dropout: 0, training: 0 };
    const a = ffn(x, p, 42);
    near(
      a.frames[2].tensors[2].values.flat(),
      matmul(
        [
          [1, 2],
          [-1, 1],
          [2, -1],
          [-2, -1],
        ],
        matmul(FFN_W1, FFN_W2),
      ).flat(),
    );
    expect(error(a.signature, ffn(x, { ...p, nonlinear: 1 }, 42).signature)).toBeGreaterThan(0);
  });
  it('推理忽略 Dropout 种子与丢弃率；训练掩码只改变分支', () => {
    const x = [1, 2, -1, 1, 2, -1, -2, -1],
      p = { nonlinear: 1, training: 0, dropout: 0.5 };
    near(ffn(x, p, 1).signature, ffn(x, { ...p, dropout: 0.75 }, 900).signature);
    const run = ffn(x, { ...p, training: 1 }, 42),
      f = run.frames[2].tensors[2].values.flat(),
      mask = run.frames[4].tensors[0].values.flat();
    near(
      run.signature,
      x.map((v, i) => v + (mask[i] * f[i]) / 0.5),
    );
  });
  it('残差局部 Jacobian 与独立有限差分一致（避开 ReLU 零点）', () => {
    const x = [1, 2, -1, 1, 2, -1, -2, -1],
      p = { nonlinear: 1, training: 0, dropout: 0 },
      r = ffn(x, p, 42),
      j = r.frames[3].tensors[3].values;
    for (let d = 0; d < 2; d++) {
      const changed = x.slice();
      changed[d] += 1e-6;
      near(
        ffn(changed, p, 42)
          .signature.slice(0, 2)
          .map((v, k) => (v - r.signature[k]) / 1e-6),
        j[d],
        1e-8,
      );
    }
  });
});
describe('LoRA 与分组量化', () => {
  it('α=0 返回冻结底座，双支路与合并路径等价且不修改底座', () => {
    const before = JSON.stringify(BASE_WEIGHT);
    for (const rank of [1, 2]) {
      const r = lora([1, 2, -1, 1], { rank, alpha: 0 });
      near(
        r.signature,
        matmul(
          [
            [1, 2],
            [-1, 1],
          ],
          BASE_WEIGHT,
        ).flat(),
      );
      expect(lora([1, 2, -1, 1], { rank, alpha: 2 }).checks[0].error).toBeLessThan(1e-10);
    }
    expect(JSON.stringify(BASE_WEIGHT)).toBe(before);
  });
  it('无裁剪量化的误差不超过半格，全零组安全，负半格远离零', () => {
    for (const bits of [2, 4, 8]) {
      const q = quantize([-0.9, -0.4, 0.2, 0.8], bits, 4, 8);
      q.difference.forEach((v) => expect(Math.abs(v)).toBeLessThanOrEqual(q.scales[0] / 2 + 1e-12));
    }
    near(quantize([0, 0], 4, 2, 8).restored, [0, 0]);
    const q = quantize([-1, -0.5, 0.5, 1], 2, 4, 1);
    near(q.q, [-1, -1, 1, 1]);
  });
  it('裁剪误差可超过半格，分组与尺度元数据都来自实际数值', () => {
    const q = quantize([0.2, 0.4, 0.6, 8], 4, 4, 1);
    expect(Math.abs(q.difference[3])).toBe(7);
    const grouped = quantize([0.2, 0.4, 0.6, 8], 4, 2, 8);
    expect(grouped.scales.length).toBe(2);
    expect(grouped.bytes).toBe(2 + 8);
    expect(Math.abs(grouped.difference[0])).toBeLessThan(
      Math.abs(quantize([0.2, 0.4, 0.6, 8], 4, 4, 8).difference[0]),
    );
  });
  it('非对称量化精确保留 0，偏态正值格点比对称更密且计入零点字节', () => {
    const pos = [0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 4.0];
    const sym = quantize(pos, 4, 8, 8, false);
    const asym = quantize(pos, 4, 8, 8, true);
    expect(asym.scales[0]).toBeLessThan(sym.scales[0] * 0.55);
    expect(asym.mse).toBeLessThan(sym.mse);
    expect(asym.zeroPointBytes).toBe(1);
    expect(asym.bytes).toBe(sym.bytes + 1);
    const withZero = quantize([-1, 0, 1.5, 3], 4, 4, 8, true);
    expect(withZero.restored[1]).toBe(0);
  });
  it('AWQ 等价缩放保持点积恒等并降低高激活输出误差；GPTQ 逆 Hessian 精确且降低二次损失', () => {
    const w = [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6];
    const rtn = quantize(w, 4, 8, 8, false);
    const awq = quantizeAWQ(w, QUANT_ACTIVATION, 4, 8, 8, false);
    expect(awq.scaledOutput).toBeCloseTo(awq.directOutput, 12);
    const full = w.reduce((s, v, i) => s + v * QUANT_ACTIVATION[i], 0);
    const rtnOutErr = Math.abs(rtn.restored.reduce((s, v, i) => s + v * QUANT_ACTIVATION[i], 0) - full);
    const awqOutErr = Math.abs(awq.directOutput - full);
    expect(awqOutErr).toBeLessThan(rtnOutErr);

    const gptqGrouped = quantizeGPTQ(w, 4, 4, 8, false);
    expect(gptqGrouped.identityError).toBeLessThan(1e-10);
    expect(gptqGrouped.hessianLoss).toBeLessThan(gptqGrouped.rtnHessianLoss);
    const gptqBalanced = quantizeGPTQ([-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 1.2], 4, 8, 8, false);
    expect(gptqBalanced.hessianLoss).toBeLessThan(gptqBalanced.rtnHessianLoss);
  });
});
describe('解码集合与短序列搜索', () => {
  it('Top-P=1不会因浮点累计提前到1而丢失极小正概率', () => {
    const p = softmax([32, 0, -32]);
    expect(p[2]).toBeGreaterThan(0);
    expect(filterTokens(p, 3, 1).kept).toEqual([0, 1, 2]);
    expect(filterTokens(p, 3, 0.999).kept).toEqual([0]);
  });
  it('Top-P 保留首次越过阈值项，重新归一化', () => {
    const result = filterTokens([0.5, 0.3, 0.2], 3, 0.6);
    expect(result.kept).toEqual([0, 1]);
    near(result.probabilities, [0.625, 0.375, 0]);
    expect(filterTokens([0.5, 0.3, 0.2], 3, 1).kept).toEqual([0, 1, 2]);
  });
  it('同分用ID，Top-K后零项不回流，采样遵循区间边界', () => {
    expect(filterTokens([1 / 3, 1 / 3, 1 / 3], 2, 1).kept).toEqual([0, 1]);
    expect(filterTokens([1, 0, 0], 3, 1).kept).toEqual([0]);
    expect(sampleToken([0.2, 0.8, 0], 0)).toBe(0);
    expect(sampleToken([0.2, 0.8, 0], 0.2)).toBe(1);
    expect(sampleToken([0.2, 0.8, 0], 1 - 1e-12)).toBe(1);
    expect(uniforms(42, 3)).toEqual(uniforms(42, 3));
  });
  it('玩具树贪心 aa=0.2，beam2 ba=0.27，与穷举核对', () => {
    const s = searchTree([0.5, 0.3, 0.2]);
    expect(s.greedy.path).toBe('aa');
    expect(s.greedy.score).toBeCloseTo(0.2);
    expect(s.beam.path).toBe('ba');
    expect(s.beam.score).toBeCloseTo(0.27);
    expect(s.best).toEqual(s.beam);
  });
});
describe('偏好目标与组优势', () => {
  it('优势均值0、总体方差1、奖励平移不变，零方差设0', () => {
    const a = groupAdvantages([0, 1, 2, 3]).advantages;
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12);
    expect(a.reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(1, 12);
    near(a, groupAdvantages([3, 4, 5, 6]).advantages);
    near(groupAdvantages([2, 2, 2, 2]).advantages, [0, 0, 0, 0]);
  });
  it('PPO 的正负优势按 min 处理，不能无条件 clip 后乘', () => {
    expect(ppoTerm(1.5, 1, 0.2)).toBe(1.2);
    expect(ppoTerm(0.5, 1, 0.2)).toBe(0.5);
    expect(ppoTerm(0.5, -1, 0.2)).toBe(-0.8);
    expect(ppoTerm(1.5, -1, 0.2)).toBe(-1.5);
  });
  it('偏好回答概率提高会减小 DPO loss；零优势仍可能有参考约束', () => {
    const p = { shift: 0, policy: 0.5, beta: 0.1, ratio: 1.5, advantage: 1, clip: 0.2, kl: 0.02 };
    const a = preference([2, 2, 2, 2], p),
      b = preference([2, 2, 2, 2], { ...p, policy: 2 });
    expect(b.frames[1].metrics[2][1]).toBeLessThan(a.frames[1].metrics[2][1] as number);
    expect(a.frames[4].metrics[0][1]).toBeLessThan(0);
  });
});
describe('稀疏、线性与分段记忆', () => {
  it('窗口1只读取自身，窗口4等于同条件完整因果基线', () => {
    const x = [1, 0, 0, 1, 1, 1, -1, 0.5],
      p = { window: 1, gate: 0, delta: 0 },
      self = memory(x, p);
    near(self.signature, x);
    const full = memory(x, { ...p, window: 4 });
    near(full.signature, full.frames[0].tensors[2].values.flat());
  });
  it('核递推与显式前缀权重基线在多组输入等价', () => {
    for (let seed = 1; seed <= 12; seed++) {
      const values = randomValues(seed, 24),
        q = Array.from({ length: 4 }, (_, i) => values.slice(i * 2, i * 2 + 2)),
        k = Array.from({ length: 4 }, (_, i) => values.slice(8 + i * 2, 10 + i * 2)),
        v = Array.from({ length: 4 }, (_, i) => values.slice(16 + i * 2, 18 + i * 2));
      near(linearScan(q, k, v).outputs.flat(), linearReference(q, k, v).flat());
    }
  });
  it('Infini 当前段从旧记忆读取，没有未来输入泄漏；初始记忆0', () => {
    const q = [
        [1, 0],
        [0, 1],
        [1, 1],
        [-1, 0.5],
      ],
      a = infiniSegments(q, q, q, 0.5, false);
    near(a.memory[0], [0, 0]);
    near(a.memory[1], [0, 0]);
    const old = linearScan(q.slice(0, 2), q.slice(0, 2), q.slice(0, 2));
    const f = phi(q[2]),
      den = f[0] * old.z[0] + f[1] * old.z[1];
    near(
      a.memory[2],
      [0, 1].map((d) => (f[0] * old.S[0][d] + f[1] * old.S[1][d]) / den),
    );
    const changed = q.map((v) => v.slice());
    changed[3] = [8, 8];
    near(a.memory[2], infiniSegments(changed, changed, changed, 0.5, false).memory[2]);
  });
  it('Delta 第一段与Linear相同；第二段按同一个旧状态写残差', () => {
    const q = [
        [1, 0],
        [0, 1],
        [1, 1],
        [-1, 0.5],
      ],
      a = infiniSegments(q, q, q, 0.5, true),
      old = linearScan(q.slice(0, 2), q.slice(0, 2), q.slice(0, 2)),
      expected = old.S.map((r) => r.slice());
    for (let t = 2; t < 4; t++) {
      const f = phi(q[t]),
        den = f[0] * old.z[0] + f[1] * old.z[1],
        prediction = [0, 1].map((d) => (f[0] * old.S[0][d] + f[1] * old.S[1][d]) / den);
      for (let d = 0; d < 2; d++)
        for (let c = 0; c < 2; c++) expected[d][c] += f[d] * (q[t][c] - prediction[c]);
    }
    near(a.S.flat(), expected.flat());
  });
});
describe('选择性 SSM', () => {
  it('固定输入扫描与连续系统的闭式解一致，A=0 极限可计算', () => {
    const A = -0.7,
      delta = 0.5,
      h0 = 1.2,
      x = 2,
      s = selectiveScan([x, x, x, x], A, delta, false, h0),
      closed = Math.exp(4 * delta * A) * h0 + (Math.expm1(4 * delta * A) / A) * x;
    expect(s.states[3]).toBeCloseTo(closed, 12);
    near(selectiveScan([1, 2], 0, 0.5, false, 1).states, [1.5, 2.5]);
  });
  it('A<0 更大步长降低保留系数，修改后续输入不改前缀输出', () => {
    const x = [1, 0, 2, -1],
      a = selectiveScan(x, -1, 0.5, true, 1),
      b = selectiveScan(x, -1, 1, true, 1);
    a.retention.forEach((v, i) => expect(b.retention[i]).toBeLessThan(v));
    near(a.outputs.slice(0, 2), selectiveScan([1, 0, 8, 8], -1, 0.5, true, 1).outputs.slice(0, 2));
  });
});
