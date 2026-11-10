import { attention, type Matrix } from '../attention';
import {
  check,
  dot,
  flat,
  frame,
  pairRows,
  softmax,
  tensor,
  type Computation,
  type Parameters,
} from './shared';
import { T } from '../../i18n';
export const phi = (x: number[]) => x.map((v) => (v >= 0 ? v + 1 : Math.exp(v)));
const zero = () => [
  [0, 0],
  [0, 0],
];
function retrieve(q: number[], S: Matrix, z: number[]) {
  const a = phi(q),
    den = dot(a, z);
  return [0, 1].map((d) => (den === 0 ? 0 : (a[0] * S[0][d] + a[1] * S[1][d]) / den));
}
export function linearScan(q: Matrix, k: Matrix, v: Matrix) {
  const S = zero(),
    z = [0, 0],
    outputs: Matrix = [];
  for (let t = 0; t < q.length; t++) {
    const f = phi(k[t]);
    for (let d = 0; d < 2; d++) {
      z[d] += f[d];
      for (let c = 0; c < 2; c++) S[d][c] += f[d] * v[t][c];
    }
    outputs.push(retrieve(q[t], S, z));
  }
  return { S, z, outputs };
}
export function linearReference(q: Matrix, k: Matrix, v: Matrix) {
  return q.map((row, t) => {
    const weights = k.map((key, j) => (j <= t ? dot(phi(row), phi(key)) : 0)),
      den = weights.reduce((a, b) => a + b, 0);
    return [0, 1].map((d) => weights.reduce((s, w, j) => s + w * v[j][d], 0) / den);
  });
}
export function infiniSegments(q: Matrix, k: Matrix, v: Matrix, gate: number, delta: boolean) {
  let S = zero(),
    z = [0, 0];
  const memory: Matrix = [],
    local: Matrix = [],
    outputs: Matrix = [];
  for (let start = 0; start < q.length; start += 2) {
    const Q = q.slice(start, start + 2),
      K = k.slice(start, start + 2),
      V = v.slice(start, start + 2),
      read = Q.map((row) => retrieve(row, S, z)),
      prediction = K.map((row) => retrieve(row, S, z)),
      within = attention(Q, K, V, true).output,
      newS = S.map((row) => row.slice()),
      newZ = z.slice();
    for (let j = 0; j < K.length; j++) {
      const f = phi(K[j]);
      for (let d = 0; d < 2; d++) {
        newZ[d] += f[d];
        for (let c = 0; c < 2; c++) newS[d][c] += f[d] * (V[j][c] - (delta ? prediction[j][c] : 0));
      }
    }
    memory.push(...read);
    local.push(...within);
    outputs.push(...read.map((row, j) => row.map((v, d) => gate * v + (1 - gate) * within[j][d])));
    S = newS;
    z = newZ;
  }
  return { S, z, memory, local, outputs };
}
export function memory(x: number[], p: Parameters): Computation {
  const Q = pairRows(x),
    K = Q.map(([a, b]) => [b, a]),
    V = Q.map((row) => row.slice()),
    dense = attention(Q, K, V, true),
    masked = dense.scores.map((row, t) =>
      row.map((s, j) => (j > t || j < t - p.window + 1 ? -Infinity : s)),
    ),
    weights = masked.map(softmax),
    sparse = weights.map((row) => [0, 1].map((d) => row.reduce((s, w, j) => s + w * V[j][d], 0))),
    scan = linearScan(Q, K, V),
    reference = linearReference(Q, K, V),
    gate = 1 / (1 + Math.exp(-p.gate)),
    infini = infiniSegments(Q, K, V, gate, !!p.delta);
  const collisionA = linearScan(
      [
        [1, 1],
        [1, 1],
      ],
      [
        [1, 1],
        [1, 1],
      ],
      [
        [1, 0],
        [0, 1],
      ],
    ),
    collisionB = linearScan(
      [
        [1, 1],
        [1, 1],
      ],
      [
        [1, 1],
        [1, 1],
      ],
      [
        [0.5, 0.5],
        [0.5, 0.5],
      ],
    );
  return {
    frames: [
      frame(T('完整因果 Attention 访问所有可见历史'), [
        tensor(T('Q（由输入给定）'), Q),
        tensor(T('P（行 Q、列 K）'), dense.probabilities),
        tensor(T('Softmax 输出'), dense.output),
      ]),
      frame(
        T('因果滑动窗口只保留最近 w 个 Key'),
        [tensor(T('窗口概率 P'), weights), tensor(T('窗口输出'), sparse)],
        [[T('窗口 w（含自身）'), p.window]],
        T('本页用稠密矩阵加 Mask 验证数学；只有相应稀疏实现才能减少实际计算。'),
      ),
      frame(
        T('换成正核特征，定义另一种 Attention'),
        [
          tensor('φ(Q) = ELU(Q)+1', Q.map(phi)),
          tensor(T('显式核 Attention 输出'), reference),
          tensor(T('Softmax 输出（对照）'), dense.output),
        ],
        [],
        T('核归一化不是标准 Softmax 的恒等重写，输出一般不同。'),
      ),
      frame(
        T('核权重可通过固定大小 S、z 状态递推'),
        [
          tensor(T('最后 S [2, 2]'), scan.S),
          tensor(T('最后 z [1, 2]'), scan.z),
          tensor(T('线性递推输出'), scan.outputs),
        ],
        [[T('状态标量数'), 6]],
        T('包含当前 Key 后再读取。固定特征维度时扫描对 T 线性；本页额外保存轨迹以便核对。'),
      ),
      frame(
        T('Infini：先读上一段记忆，再合并本段局部结果'),
        [
          tensor(T('上一段记忆读出'), infini.memory),
          tensor(T('本段因果 Softmax'), infini.local),
          tensor(T('门控合并输出'), infini.outputs),
        ],
        [
          [T('记忆门 sigmoid(β)'), gate],
          [T('更新规则'), p.delta ? 'Delta' : 'Linear'],
        ],
        T(
          '每段 2 token；整段更新都读取同一个旧记忆。初始空记忆返回零。Delta 更新预测 V 后写残差，z 仍累加。',
        ),
      ),
      frame(
        T('两段不同历史，可以压缩到同一个状态'),
        [
          tensor(T('历史 A 的 V'), [
            [1, 0],
            [0, 1],
          ]),
          tensor(T('历史 B 的 V'), [
            [0.5, 0.5],
            [0.5, 0.5],
          ]),
          tensor(T('二者相同的最终 S'), collisionA.S),
        ],
        [[T('共同 z'), collisionA.z.join(', ')]],
        T(
          '这说明此压缩映射有碰撞。可计算长流不代表无损保存所有历史，也不保证任意长上下文的可靠性。',
        ),
      ),
    ].map((item, index) =>
      index < 2 ? { ...item, connections: index === 0 ? dense.probabilities : weights } : item,
    ),
    checks: [
      check(T('线性递推与显式核基线等价'), flat(scan.outputs), flat(reference)),
      check(
        T('不同历史的压缩状态碰撞'),
        flat(collisionA.S).concat(collisionA.z),
        flat(collisionB.S).concat(collisionB.z),
      ),
    ],
    signature: flat(sparse),
  };
}
