import {
  check,
  dot,
  frame,
  mean,
  tensor,
  type Computation,
  type NumberLine,
  type Parameters,
} from './shared';
export function quantize(x: number[], bits: number, groupSize: number, cap: number) {
  const qmax = 2 ** (bits - 1) - 1,
    q: number[] = [],
    restored: number[] = [],
    scales: number[] = [],
    clipped: number[] = [];
  for (let start = 0; start < x.length; start += groupSize) {
    const group = x.slice(start, start + groupSize),
      extent = Math.min(cap, Math.max(...group.map(Math.abs))),
      scale = extent === 0 ? 1 : extent / qmax;
    scales.push(scale);
    for (const value of group) {
      const limited = Math.max(-extent, Math.min(extent, value));
      const signed = Math.sign(limited) * Math.floor(Math.abs(limited / scale) + 0.5);
      q.push(signed);
      restored.push(signed * scale);
      clipped.push(limited);
    }
  }
  const difference = x.map((v, i) => restored[i] - v),
    mse = mean(difference.map((v) => v * v));
  return {
    qmax,
    q,
    restored,
    scales,
    clipped,
    difference,
    mse,
    bytes: Math.ceil((x.length * bits) / 8) + scales.length * 4,
  };
}
export function quantization(x: number[], p: Parameters): Computation {
  const q = quantize(x, p.bits, p.group, p.cap),
    global = quantize(x, p.bits, x.length, p.cap);
  const activation = [1, -1, 0.5, 0, 1, -0.5, 2, -1];
  const fullOutput = dot(activation, x),
    quantOutput = dot(activation, q.restored);
  // One group has one grid; several groups have different grids, so none is drawn for them.
  const single = q.scales.length === 1,
    extent = Math.min(p.cap, Math.max(...x.map(Math.abs))),
    ticks = !single
      ? undefined
      : extent === 0
        ? [0]
        : Array.from({ length: 2 * q.qmax + 1 }, (_, k) => (k - q.qmax) * q.scales[0]),
    bounds = (row: number) =>
      single && extent > 0
        ? [
            { row, value: -extent, label: '−边界' },
            { row, value: extent, label: '+边界' },
          ]
        : [],
    names = x.map((_, i) => String(i)),
    gridNote = single
      ? '短竖线是能表示的全部格点'
      : `${q.scales.length} 组各有自己的格点间距，图中不画格点`;
  const lines: (NumberLine | undefined)[] = [
    {
      rows: [{ label: '原 x', values: x }],
      names,
      shared: true,
      ticks,
      markers: bounds(0),
      caption: `${gridNote}；每个数只能落到其中一个格点上`,
    },
    {
      rows: [
        { label: '裁剪后 x', values: q.clipped },
        { label: '还原 x̂', values: q.restored },
      ],
      names,
      shared: true,
      ticks,
      caption: `${gridNote}；连线从原位置指向最近的格点`,
    },
    {
      rows: [
        { label: '原 x', values: x },
        { label: '还原 x̂', values: q.restored },
      ],
      names,
      shared: true,
      ticks,
      markers: bounds(0),
      caption: `${gridNote}；虚线是校准边界，超出的数被压到边界上`,
    },
    {
      rows: [
        { label: '原 x', values: x },
        { label: '全向量一组', values: global.restored },
        { label: '当前分组', values: q.restored },
      ],
      names,
      shared: true,
      caption: '同一把尺：比较一组共用尺度与分组各用尺度，小数值离原位置有多远',
    },
    {
      rows: [
        { label: '原 x', values: x },
        { label: '还原 x̂', values: q.restored },
      ],
      names,
      shared: true,
      ticks,
      caption: `${gridNote}；连线越斜，这一项的误差越大`,
    },
  ];
  return {
    frames: [
      frame(
        '对称整数格点：共享尺度与零点 0',
        [tensor('原 x', x), tensor('每组 scale', q.scales)],
        [
          ['整数范围 ±qmax', q.qmax],
          ['位数', p.bits],
        ],
        '采用 [−qmax, qmax]，有一个补码编码未使用；不是所有量化格式的通用规则。',
      ),
      frame(
        '先裁剪，再按尺度舍入到整数',
        [
          tensor('裁剪后的 x', q.clipped),
          tensor('整数 q', q.q),
          tensor('还原 x̂ = q × scale', q.restored),
        ],
        [['第 0 组 scale', q.scales[0]]],
        '本例半整数采用远离零的舍入规则；不同实现的 ties-to-even 可能得到不同结果。',
      ),
      frame(
        '校准上限变小，格点更密，也更容易截断异常值',
        [tensor('原 x', x), tensor('裁剪后的 x', q.clipped), tensor('还原 x̂', q.restored)],
        [['校准上限 cap', p.cap]],
        'cap 限制每组最大绝对值；这是可控的教学校准规则。',
      ),
      frame(
        '分组让不同区间拥有不同尺度',
        [
          tensor('全向量共用尺度的还原', global.restored),
          tensor('当前分组还原', q.restored),
          tensor('当前分组 scale', q.scales),
        ],
        [
          ['组大小', p.group],
          ['组数', q.scales.length],
        ],
        '组更小需要更多尺度元数据；不是无代价提高精度。',
      ),
      frame(
        '一个异常大值，会拉宽所在组的格点',
        [tensor('原 x', x), tensor('还原 x̂', q.restored), tensor('逐项误差 x̂ − x', q.difference)],
        [['均方误差 MSE', q.mse]],
      ),
      frame(
        '存储账本与误差同时检查',
        [
          tensor('固定激活 u', activation),
          tensor('线性输出 [u·x, u·x̂]', [fullOutput, quantOutput]),
          tensor('逐项误差', q.difference),
        ],
        [
          ['打包整数载荷 B', Math.ceil((x.length * p.bits) / 8)],
          ['float32 尺度 B', q.scales.length * 4],
          ['合计 B', q.bytes],
          ['float32 基线 B', x.length * 4],
          ['MSE', q.mse],
          ['线性输出误差', quantOutput - fullOutput],
        ],
        '将当前八个数看作线性层一行权重，用固定激活 u 核对输出误差。账本假设整数紧密打包、每组一个 4B scale，不含对齐和格式头；JS 未实际位打包。GGUF 是格式，非单一量化算法。',
      ),
    ].map((item, index) => (lines[index] ? { ...item, line: lines[index] } : item)),
    checks: [
      check(
        '整数在有效编码范围内',
        q.q.map((v) => Math.max(0, Math.abs(v) - q.qmax)),
        q.q.map(() => 0),
      ),
      check('零输入可还原', [quantize([0, 0], p.bits, 2, p.cap).restored[0]], [0]),
    ],
    signature: [q.mse],
  };
}
