# 数值约定

## 单头自注意力

```text
X: [T, D]
Q = X W_Q: [T, d_k]
K = X W_K: [T, d_k]
V = X W_V: [T, d_v]
S = Q K^T / sqrt(d_k): [T, T]
P = softmax(S + M, axis=key): [T, T]
O = P V: [T, d_v]
```

本例 T = 3、D = d_k = d_v = 2。所有位置采用行向量。注意力矩阵行是 Query，列是 Key；参数矩阵行是输入特征，列是输出特征。省略 batch、偏置、Dropout、位置编码、残差和输出投影。W 是固定教学参数，不来自真实模型。

页面整体架构按原书第 3 章的 Post-Norm 顺序绘制：MHA → 残差相加 → LN → FFN → 残差相加 → LN。它是模块定位图，没有执行这些非 Attention 模块。第 15 章 GPT-2 的 Pre-Norm 方案单独说明。单头实验的数值 O 只是一个头的输出（书中 A 的一片）；多头的切分、拼接与 Wo 见下文“多头注意力”。

默认因果 Mask 开启，观察中间位置 t1。导读中的逐 Key 分数、Softmax 指数、归一化分母、各 V 的贡献向量均来自 `src/math/attention.ts`，与完整矩阵视图消费相同结果；不使用预先编造的语义分数。

因果模式包含当前位置，j > i 时 M[i,j] = −∞，其余为 0。Softmax 逐行减去最大值以避免溢出。全屏蔽行的数学 Softmax 不定义；本实现特意约定为全零，避免 NaN。内置因果场景总有可见的自身位置。

点积受方向和模长共同影响。缩放的方差解释采用各 q/k 分量独立、零均值、单位方差假设，因此点积方差为 d_k；它不意味着任意一对真实激活都恰好满足该方差。

## 对照实验

K 与 V 的滑杆在投影后干预激活的 [1,0] 元素，保持 W 不变。这用于隔离变量；改变真实隐藏状态 X 通常会同时改变 Q/K/V。

固定 Q/K、Mask、缩放时，ΔV 不影响 P，输出第 0 维变化为 `P[i,1] × ΔV`。第 1 维不变。当 t1 不可见，当前输出也不变。对照输出使用相同 X、K、Mask 和缩放，只去掉 V 干预。

固定其他输入只改 K[1,0] 会影响 S 的第 1 列；Softmax 的公共分母可能让该 Query 的整行 P 都改变。当相应 Query 分量为 0 或该位置被屏蔽时，可能没有变化。

## 精度与验证

运算使用 JavaScript float64，不把界面舍入值带回计算。界面通常显示 3 位小数；以显示值写出的输出展开式采用近似号。数值性质测试采用绝对误差 1e-12，独立标量参考只用于输入范围内的小矩阵比较。

覆盖：矩阵乘法轴对应、Softmax 平移不变性/极端值/屏蔽边界、20 个确定种子下的 Attention 参考对照、V 干预不变性、K 分数列影响、均匀分布和因果首行。

固定种子通过 32 位线性同余生成器产生 6 个半整数，范围 [−2, 2]。它只用于可复现教学输入，不用于密码学或真实模型权重初始化。

依据：[原论文 §3.2.1、§3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2)，核对于 2026-11-05。

## 多头注意力（原书第 11–12 章）

```text
X: [T, d_model]                      本例 T = 3，d_model = 4
Q = X W_Q, K = X W_K, V = X W_V      W: [d_model, d_model]
d_key = d_model / num_heads          num_heads ∈ {1, 2, 4}
Q → view [T, num_heads, d_key] → transpose [num_heads, T, d_key]（K、V 相同）
A_h = softmax(Q_h K_hᵀ / √d_key + M) V_h           [T, d_key]
A = Concat(A_1, …, A_h) → [T, d_model]
Output = A W_O = Σ_h A_h W_O^(h)                   W_O^(h) 为 W_O 中与第 h 个头相乘的行块
```

切分与 PyTorch `view(T, num_heads, d_key).transpose` 相同：第 h 个头取连续的列 `[h·d_key, (h+1)·d_key)`，token 行的顺序不变。每个头按自己的 √d_key 缩放。书中“方式一”（每头独立的 Wq_h）与“方式二”（一次大投影再切分）在 Wq_h 取 W_Q 对应列块时完全相同，页面实时核对这一点。教学权重的每个头列块都读取 X 的全部输入维度，避免把“切分头”误解为切分输入特征。

页面在每一步并列本例形状与书中 batch = 4、ctx_length = 16、d_model = 512、num_heads = 4 的形状：`[4,16,512] → [4,16,4,128] → [4,4,16,128]`，各头分数 `[4,4,16,16]`，合并回 `[4,16,512]`，再乘 `[512,512]` 的 Wo。参数量按 4 × d_model² 计（忽略偏置）。打分乘加次数为 num_heads × T² × d_key = T² × d_model，与头数无关。

对照实验在投影后修改激活 K[t0, d3]（权重不变）：只有拥有 d3 的那个头的 P 与 A 列会变，其他头完全不变；经 Wo 混合后 Output 的各列都可能变化。t0 在因果 Mask 下只能看自己，它那一行的权重恒为 [1, 0, 0]。

与原书对照：

- 第 11.8 节的“语法头”“指代头”等是训练后观察到的模式，头的分工不是预先指定的；本例两个头的差异只来自不同的权重列。
- 第 12 章说 (Q×K)×V 在“更新 Token Embedding”。本项目区分前向与训练：前向只产生新的表示；嵌入表与 Wq、Wk、Wv、Wo 只在训练中由反向传播和优化器更新，推理时保持不变。
- 早先版本把单头输出称为“H_head”，原书并无此记号，已改为书中第 11–12 章的 A。

验证：view/transpose 往返与 token 对应、1/2/4 个头与独立逐头标量参考实现比较（含因果 Mask）、单头退化情形、Σ_h A_h W_O^(h) 分解、K 干预的局部性、乘加次数与参数量。容差 1e-12。

## 位置编码（原书第 5、14、25 章）

位置从 0 开始编号；所有向量为行向量。

**正弦位置编码**：`PE(pos, 2i) = sin(pos·ω_i)`，`PE(pos, 2i+1) = cos(pos·ω_i)`，`ω_i = 10000^(−2i/d_model)`。每个 (sin, cos) 维度对是单位圆上的点，位置每加 1 转 ω_i 弧度。书中第 5 章“LLM张老师”表格是 d_model = 64 时的前 4 维，并截断（不是四舍五入）到两位小数；测试逐格核对这一点。页面提供 d_model = 4、64、512 三种设置。

**位置盲**：令 W_Q = W_K = W_V = I（d_model = 4），只看位置的作用。不加 Mask 时，“我爱你”与“你爱我”中“我”的输出相同；加入正弦位置编码后不同。加因果 Mask 时，“我爱你”与“爱我你”中最后的“你”看到同一组向量，输出相同：Mask 限制可见范围，但不提供可见范围内的先后。

**加法**：`X = E[token] + PE[pos]`，发生在投影之前，Q、K、V 都带位置。“L”的嵌入取书中列出的前 4 维 [0.62, −0.51, 0.09, 0.85]。点积的四项分解（第 25.2.2 节）在 W_Q = W_K = I 时为 `e·e + e·p + p·e + p·p`。

**RoPE**：d = 8、base = 10000，θ_i = 10000^(−2i/8)，恰为 1、0.1、0.01、0.001。`R_m` 把第 i 对 (x, y) 逆时针旋转 m·θ_i：`[x cos − y sin, x sin + y cos]`。只作用于 Q、K，不作用于 V，不加到 X 上。内容 q、k 固定时 `(R_m q)·(R_n k) = q·(R_{n−m} k)`，同时平移 m、n 分数不变；该性质需要固定内容与同一套 θ。单个分数随距离上下起伏，并不单调下降。

**ALiBi**：`P = softmax(QKᵀ/√d + m·B + M)`，`B[i][j] = −(i − j)`（j ≤ i），未来位置由因果 Mask 屏蔽。8 个头的斜率为 2^(−8h/8) = 2^(−h)，h = 1…8。内容分数相同时，权重随距离按 e^(−m) 等比衰减。偏置公式对任意长度都能计算，但这不保证模型能可靠利用超出训练长度的上下文。示例内容分数为人为设定。

**动画**：指针与向量只在相邻整数位置之间做 0.45 秒的转动过渡，跳跃时直接到位；过渡帧不是算法状态，编码只在整数位置取值。波形图中曲线是函数本身，圆点才是实际使用的数值。减少动态效果时不做过渡。

与原书对照：

- 第 5.3.2 节说 Col 6 频率最低、Col 9 最高。按公式同一对 sin/cos 频率相同，维度序号越大频率越低；书中波形图本身也是 Col 6、7 比 Col 8、9 转得快。
- 第 25.3.6 节的 θ = 0.1、0.2、0.4 是示意数字；按公式 θ 随维度对序号变小。
- 第 25.3.8 节说相邻 token 能获得更高注意力。RoFormer 证明的是一个上界随距离衰减；单个分数可能随距离升高，见“相对距离”一步。

验证：书中表格逐格一致、单位圆与频率单调性、64 个位置互不相同、PE(m)·PE(n) 只依赖 m − n、四项分解、位置盲与因果对照、RoPE 范数保持、平移不变与 R_mᵀR_n = R_{n−m}、分数非单调、ALiBi 斜率、偏置矩阵、等比衰减与 200 个 token 的长序列。容差 1e-12（三角函数平移比较 1e-9）。

依据：原书 GitHub 源文件（commit 4a77788，2026-11-06 读取）；公式按 [Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)、[RoFormer](https://arxiv.org/abs/2104.09864)、[ALiBi](https://arxiv.org/abs/2108.12409) 的记法编写。此次环境无法访问 arXiv，论文原文未在线复核。

## 训练、生成与缓存（原书第 2、16、22 章）

玩具解码器，行向量，位置从 0 开始：

```text
词表 a–f（6 个符号，无语义）    d_model = d_k = d_v = 2，L = 2 层，单头
X[t] = E[token_t] + PE[t]        PE 为正弦位置编码（d = 2：[sin t, cos t]）
每层 H' = H + softmax(causal(Q Kᵀ / √2)) V，Q = H W_Q，K = H W_K，V = H W_V
logits = H_L Eᵀ（输出头复用嵌入表），贪心解码取最大概率，平局取编号最小的 token
```

省略 W_O、FFN、LayerNorm、偏置与 Dropout。权重是固定教学数字，不是训练好的模型。

**训练**：样本 a b c d e，输入 x = a b c d，目标 y = b c d e（右移一位）。一次因果前向得到 4 行输出分布，每行 loss = −ln p(y_t)。只做前向与 loss，不做反向传播和参数更新。反事实只改输入 x_j、目标不动：第 0…j−1 行的输出分布逐位相同（被屏蔽的位置权重为 0，0 × V_j 精确为 0），所以测试用严格相等而不是容差。

**生成**：提示 a b c。Prefill 对提示做一次因果前向，把每层每个位置的 K、V 写入缓存，用最后一个位置的分布选第一个新 token。第 s 次 Decode 喂入上一步选出的 token（位置 2 + s），每层只为它算 q、k、v，把 k、v 追加进缓存，用新 q 读缓存里全部 K、V。不存 Q：旧位置的输出在之前的步骤里已经算完。

**对照**：每一步同时做一次“整段重算”（对全部 token 的完整因果前向，只取最后一行）。两条路线的 logits 最大差、选出的 token 都逐步比较，容差 1e-12；测试穷举全部 216 个 3-token 提示、各 5 步。工作量计数器在实现里实际执行处累加，并与闭式公式核对：

```text
完整前向（训练、Prefill、整段重算）  K/V 投影 L·n 次，打分 L·n(n+1)/2 次（只数因果可见的配对）
缓存 Decode（序列长 n）            K/V 投影 L 次，  打分 L·n 次
```

只统计 Attention 的 K/V 投影与 Q·K 打分，不含 FFN、输出头和内存读写，不代表实际耗时。含 Mask 的朴素实现可能算满 n×n 再屏蔽，计数会不同。

**失效**：缓存条目按“同一层、同一位置的 K 与 V 是否在容差 1e-12 内相同”判断。改动第 j 个 token、其余不变时：第 1 层只有位置 j 变化（K、V 只取决于本位置的 token 与位置编码）；第 2 层位置 j 之前不变，从 j 起一般都会变（输入混入了 x_j）。测试对多组提示验证前两条，第 2 层位置 j 本身必变。滑动窗口丢掉最早的 token 并重新从 0 编号时，加法位置编码下第 1 层每个旧条目都变化；这一结论针对本例的绝对位置编码，不代表 RoPE 等相对位置方案下的行为。

与原书对照：第 16、22 章的相关说法与本实验一致（训练并行来自已知序列与因果 Mask；第 22 章明确不承诺固定加速倍数，并指出多轮对话复用缓存需要完全相同的 token 前缀）。第 22 章用 Llama 2 7B 手算缓存字节数（每 token 512 KiB）的内容留给“KV 架构对照”实验。

依据：原书第 2、16、22 章（2026-11-07 在线读取）；[Attention Is All You Need §3.2.3](https://arxiv.org/abs/1706.03762)；[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 对 Prefill 与 Decode 两个阶段的划分。

## KV 架构对照（原书第 22、23、30 章，DeepSeek-V2 论文）

**缓存字节**（只统计 K、V，不含权重、激活与分配器开销）：

```text
bytes = B × L × T × H_KV × (d_k + d_v) × b
MLA：  bytes = B × L × T × (d_c + d_h^R) × b      latent 与共享的 RoPE Key 都要缓存
```

已逐字节核对：原书第 23 章（32 层、d_h = 128、1024 token、2 字节）MHA 536,870,912 B = 512 MiB、GQA-8 128 MiB、MQA 16 MiB；第 22 章 Llama 2 7B 每 token 524,288 B = 512 KiB，4096 token 2 GiB。只有 W_K、W_V 随 KV 头数变化：H_Q = 32 时注意力投影参数 MHA 4D²、GQA-8 2.5D²、MQA 2.0625D²。页面用二进制单位（1 KiB = 1024 B）。

**MHA / GQA / MQA**（玩具：T = 4、D = 4、H = 4 个 Query 头、d = 2，因果 Mask）：Query 头 h 读第 g(h) = ⌊h / (H/G)⌋ 组 K/V，G ∈ {4, 2, 1}。共享的 K/V 投影按 GQA 论文的转换方式由组内各头取平均得到；论文之后还会以约 5% 的原预训练算力继续训练，本页没有，所以 GQA/MQA 的输出不代表转换后的效果。按组查表与“把每组 K/V 显式复制给每个 Query 头”的参考计算（复用 `attention.ts` 的独立实现）比较，容差 1e-12；复制只用于核对，不是推荐的存储方式。打分次数 H × T(T+1)/2 = 40 与 G 无关，缓存每 token 每层 2Gd 个数。

**MLA**（玩具：d_c = 3、d_h^R = 2，RoPE θ = 1）。论文用列向量 W h；本页转写为行向量：

```text
c_t = h_t W_DKV                     缓存
k^C_{t,i} = c_t W_UK^(i)，v^C_{t,i} = c_t W_UV^(i)
k^R_t = RoPE_t(h_t W_KR)            缓存，所有头共享
q^C_{t,i} = h_t W_QC^(i)，q^R_{t,i} = RoPE_t(h_t W_QR^(i))
score = (q^C·k^C + q^R·k^R) / √(d_h + d_h^R)，o_{t,i} = Σ_j p v^C_{j,i}，u_t = [o_1; …; o_H] W_O
```

省略了论文中的 Query 压缩（W^DQ，用于减少训练时的激活内存，不影响 KV 缓存），RoPE Query 因此直接由 h 投影（论文中由 c^Q 投影）。

吸收路径：W̃_Q^(i) = W_QC^(i) W_UK^(i)ᵀ（[D, d_c]），W̃_O^(i) = W_UV^(i) W_O^(i)（[d_c, D]）；打分只读缓存的 c_j 与 k^R_j，加权和 Σ p c_j 停在 latent 空间。与展开路径的分数、权重、输出比较，容差 1e-12。反例：若 RoPE 直接加在 q^C 与还原出的 k^C 上，q 与 c_j 之间的矩阵为 R_t R_jᵀ W_UKᵀ，随 t − j 变化；同一个吸收矩阵 W_UKᵀ 只在 j = t 时正确，测试逐头、逐位置验证这一点。

**真实规模对照**：DeepSeek-V2 取 60 层、128 头、d_h = 128、d_c = 512、d_h^R = 64（论文 §3.1.2 与表 1 说明 d_c = 4d_h、d_h^R = d_h/2）。MLA 每 token 每层 576 个数 = (9/2) d_h，相当于 2.25 组的 GQA；只算 latent 会少算 64 个（11.1%）。同头数同头维的 MHA、GQA-8、MQA 是套公式得到的假想配置，只用来对照缓存；页面不比较效果，也不给排行。原书第 30.4.3 节“KV Cache 减少 75%”是假设 latent 维度为 1/4 时的示意计算，不是 DeepSeek 的实测。

依据：原书第 22、23、30 章；[MQA（Shazeer, 2019）](https://arxiv.org/abs/1911.02150)；[GQA（Ainslie et al., 2023）](https://arxiv.org/abs/2305.13245) §2.1–2.2；[DeepSeek-V2](https://arxiv.org/html/2405.04434v5) §2.1、表 1、§3.1.2。2026-11-07 在线读取。

## MoE 与专家并行（原书第 30 章）

单个、未训练的教学 MoE 层。行向量，T = 4、D = 2、E = 4 个专家、F = 3；忽略 batch 和偏置。输入是无语义的二维向量；专家编号 E0–E3 不表示预设的语义分工。

```text
X: [T, D]                         W_R: [D, E]
Z = X W_R: [T, E]                 p = softmax(Z, axis=expert)
A_t = TopK(p_t, K)                K ∈ {1, 2}，并列时取编号小的专家
g[t,e] = p[t,e] / Σ_{j∈A_t} p[t,j]（e∈A_t）；未选中为 0
f_e(x_t) = ReLU(x_t W_1,e) W_2,e   W_1,e: [D, F]，W_2,e: [F, D]
y_t = Σ_{e∈A_t} g[t,e] f_e(x_t)    Y: [T, D]
```

**门控变体**：采用 Softmax → Top-K → 选中概率重新归一化；Top-1 的唯一门值为 1。对有限分数，这等价于把未选中的 logits 设为 −∞ 后做 Softmax；没有 Shazeer 2017 的噪声。Mixtral 的路由实现也有选中后归一化，但它的专家网络与本例不同。Switch 的 Top-1 保留原 Softmax 门值，因此不能用本例的门值冒充 Switch 的完整复现。Softmax 计算减去每行最大值，内部保留 float64，显示时才舍入。

**隔离变量**：输入对照仅编辑所选 token 的第 0 维，其他输入和权重不变。E0 输出倍率属于专家计算后的干预，默认 1；改变它不会改变 Router 的 logits、概率、选中集合或门值，只改变经过 E0 的分支输出。该倍率不表示训练，也不增加模型参数计数。分散/集中预设用来观察负载，固定随机种子可重建相同教学权重。

**参数与计算账本**：一个专家有 DF + FD = 12 个参数，路由器有 DE = 8 个。总参数为 DE + E·2DF = 56；每 token 激活参数为 DE + K·2DF，K = 1 时 20，K = 2 时 32。即使某专家当前没有任务，仍需保存它的权重；本例未实现权重卸载。只计矩阵乘加时，Router 为 TDE = 32 MAC，专家为 TK·2DF，分别为 48 或 96 MAC；不含 Softmax、Top-K、ReLU、门值乘法、合并和通信。一个 MAC 是一次乘加，不能直接标作完整 FLOPs 或实际耗时。负载 n_e 是派发到专家 e 的 token 分支数，Σ_e n_e = TK。

**Dispatch / Combine**：先按专家分组计算，保留每条分支的 token 编号和门值；返回后按原 token 编号加权累加，输出行顺序仍为 t0、t1、t2、t3。独立基线逐 token 直接计算选中专家，用于比较分组执行是否保持数据对应；比较容差 1e-12。

**设备与通信账本**：单设备时全部 token 与专家都在设备 0。双设备时，token 的起始设备为 t mod 2；E0、E1 在设备 0，E2、E3 在设备 1。专家权重不随任务移动；远端分支传一份 D 维输入，再返回一份 D 维输出。若远端分支数为 R，假设每元素 b = 4 字节，则派发 RDb、返回 RDb、合计 2RDb = 16R 字节。这是理论有效载荷计数，数值运算仍为 float64；不含 token 编号、门值、填充、协议、链路跳数、训练反向通信，也不合并发给同设备的重复输入。图中的箭头表示逻辑来源与目的地，不假设中央中转节点；All-to-All 可以批量完成这类交换，实际通信成本取决于具体实现和拓扑。改变放置只改变通信，不改变数学输出。

**简化边界**：没有共享专家、容量限制、丢弃、重路由、辅助负载损失、反向传播与优化器更新，也没有 Attention、残差和归一化。本例只计算 MoE 子层，不把激活参数比例当作整个模型的计算量或加速比。原书第 30.2.3 节的语义专家命名仅可作为类比，本例用编号；第 30.2.5 节“Top-2 计算是 Top-1 两倍”仅对应固定专家结构下的专家部分，Router 的工作量并不翻倍。

依据：[原书第 30 章](https://waylandz.com/llm-transformer-book/第30章-Mixture-of-Experts-稀疏激活的秘密/) 用于章节映射；[Shazeer et al., 2017](https://arxiv.org/html/1701.06538v1) §2–4；[Switch Transformers](https://arxiv.org/html/2101.03961v3) §2.1、§5.4；[GShard](https://arxiv.org/html/2006.16668v1) §3.1、§3.3.2；[Transformers 的 Mixtral 路由实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)。2026-11-07 在线读取；说明和教学数值自行编写，没有复用论文或原书图文。

## FlashAttention（原书第 21 章）

单头教学配置，T = 4、d_k = d_v = 2，采用与 Attention 实验一致的行向量记法：

```text
Q、K: [T, d_k]，V: [T, d_v]
S = Q Kᵀ / √d_k                    [T, T]，行是 Query、列是 Key
P = softmax(S + M, axis=key)        [T, T]
O = P V                            [T, d_v]
```

Q/K/V 是已经投影好的教学激活；本页省略投影、位置编码、batch、Dropout 和反向传播。因果 Mask 在 Softmax 之前屏蔽 j > i 的 Key，包含当前位置。固定递增预设让后续 Key 块出现更大的最大值；混合预设用可复现种子生成输入。ΔK 在投影后只修改 K[3,0]，Q 与 V 不变，用于观察最后一列的分数及每行归一化变化；它不表示参数训练。开启因果 Mask 时，t0–t2 不能读取 t3，因而这些输出保持不变。

**逐行分块**：固定 Query i，把 Key/V 轴按 B ∈ {1, 2, 3, 4} 切成连续块。B = 3 时最后一块只有一个 Key。图形与数值都来自同一计算轨迹，切换块只是在查看可重建的算法状态，不对数值做动画插值。完成所有 Key 块后才得到完整行输出；中途的 a/l 只针对已处理的可见 Key。

**Online Softmax**：每个 Query 保留标量 m、l 和 d_v 维行向量 a。初始 m = −∞、l = 0、a = 0。m 是已经处理的可见分数的最大值，l = Σ exp(s_j − m)，a = Σ exp(s_j − m) v_j。对新块 C，递推为：

```text
m_C = max_{j∈C} s_j
m′ = max(m, m_C)
α = exp(m − m′)
e_j = exp(s_j − m′)                 屏蔽项 e_j = 0
l′ = α l + Σ_{j∈C} e_j
a′ = α a + Σ_{j∈C} e_j v_j
O_i = a′ / l′                       完成所有块后
```

后续块产生更大 m′ 时，旧 l 与旧 a 必须乘同一 α，从旧指数坐标换到新坐标。单独对每块做 Softmax 再平均块输出，会丢失块之间的归一化权重，一般不等价。初始化后遇到第一个有效块时取 α = 0；全部屏蔽的块取 α = 1 并保持 m/l/a，不计算 −∞ − (−∞)。全屏蔽行的 Softmax 数学上没有定义；本实现延续共享 Attention 工具的教学边界约定，返回全零输出以避免 NaN。内置因果场景每行至少可见自身。

**核对与精度**：完整基线先计算 S、P、O；在线路径只用分数块更新 m/l/a，最后逐行与基线比较。JavaScript 运算使用 float64，内部不舍入，显示时才舍入。改变分块大小改变求和次序，数学结果相同，浮点结果以绝对误差 1e-12 比较，不要求逐 bit 一致。教学页面为查看中间过程而保留基线矩阵与各块轨迹；页面本身的内存占用不是 FlashAttention kernel 的内存实现。

**存储和 I/O 账本**：页面采用一个明确的逐 Query 行流式调度，固定每个元素 b = 4 字节作假想存储计算；这不代表 JavaScript 的真实内存布局或 GPU 精度。账本只估算所列中间量，不包含共同的 Q/K/V 输入、最终 O 输出、模型权重、KV 缓存、分配器或运行时开销。

```text
朴素完整基线的 S、P 元素数：             2T²
逐行分块工作区元素数：                  B(d_k + d_v + 1) + d_k + 2d_v + 3
朴素路径 S、P 各写一次、读一次：         4T² 个元素传输
分块路径将完整 S、P 写到外部存储再读回：  0 个元素传输
```

分块工作区依次计 K/V 块、指数分数块、一个 Query、两个 d_v 维累积向量和三个标量槽位；这是便于教学的工作区约定，不是优化后的 kernel 分配公式。T = 4 的玩具下分块工作区未必更小，不能把这一点隐藏。选定中间量不落盘到 HBM 的理想计数也不是完整 HBM I/O：真实 kernel 还需要读 Q/K/V、写 O，加载次数取决于块调度、片上存储容量与硬件实现。页面不测量 CUDA kernel，不推断运行时间或加速倍数。

完整稠密 Attention 的 Query–Key 配对规模仍为 T²；因果可见配对是 T(T+1)/2。FlashAttention 的关键在于分块和融合计算，避免保存完整中间矩阵并减少 HBM I/O，而非把标准完整 Attention 变成线性配对计算。额外中间存储与推理时跨 token 保存的 KV Cache 是不同的量。

依据：[FlashAttention（NeurIPS 2022）](https://arxiv.org/html/2205.14135v2) §2.2、§3.1–3.2 的 Online Softmax、分块与 I/O 分析；[作者实现](https://github.com/Dao-AILab/flash-attention)；[原书第 21 章](https://waylandz.com/llm-transformer-book/第21章-Flash-Attention-内存优化原理/) 用于章节映射。2026-11-08 在线读取；图形、教学输入和逐行调度账本由本项目编写。

## P1 八组机制实验

以下实现均为行向量、浏览器 float64 的前向计算或给定样本的标量目标，不运行模型训练。每组六步的所有张量、图形与预测结果都消费同一纯数值模块。`#lab=` v1 状态按主题白名单校验输入长度、有限数值、参数范围/步长、版本和链接长度；不保存待执行预测或私人笔记。

### LayerNorm、RMSNorm、Softmax

四维隐藏向量沿特征轴统计：μ=mean(x)，σ²=mean((x−μ)²)，方差分母 D，不使用 D−1。LayerNorm 核心 z=(x−μ)/√(σ²+ε)，RMS 核心 r=x/√(mean(x²)+ε)。仿射 y_LN=γ⊙z+β，y_RMS=γ⊙r；仅第0维参数可编辑，其他 γ=1、β=0。参数在前向中不更新。

ε 始终正并在根号内，因此常量向量的 z=0，非退化 z 的方差是 σ²/(σ²+ε)，不强行声称等于1。共同平移对固定ε的LN和Softmax不改变结果，对RMS一般会改变；正倍缩放在固定ε下也不能普遍声称完全不变。Softmax 将同一组数字重新解释为候选分数，先除正温度，再加Mask、减最大值、指数归一化。全部屏蔽不构成合法概率分布，本工具显式返回零向量并标记。

依据：[LayerNorm §3](https://arxiv.org/html/1607.06450v1)、[RMSNorm §4](https://arxiv.org/html/1910.07467v1)、[PyTorch LayerNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)、[Softmax](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softmax.html)。

### FFN、残差与 Dropout

X:[4,2]，W1:[2,3]，W2:[3,2]，F=ReLU(XW1)W2，Y=X+Dropout(F)。逐token共享权重，无跨token读取，省略偏置。关闭ReLU后与 X(W1W2) 核对。残差Jacobian按输入行、输出列展示 I+J_F；零点ReLU导数约定0，测试有限差分避开零点，且不宣称恒等支路保证总梯度永不消失。

训练使用 m∼Bernoulli(1−p)，D(F)=m⊙F/(1−p)，本例 p≤0.75；推理为恒等。固定种子重现掩码只为教学，真实训练通常每次前向重采样。二维点图显示同一token变换前后的真实二维坐标，形状不同的中间向量不伪装成二维语义图。

依据：[Transformer §3.3、§5.4](https://arxiv.org/html/1706.03762v7)、[PyTorch Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html)。

### LoRA、QLoRA 与量化

LoRA行向量：Y=XW0+(α/r)XAB，A:[D,r]、B:[r,K]，ΔW=(α/r)AB；与 X(W0+ΔW) 核对。冻结W0，只讨论A/B的训练资格，没有优化器；可训练参数 r(D+K)。rank约束更新，不是底座压缩，二维例子r=2没有参数节省。α=0恢复底座，常见零B初始化也使更新为0。

真实QLoRA采用NF4底座存储、通常BF16计算及适配器训练，并包含双重量化和分页优化器。本页用均匀4bit底座与float64计算演示两支路的数据流，没有复现NF4或完整QLoRA训练；各精度角色与假想存储字节数分别标明。

量化独立实验输入8个数。每组 qmax=2^(bits−1)−1，extent=min(cap,max|x_group|)，scale=extent/qmax，zero_point=0；全零组scale=1。先裁剪到±extent，再以“半格远离零”舍入到整数，最后还原 q×scale。范围±qmax留下一个补码编码未使用，不能推广到所有格式。无裁剪时误差至多约半格，裁剪可能使误差更大。

理论紧密打包载荷 ceil(N×bits/8)，每组一个4B scale，忽略对齐、头与其他元数据；JS数组未真的压成该大小。除重建MSE外，固定激活 u=[1,−1,0.5,0,1,−0.5,2,−1]，把输入看成一行权重，比较 u·x 与 u·x̂ 的线性输出误差。GGUF是格式，不是单一算法。

依据：[LoRA §4.1](https://arxiv.org/html/2106.09685v2)、[QLoRA §3、附录E](https://arxiv.org/html/2305.14314v1)、[PyTorch fake quantize](https://docs.pytorch.org/docs/2.15/generated/torch.fake_quantize_per_tensor_affine.html)、[GGUF官方格式](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)。

### 解码策略

词表a/b/c。先softmax(logits/τ)，再Top-K（按概率降序、token ID破并列、恰好K项），归一化后Top-P：保留首次使累计质量达到阈值的token，再归一化。这个顺序明确，不冒充所有库组合。某些库用阈值保留Top-K并列项，数量可能超过K，本例不同。

按token ID排列概率区间，固定LCG种子得到u∈[0,1)，用严格累计概率>u选token。两步树第一步用当前温度分布，不做K/P，条件行为 [[0.4,0.3,0.3],[0.9,0.05,0.05],[0.5,0.25,0.25]]。默认第一步[0.5,0.3,0.2]时Greedy=aa，联合0.2；Beam-2=ba，联合0.27，穷举九路径核对。固定长度2，无EOS或长度惩罚，不推广为Beam必得全局最优。

依据：[Nucleus Sampling](https://arxiv.org/abs/1904.09751)、[Transformers实现](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)、[官方生成策略](https://huggingface.co/docs/transformers/main/en/generation_strategies)。

### 偏好学习、PPO、DPO 与 GRPO

输入是同一prompt的四个给定回答奖励。Bradley–Terry概率sigmoid(r0−r1)。DPO偏好标签固定回答0胜过回答1，不随奖励编辑自动反转；当前完整回答概率由softmax([policy,0,−0.5,0])构造，参考均匀，DPO loss=softplus(−β[(logπw−logπref,w)−(logπl−logπref,l)])。真实回答的log概率为逐token条件log概率之和。

单项PPO最大化 min(ρA,clip(ρ,1−ε,1+ε)A)，正负优势都按min处理；不把目标clipping误画成策略比率被强制约束。省略critic、GAE与优化器。

GRPO按总体标准差√mean((r−mean(r))²)标准化同prompt组奖励；R1原文std未指定总体/样本分母，本页明确采用总体约定，零方差优势设0。旧策略softmax([0.5,0,−0.5,0])，比率π/πold；R1样本参考项 k=πref/π−log(πref/π)−1。简化目标 mean(min(ρA,clip(ρ)A)−βKL×k)。把完整回答简化为一个动作，省略逐token平均。k不是单个回答上的完整分布KL，零优势仍可能有参考约束。奖励共同平移优势不变。蒸馏仅用奖励Softmax作玩具教师，计算交叉熵，不涉及闭源训练细节或隐藏推理。

依据：[DPO Eq.7](https://arxiv.org/html/2305.18290v3)、[PPO作者说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)、[R1 §2.2.1](https://arxiv.org/html/2501.12948v1)。

### Sparse、Linear 与 Infini-attention

四token、二维激活，Q取输入，K交换两维，V取输入。稀疏窗口包含自身，max(0,t−w+1)≤j≤t；本页仍执行稠密数值，不宣称Mask自动带来算术节省。

核特征φ(x)=ELU(x)+1，逐元素x≥0时x+1，否则exp(x)。线性因果递推：S_t=S_prev+φ(k_t)^T v_t，z_t=z_prev+φ(k_t)，o_t=φ(q_t)S_t/(φ(q_t)z_t^T)。与显式核权重φ(q)·φ(k)的归一化基线比较；不是标准Softmax Attention的等价变形。状态共2×2+2=6标量，不含可视化轨迹。

Infini每段2token：先从上一段M,z读取A_mem，再计算当前段因果Softmax A_local，合并 sigmoid(β)A_mem+(1−sigmoid(β))A_local。初始零分母返回零。Linear更新 M+=φ(K)^TV；Delta更新 M+=φ(K)^T(V−Vhat_old)，整段Vhat都从同一旧M/z读取，不能逐token写后再读。两者z均累加φ(K)行和。

碰撞例子使用两行相同Key，V历史 [[1,0],[0,1]] 与 [[0.5,0.5],[0.5,0.5]] 得到相同S/z；证明本例压缩有损，不声称整个架构任务质量排序。

依据：[Longformer](https://arxiv.org/html/2004.05150v2)、[Linear Attention Eq.7–12](https://arxiv.org/html/2006.16236v3)、[Infini Eq.3–6](https://arxiv.org/html/2404.07143v1)。

### SSM 与 Mamba

标量ZOH：Abar_t=exp(Δ_t A)，Bbar_t=expm1(Δ_t A)/A×B_t，A=0时取极限Δ_t B_t。h_t=Abar_t h_prev+Bbar_t x_t，y_t=C_t h_t。固定模式Δ_t=Δ、B=C=1；选择性模式Δ_t=Δ×softplus(0.5x_t)、B_t=1+0.25x_t、C_t=1−0.1x_t，A固定。投影是自编教学设定，不是训练后Mamba参数。

先展示固定递推，再展示输入相关参数。时间步控件显示到当前步的状态；未来状态不伪装成已执行。固定A<0时Δ越大，旧状态系数越小，旧信息衰减更快；同时Bbar也变化，不能推断输出必然更小。修正原书32.3.4“小Δ快速遗忘”的相反表述。A=0时旧状态保留系数恒1。

固定参数的两步历史 [1,0] 与 [2,−λ]，λ=exp(ΔA)，在相同初始状态下得到相同最终状态；展示固定状态的碰撞。与因果Attention并列只对照历史访问机制，不判等、不排序。省略Mamba卷积、门控与扩展投影，未复现完整模型或GPU扫描性能。

依据：[Mamba Eq.1–4、Algorithm2](https://arxiv.org/html/2312.00752v2)。上述P1来源于2026-11-08在线核对，图文和教学数值自行编写。
