# 章节主线与综合实验

核对日期：2026-11-15。章节结构按原书，算法边界按文末列出的原始论文或作者代码核对；讲解与图形为本站编写。

## 路由与覆盖状态

- `/chapters`：32 章和 3 个附录集中在一个可搜索、筛选的目录中。每项有核心问题、实质导读、前置知识、覆盖条目及原书链接。`#chapter-17` 等链接直接展开对应条目，没有为未实现机制创建空页面。
- `/chapters/15`：完整前向数值回放。
- `/chapters/32`：后 Transformer 架构导读与两个静态计算实验，支持小节锚点。
- `/notes`：仅为旧链接保留的跳转页。`#notes=` 与 `#rlhf` 等锚点按 `src/content/experimentLinks.ts` 打开对应实验，并尽量换算为该实验的 `#lab=` 状态。
- 英德版本使用 `/en/`、`/de/` 前缀，切换语言保留 hash。

SSM / Mamba 实验的“进一步理解”链接到 `/chapters/32#family-rwkv` 并展开 RWKV 条目。`#family-retnet` 等架构条目锚点同样可直达；小节或条目跳转保留已编辑的计算输入。

`src/content/chapters.ts` 的状态针对每一个覆盖条目：

| 状态       | 含义                                                |
| ---------- | --------------------------------------------------- |
| 可操作实验 | 有直接可操作的例子，链接到真实实验或章节工具        |
| 已有讲解   | 有解释或结构导读，不据此宣称完整模型数值复现        |
| 原书阅读   | 本站提供定位，深入内容或真实实现仍需阅读原书/原论文 |
| 尚待补充   | 尚缺专门讲解或实验，保留明确缺口                    |

单条有实验不等于整章已完整覆盖。这些标签是内容审计，不是个人复习进度。新内容交付后仅更新确实改变的条目。

## 知识笔记与补充实验

笔记内容在 `src/content/notebook.ts`，归属页面在 `src/content/experimentLinks.ts`，由 `src/components/RelatedNotes.tsx` 渲染在该页的“进一步理解”中（标题、两段解释、适用边界）。P0 实验、P1 实验与第 15 章共用这个面板；`tests/notebook.spec.ts` 检查 26 条笔记都能在对应页面读到。

补充实验（`src/content/p1/learning.ts`、`learningFrames.ts`，数值在 `src/math/learning.ts`）：

- 训练一步：固定 h:[1,2]、W:[2,3]，单样本交叉熵与 SGD。梯度与中心差分核对；所有 logits 同加常数不改变更新。只训练输出投影，不是完整 Transformer 反向。
- 策略更新：两个完整回答作为终止动作，按 π_old 精确枚举期望；梯度含 PPO 裁剪项与对 π_ref 的完整 KL，与差分核对；完整期望下价值基线平移不改变更新。未实现逐 token GAE 与 Critic 训练。DPO 步骤固定 A 为 chosen。
- 学习率轨迹与计算账本（训练实验的“进一步理解”）：`L(w)=w²/2`、`w₀=2` 的 12 步 SGD，可选 3 步 Warmup 加余弦衰减；稠密模型粗估 `C≈6ND`。数值仍随 `#lab=` 分享。
- 提示链、投票、思路搜索：模板字段、手工候选与固定访问顺序，浏览器做精确算术核对。投票对照比较的是选出的答案（平局单独标记），不是票数；票数比例不称为正确率或置信度。

## 分词、优化器与上下文扩展

- `/tokenization`（`src/math/bpe.ts`、`src/state/bpe.ts`、`src/features/bpe/BPELab.tsx`）：Sennrich 等人 Algorithm 1 的字符级 BPE，词尾符号 `</w>`，相邻对按单词次数加权计数，从左到右不重叠合并，最多 12 次。原论文未规定平局，本页取字典序最小的一对。不重叠时每次合并使 token 总数减少该对的计数；同一字母连续出现时减少量更小（单测覆盖 `aaa`）。Embedding 行是按 ID 生成的教学数值。字节一节只显示 UTF-8 字节，不运行真实 tokenizer。语料限 6 个单词、每词 1–12 个小写字母，次数 1–50；自由文本不进入链接。
- `/optimizer`（`src/math/p1/optimizer.ts`）：L = ½(w₀² + b·w₁²)。AdamW 按 PyTorch 先乘 (1 − ηλ) 再做 Adam 更新；“Adam + L2”把 λw 加进梯度。性质核对：第 1 步 m̂ = g、v̂ = g²；SGD 中 L2 与解耦衰减相同；损失整体放大 100 倍时 Adam 更新基本不变（相对误差 ≤ 1e−6）。
- `/context-extension`（`src/math/p1/context.ts`）：按 YaRN（arXiv 2309.00071）核对：θ_d = b^(−2d/|D|)；PI 用附录 A.1 的 m·L/L′（正文 Eq. 9 的抽取文本与之矛盾，按附录实现）；NTK-aware 底数 b·s^(|D|/(|D|−2))；NTK-by-parts 的 r = L/λ、γ 斜坡 α = 1、β = 32；温度 √(1/t) = 0.1·ln s + 1。“训练中见过”按角度判断：λ_d ≤ L 的维度见过全部角度，其余只见过 [0, L·θ_d)。这是角度层面的判断，不测量模型质量。

## 参数量与计算量（`/model-budget`）

计算在 `src/math/budget.ts`，分享状态与预设在 `src/state/budget.ts`，页面在 `src/features/budget/BudgetLab.tsx`。覆盖原书第 4.5、7.8、11.7、12.12、15.7、18.11 节与附录 A.2、A.3、A.6 的参数与计算量算式。

```text
d_h = d / H,  K/V 宽度 = H_kv · d_h
每层 Attention = 2d² + 2d·H_kv·d_h   (+ 2d + 2H_kv·d_h，若含偏置)
每层 FFN       = 2·d·d_ff (GELU) 或 3·d·d_ff (SwiGLU)   (+ 偏置)
归一化         = (2L+1) · (2d LayerNorm 或 d RMSNorm)
N = V·d + P·d + L·(Attention + FFN) + 归一化 + [不共享时] V·d
前向 FLOPs(s) = L·(2s·M + 4s²·d) + 2s·d·V，M = 每层 Attention 与 FFN 权重元素数
单步 Decode   = 2·(L·M + d·V) + L·4s·d
训练每 token  ≈ 3 × 前向每 token
```

1 次乘加计 2 FLOPs；不计归一化、Softmax、激活、偏置加法与查表；配对项按完整 s×s 计，因果实现约减半，与 Kaplan 等人表 1 的 `2·n_layer·n_ctx·d_attn` 一致。只覆盖稠密 Decoder-only，不含 MoE、MLA、优化器状态、激活与 KV 缓存。

预设与独立核对：GPT-2 Small 124,439,808（逐项与原书 15.7 节一致），原书第 18 章模型 44,503,040 / 不绑定 70,103,040（18.11 节），Llama 2 7B 6,738,415,616 与 70B 68,976,648,192（与逐张量形状清单相同）。Llama 2 的层宽与头数按 NousResearch 镜像的 config.json 核对，Meta 原仓库需要授权。`#budget=` v1 校验版本、类型、整数范围、长度，以及 `H | d_model`、`H_kv | H`；输入框中的无效组合会提示原因并保留最近一次有效结果。

## 第 15 章：完整前向

采用 3 个 token、4 个词表符号 a–d、隐藏维 4、2 个头（每头 2 维）、1 或 2 个 Pre-Norm Block、FFN 中间维 6。参数用 uint32 LCG 从种子生成；前向不会更新权重。

```text
token_ids → E 查表 → 加正弦位置编码 → H_0
每个 Block l：
  U_l = LN(H_{l-1})
  A = MHA(U_l)                        含每头 Softmax、拼接、W_O
  R_l = H_{l-1} + A
  Z_l = LN(R_l)
  G = GELU(Z_l W_1)
  F = G W_2
  H_l = R_l + F
H_f = LN(H_L)
logits = H_f E^T                      输出权重绑定
p = softmax(logits, axis=vocabulary)
next = argmax p[T-1]                  平局取较小 ID
```

LN 逐 token 统计，γ=1、β=0、ε=1e-5。GELU 使用 tanh 近似，系数 0.044715。采用 GPT 风格 Pre-Norm 顺序与固定正弦位置，省略 batch、偏置、训练的 Norm 仿射参数、Dropout、反向与优化器；不是 GPT-2 配置或实训权重。a–d 无语义，没有执行真实 Tokenization/BPE。

计算、轨迹、解释与渲染分别在 `src/math/forward.ts`、`src/content/forward.ts`、`src/features/chapters/ForwardChapter.tsx` 中组织。内部 float64，显示时才舍入。轨迹包含 7 + 8L 个可重建步骤，支持播放、暂停、回退、跳步、恢复预设；编辑参数会停止播放并重建步骤，页面进入后台会暂停。

分享使用版本化 `#forward=` 状态，校验 token 范围、种子、层数、步骤与展示选项，并限制链接长度。`#scope` 是合法阅读锚点。因果对照通过替换最后一个输入，核对最早位置的最终词表分布；关闭 Mask 是明确标注的双向对照。

依据：[原书第 15 章](https://waylandz.com/llm-transformer-book-en/chapter-15-full-forward-pass/)、[第 18 章](https://waylandz.com/llm-transformer-book-en/chapter-18-model-py/)、[minGPT Block / NewGELU](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)、[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

## 第 32 章：架构关系与计算口径

导读覆盖 S4/LTI SSM、Mamba-1、Mamba-2/SSD、RWKV-4（2023）、RetNet（2023）、Hyena（2023）及 Jamba/Hybrid。区分序列混合器与 FFN/MoE 两个轴，不给出无配置的质量或速度排行榜。

Mamba-1 模块导读包含输入投影、短因果逐通道卷积、SiLU、选择性参数、SSM 扫描、门控、输出投影和预归一化残差包装。SSM 公式明确用单通道行向量，原有标量实验继续作为局部数值核对；完整模型训练未实现。

SSD 小例子固定 B=C=1、初始 h[-1]=0、共享保留比例 a，实际计算四步递推与下三角矩阵，输出行向量满足 y=xL^T。它只验证受约束的标量切片，不能推广为任意 SSM 与标准 Softmax Attention 等价。矩阵在浏览器物化仅用于小例子验证。

八层混合账本采用以下教学设定：

```text
D = 4；Attention：2 Query 头、2 KV 头，d_k = d_v = 2
SSM：D_inner = 8，N = 4，卷积宽度 k = 4
KV 元素数 = batch × attention_layers × tokens × 2 × (2+2)
SSM 状态数 = batch × ssm_layers × 8 × 4
卷积历史数 = batch × ssm_layers × 8 × (4-1)
字节数 = 元素数 × bytes_per_element
```

卷积按最少 k−1 个旧输入计数，真实实现可能使用更多槽位。账本排除权重、其他激活、工作区、分配器，不是显存测量。原始 Jamba 论文发布配置比例为 1:7；本站只借用比例，不复刻其真实尺寸和层序。保留完整 Attention 层时全序列仍有二次配对项，单步 Decode 的历史访问另算。

FFN/MoE 参数另外计数：Dense 为 2DF，专家数 E=4、选中数 K=2，Router 为 DE；这里只统计 FFN/MoE 子层，改变 MoE 不影响历史缓存。短序列 T=1 可观察替换 Attention 后 KV 减少、但 SSM/卷积状态让总历史存储增大的反例。

分享使用版本化 `#architecture=` 状态，校验八层配置、输入、保留比例、展示选项与长度；小节锚点不会清空已编辑状态。复述模式可以隐藏图与说明。

依据：[Mamba §3.2–3.4](https://arxiv.org/html/2312.00752v2)、[SSD §2.4 / §5](https://arxiv.org/html/2405.21060v1)、[Jamba §2–3](https://arxiv.org/html/2403.19887v2)、[S4](https://arxiv.org/abs/2111.00396)、[RWKV](https://arxiv.org/abs/2305.13048)、[RetNet](https://arxiv.org/abs/2307.08621)、[Hyena](https://arxiv.org/abs/2302.10866)。
