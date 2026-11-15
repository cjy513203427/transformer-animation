# 章节主线与综合实验

核对日期：2026-11-15。章节结构按原书，算法边界按文末列出的原始论文或作者代码核对；讲解与图形为本站编写。

## 路由与覆盖状态

- `/chapters`：32 章和 3 个附录集中在一个可搜索、筛选的目录中。每项有核心问题、实质导读、前置知识、覆盖条目及原书链接。`#chapter-17` 等链接直接展开对应条目，没有为未实现机制创建空页面。
- `/chapters/15`：完整前向数值回放。
- `/chapters/32`：后 Transformer 架构导读与两个静态计算实验，支持小节锚点。
- `/notes`：26 个按概念组织的知识笔记，关联32章与3个附录。RLHF、Prompt Engineering 有直接导航入口，补充训练、学习率、量化方法、推理预算与 Scaling Law，串联已有实验；不是按章复刻或宣称完整实现所有代码。
- 英德版本使用 `/en/`、`/de/` 前缀，切换语言保留 hash。

机制目录的“架构扩展”组另有未编号的 RWKV / RetNet / Hyena 导读入口，直达 `/chapters/32#family-rwkv` 并展开该条目。`#family-retnet` 等架构条目锚点同样可直达；小节或条目跳转保留已编辑的计算输入。

`src/content/chapters.ts` 的状态针对每一个覆盖条目：

| 状态       | 含义                                                |
| ---------- | --------------------------------------------------- |
| 可操作实验 | 有直接可操作的例子，链接到真实实验或章节工具        |
| 已有讲解   | 有解释或结构导读，不据此宣称完整模型数值复现        |
| 原书阅读   | 本站提供定位，深入内容或真实实现仍需阅读原书/原论文 |
| 尚待补充   | 尚缺专门讲解或实验，保留明确缺口                    |

单条有实验不等于整章已完整覆盖。这些标签是内容审计，不是个人复习进度。新内容交付后仅更新确实改变的条目。

## 知识笔记与补充工具

内容在 `src/content/notebook.ts`，数值工具在 `src/math/notebook.ts`，分享状态在 `src/state/notebook.ts`。知识点可搜索、分类、链接到原书或机制实验，并可隐藏解释后复述。章节目录每项也链接到对应笔记。笔记包含来源、简化边界与核对日期；英德内容按现有目录维护。

- RLHF：InstructGPT式 SFT、偏好数据、RM排序损失、rollout、价值与参考评估、PPO更新、独立评估七阶段。分别说明 Actor、Critic、RM、Reference、旧策略；这是流程解释，真实训练仍为“原书阅读”。
- 提示工程：角色、数据边界、Zero/One/Few-shot、CoT、Self-Consistency、ToT、ReAct、评估与RAG扩展。拼装器只显示实际组成的输入，不生成虚构模型回答。
- 投票：手工给定候选而非模型采样，统计有效答案与提取失败，平局不强行选答案；票数比例不称为正确率或置信度。
- 学习率：实际执行 `L(w)=w²/2`、`w₀=2` 的12步SGD，支持固定学习率与3步线性Warmup后余弦衰减。表格使用未舍入值递推；最后一行不再更新。此二次例子的稳定区间不推广为真实网络学习率。
- 计算账本：仅用稠密模型的粗略口径 `C≈6ND` 计算FLOPs与D/N；参数和token输入单位为十亿，不预测loss、耗时或费用，不把20:1当作强制比例。

`#notes=` v1 分享固定种子0、所选知识点、RLHF阶段和公开工具输入；校验版本、类型、整数、范围与长度，不包含自由输入或私人笔记。`#rlhf`等合法锚点切换不清空工具输入，损坏状态恢复默认并提示。

新增数值验证覆盖投票平局/无效答案/多数答错、SGD与闭式解一致及振荡发散边界、调度起止与训练账本缩放；状态验证覆盖往返、非法版本/长度/类型/范围。浏览器检查覆盖入口发现、RLHF回退/跳转/复述、工具联动与分享恢复、所有知识点的英德翻译和桌面/手机宽度。

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
