# LLM 原理交互实验室

用可手算的小例子理解计算过程：修改输入、先预测、逐步核对，再隐藏说明复述原因。

## 本地开发

推荐 Node.js **24.22+（24.x）**，使用 npm；版本记录在 `.nvmrc`。当前 Cloudflare CLI 也支持 Node 22.23+（22.x）与 26.12+（26.x）。依赖版本由 `package-lock.json` 锁定。

```powershell
npm ci
npm run dev
```

打开 <http://127.0.0.1:5173/attention>（根路径也显示同一实验）、<http://127.0.0.1:5173/position> 与 <http://127.0.0.1:5173/inference-cache>。

```powershell
npm run check             # 类型检查、数值与状态测试、生产构建
npm run test:e2e          # 本机 Chrome：桌面与手机尺寸交互测试
npm run preview          # 预览 dist，http://127.0.0.1:4173
npm run build:cloudflare  # 生成并校验 Cloudflare Build Output，不发布
npm run format           # 格式化项目源码
```

浏览器测试默认使用已安装的 Google Chrome。若环境没有 Chrome，可先运行 `npx playwright install chrome`。CI 包含该安装步骤。移动测试为 Chrome 触摸/窄屏模拟，并启用 reduced-motion，不代表真实 iOS Safari 测试。

## 学习入口

两个实验都按原书的图与章节顺序组织，书中的说法与本项目的数值、记号一一对照；发现原文简化或错误时，页面写明本实验采用的修正。

### 01 Attention（`/attention`，第 3、9–12 章）

1. **整体架构**：按第 3 章自下向上的 Post-Norm 主图，点选模块查看“进来什么、做什么、交给谁”，保留位置加法、两条残差支路与 Block × N。
2. **拆开 Attention**：显示 Q/K 打分支路与 V 绕行支路。先观察一个 token，按生成 QKV、打分/缩放、Mask、Softmax、汇总 V 五步解释；公式按需展开。
3. **动手核对**：保留完整小矩阵、输入编辑、对照实验和分享链接。
4. **多头注意力**：对应第 11–12 章，按书中的“投影 → 切分多头 → 矩阵变形 → 每头 Attention → Concatenate → Wo”六步演算。图中的格子随步骤移动，展示同一份数据怎样换一种摆法；每一步都并列本例形状与书中 `[4, 16, 512]` 的形状。可切换 1、2、4 个头，先预测再验证“只改一个头的 K，哪些结果会变”。

默认开启因果 Mask 并观察中间位置 t1“喜欢”，让未来的 t2“学习”被明确排除。三个词块是人为教学切分，不是真实 tokenizer 的输出。关闭 Mask 明确标为双向读取对照。第 3 章的 Post-Norm 主图和第 15 章 GPT-2 的 Pre-Norm 顺序分别说明。单头输出 O 对应书中第 11–12 章 A 的一个头（一片 `[T, d_key]`）。

### 02 位置编码（`/position`，第 5、14、25 章）

左侧是同一条 Attention 计算链，标出三个可以加入位置的入口：① 加法位置编码加到 X（Q、K、V 都带位置）、② RoPE 旋转 Q 与 K（V 不旋转）、③ ALiBi 在 Softmax 前给分数加距离偏置。右侧七步：

1. **位置盲**：书中“我爱你 / 你爱我”实际计算，没有位置编码时“我”的输出完全相同；因果 Mask 对照“我爱你 / 爱我你”。
2. **正弦表**：每对 sin/cos 维度是一根转速不同的指针，配合波形图与位置表。d_model = 64 时与书中第 5 章表格逐格一致（书中截断到两位小数）。
3. **加到输入**：用书中“L”的嵌入演示两个相同的词加上不同位置后变得不同，并拆开加法点积的四项。
4. **RoPE 旋转**：d = 8 的四个二维平面，θ 依次为 1、0.1、0.01、0.001。
5. **相对距离**：先预测“同时平移两个位置，分数会不会变”，再核对；曲线展示单个分数随距离起伏、并不单调下降。
6. **ALiBi**：8 个头的标准斜率、偏置矩阵与逐行 Softmax，先预测哪个头更集中在附近。
7. **三处对照**：方法、加入位置、直接改变的量、V 是否带位置与常见误区。

正弦表与 RoPE 两步可以“逐个位置播放”，指针和向量只在相邻位置之间做转动过渡，跳跃时直接到位；减少动态效果时不做过渡。

### 03 训练、生成与缓存（`/inference-cache`，第 2、16、22 章）

同一个两层、单头、d = 2 的玩具解码器（词表 a–f，无语义，固定教学权重）。左侧概念图随步骤切换，格子都带文字状态；右侧六步：

1. **训练并行**：样本 a b c d e 错开一位得到输入与目标，一次因果前向算出 4 行的分布与 loss。先预测“只改输入 x_j，哪些行会变”，再核对：之前的行逐位不变。
2. **无缓存基线**：每生成一个 token 就把整段重算一遍；表格核对重算出的旧行与上一步完全相同。
3. **Prefill**：提示 a b c 一次前向，每层每个位置的 K、V 写入缓存，最后一个位置选出第一个新 token。
4. **Decode 一步**：只为新 token 算 q、k、v，追加缓存，新 q 读全部历史；可以逐步播放 1–5 次 Decode。
5. **缓存对照重算**：两条路线逐步比较 logits 与选出的 token，并列出实际执行的 K/V 投影与打分次数。
6. **缓存失效**：改一个提示 token 或滑动窗口，先预测第 2 层哪些位置失效，再重算核对。

## 已实现的范围

- Vite + TypeScript + React，KaTeX 排版，原生 SVG 与矩阵表格，无模型服务。
- `/attention` 单头计算：X 投影 → 点积与缩放 → Mask → 行 Softmax → 加权输出；多头：投影、view、transpose、各头 Attention、Concatenate 与 Wo，支持 1、2、4 个头与投影后 K 干预。
- `/position` 位置编码：位置盲、正弦表、加法、RoPE 旋转与相对位置、ALiBi，以及三处对照。
- `/inference-cache` 训练、生成与缓存：两层玩具解码器上的训练并行、无缓存基线、Prefill、Decode、缓存与重算对照，以及改动前缀或滑动窗口后哪些缓存失效。
- 播放、暂停、前进、后退、跳步、速度、从头看与恢复默认示例；修改参数会暂停播放并按新输入重算。
- 先预测再验证：只改 V（单头）、只改一个头的 K（多头）、同时平移 RoPE 位置、比较 ALiBi 斜率、只改训练输入 x_j、改前缀后哪些缓存失效。
- 分享链接：Attention 使用 v3 状态（兼容 v1、v2 链接，多头部分从默认示例开始），位置编码与训练缓存实验分别使用独立的 `#pos=`、`#inf=` 状态；都校验版本、结构、范围和长度，坏链接恢复默认示例。
- 主题延迟加载、后台暂停、键盘控件、窄屏布局（窄屏顶部提供实验切换）与减少动态效果支持。

**尚未实现：**KV 架构对照，MoE，FlashAttention。导航将这些主题明确标为未实现，没有空实验页面。

## 目录

```text
src/
  features/attention/   Attention 实验：架构导读、单头、动手核对、多头
  features/position/    位置编码实验：计算链入口图与各步骤组件
  features/inference/   训练、生成与缓存实验：缓存概念图与各步骤组件
  math/                 纯数值计算及数学性质测试（attention、multihead、position、inference）
  components/           共享外壳、矩阵与公式视图
  state/                可分享状态、预设、版本校验与测试
  content/              步骤解释、实验目录与参考来源
  styles/               视觉规则与响应式布局
tests/                  Playwright 浏览器交互测试
docs/                   数值约定与 Cloudflare 准备说明
```

## 数值与状态约定

固定行向量记法，忽略 batch 与偏置。单头实验 T = 3、D = d_k = d_v = 2；多头实验 d_model = 4、d_key = d_model ÷ num_heads；位置编码实验的各步骤维度见数学说明。所有结果由浏览器实际运算得到；显示时舍入，内部使用 JavaScript float64。词语仅为位置标签，小向量不是模型语义表示，权重是固定教学数字。

修改实际输入会暂停播放并重建计算。ΔV 滑杆为待执行输入，点击“验证预测”后才生效；分享仅包含已经计算的输入。修改 Mask/缩放跳到相应步骤，修改 X 回到投影。重置步骤保留参数，“恢复基础示例”恢复所有默认状态。离开页面到后台会暂停，返回后不自动恢复播放。

普通浏览器分享链接只能在相应站点可访问时使用；本地 `127.0.0.1` 链接需要接收者也运行此项目，不等于已发布网站。当前不收集笔记、进度或个人数据，不接入分析服务。

更多内容见 [数学说明](docs/math.md) 与 [Cloudflare 准备](docs/cloudflare.md)。

## 参考

- [Transformer 架构：从直觉到实现](https://waylandz.com/llm-transformer-book/)：第 3 章为主架构图，第 6、9–12 章为 Attention 解释主线，第 5、14、25 章为位置编码，第 2、16、22 章为训练、生成与缓存。原书源文件见 [waylandzhang/llm-transformer-book](https://github.com/waylandzhang/llm-transformer-book)（MIT）。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：§3.2 Attention，§3.5 正弦位置编码。
- [RoFormer（RoPE）](https://arxiv.org/abs/2104.09864)、[ALiBi](https://arxiv.org/abs/2108.12409)。

来源核对：单头部分 2026-11-05；多头与位置编码部分 2026-11-06，对照原书 GitHub 源文件 commit 4a77788。此次环境的网络策略不允许访问 waylandz.com 与 arXiv，网站链接沿用原书网站的章节地址格式，论文公式按论文记法编写，未在线复核原文。说明和图形自行编写，没有复制书中的文本或图片。
