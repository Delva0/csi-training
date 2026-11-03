# Stage 1 训练设计

源码即知识。目标：压缩、接通从问题现象到关键源码及其关系的调查路径，积累难以快速重建、可供未来 query 复用的知识。

```math
\text{压缩收益}
=\text{未来复用频率}\times\text{单次节省的重建成本}
-\text{平均匹配、理解与使用成本}.
```

成本统一用 tokens 或时间。训练 loss 记录知识更新量。

## 数据

| 集合 | 选择 |
| --- | --- |
| train | classifier 中明确版本的真实 query；排除 v0.30.0rc1、v0.30.0rc2 |
| val | 全集中有根因或有效 workaround 依据的真实 query；排除 train/test ID 和测试版本，允许未报告版本 |
| test | 冻结的 v0.30.0rc2 问题 |

- query：issue 标题＋首帖＋URL，不附参考结论。
- 参考：注明根因／workaround 的证据、来源与适用条件，未知部分单列。
- 选题：脚本优先，少量疑难交 agent；首帖已有答案的题单列。
- 数据划分冻结；具体题数和筛选统计随数据集保存。

## 采样

step 为 $`t=0,\ldots,T-1`$；每 step 从同一 $`K_t`$ 出发。

| 参数 | 含义 | 第一版配置 |
| --- | --- | --- |
| $`B`$ | 一个 step 的采样点数，即 minibatch／reviewer 数 | 4 |
| $`b`$ | 每个 minibatch 的 forward 数 | 4 |
| $`k`$ | 后续抽样的候选组数 | 4 |
| $`c`$ | forward、原生复盘、reviewer 共用的并发上限 | 4 |
| $`\alpha_{t,j}`$ | 第 $`t`$ step、第 $`j`$ 个 minibatch 的目标距离；小值偏细，大值偏粗 | 按下式分层抽取 |
| $`\beta`$ | 新版本偏置；0 为各题等权 | $`\ln 2`$ |
| $`s`$ | 采样随机种子 | 17 |
| $`T`$ | 总 step 数 | 运行时指定 |

**版本权重。** 按 release 时间排列，最新位置为 0、最旧为 1；多版本题取平均位置 $`a_q`$。

```math
w_q=e^{-\beta a_q}.
```

**距离。** Qwen3-Embedding-8B 向量 $`v_q`$；用训练集最大 cosine 距离归一化。

```math
\delta_{ij}=1-\cos(v_i,v_j),\qquad
M=\max_{i<j}\delta_{ij},\qquad
d_{ij}=\delta_{ij}/M.
```

$`M`$ 冻结；$`M=0`$ 时取 $`d=0`$。

**目标距离。** $`\alpha_{t,j}\in[0,1]`$ 为第 $`t`$ step、第 $`j`$ 个 minibatch 的目标题间距离，$`j=1,\ldots,B`$。$`\alpha=0`$ 偏相近题，$`\alpha=1`$ 偏远题。

**分层抽取 $`\alpha`$。** $`r_t`$ 为训练进度，$`p_t`$ 为粗层目标比例，$`n_t`$ 为粗层 minibatch 数：

```math
\begin{aligned}
r_t&=\begin{cases}t/(T-1),&T>1,\\0,&T=1,\end{cases}\\
p_t&=\frac14+\frac12r_t,\\
n_t&=\min\!\left(B-1,\max\!\left(1,\left\lfloor Bp_t+\frac12\right\rfloor\right)\right),\qquad B\ge2.
\end{aligned}
```

$`B-n_t`$ 个细层 $`\alpha\sim U[0,0.5)`$，$`n_t`$ 个粗层 $`\alpha\sim U[0.5,1)`$，抽完打乱，得到 $`\alpha_{t,1},\ldots,\alpha_{t,B}`$。$`B=4`$ 的前／中／后为 **3细1粗 → 2细2粗 → 1细3粗**。

$`B=1`$ 时按 $`p_t`$ 选粗层，否则选细层，再在该层抽 $`\alpha`$。$`\alpha`$ 与题目使用独立随机流，由 $`s`$ 确定。

**每个 minibatch 独立进行双阶段抽样：**

1. 首题：从全训练集按 $`w_q`$ 抽取。
2. 后续：以已抽题目集合 $`S`$ 计算候选，再有放回抽取，直到共 $`b`$ 条。

```math
\begin{aligned}
E_{\alpha_{t,j}}(q,S)&=\max_{x\in S}|d(q,x)-\alpha_{t,j}|,\\
\mathcal C&=\operatorname{top}_{k}(-E_{\alpha_{t,j}}),\\
p(q\mid\mathcal C)&=\frac{w_q}{\sum_{x\in\mathcal C}w_x}.
\end{aligned}
```

- top-$`k`$ 按不同 embedding 输入计名额；组内原题保留各自权重。
- 有放回抽样允许同题多次执行，也允许同类不同题。
- 可用 classifier 限定与首题共享类别的范围；不足两个输入组时回退全池。

## Forward 与课程

$`K`$ 默认仅为 skills；`learn_memory` 默认关闭，开启后 $`K`$ 为 skills＋环境 memory，并跨 step 保存 memory。每条 forward 使用 $`K_t`$ 的私有副本和新对话，默认只读知识；开启 memory 后显式安装 Memory 基础组件读取它，online／auto 写学习仍独立控制。
离线 opt 按上层显式组件映射，平等支持仅 skills、仅 memory 或两者；不选择训练默认策略。前两种在线优化仍使用显式装配的 skills＋memory。

调查目标：结合观测与源码，尽可能得到有效 workaround 和可落到具体仓库 issue 的根因。训练与评测共用，query 保留原文；普通用户会话不追加。

- $`K_0`$：复制 [`profiles/train_init/`](profiles/train_init/)，只含 skills 和公开运行配置，不提供初始 memory；指定知识或续训时使用所给 $`K`$。
- 后续 step／stage 继承已发布 $`K`$。
- 同一条继续写入的训练历史／resume 保持相同 `learn_memory` 设置；本轮消融使用独立 run。只读评测可独立选择装配或不装配 Memory。
- 源码预先准备，forward／reviewer 共享只读，不计入 $`K`$。

单条 forward 的累计输出预算（含 reasoning）固定：

```math
C_t=4096.
```

| 学习环节 | 触发与作用 |
| --- | --- |
| online（可选） | forward 中随调查学习，写入该条轨迹的私有知识 |
| 原生 auto（可选） | completed 后按计数触发：memory 每 10 query，skills 每 10 模型轮；相应写工具会重置计数 |
| minibatch review | 比较该组全部轨迹及可选 online／auto 的变化；预算截断的轨迹也参与 |

- 用完预算即保留已有轨迹；未解完整题，仍可学习已核实的局部路径。截断本身不决定路线成败或应当收手的时机。
- review 不消耗 $`C_t`$，默认不设累计 token 或步数截断。
- 同 step 的 forward 共用同一预算值，各自独立计量。

## 更新

每个 reviewer 比较自身的 $`b`$ 条轨迹，得到 $`K_{t,j}`$；全部 $`B`$ 份更新合并为 $`K_{t+1}`$。

```math
\begin{aligned}
K_{t,j}&=\operatorname{Review}\left(K_t,\{\tau_{t,j,\ell}\}_{\ell=1}^{b}\right),\\
K_{t+1}&=\operatorname{Merge}_{K_t}(K_{t,1},\ldots,K_{t,B}).
\end{aligned}
```

**Reviewer：**

- 当前证据仅含自己的 minibatch；每条 forward 的 `trace.json` 原样保存 messages，包括 reasoning、tool calls 和 tool results。
- 执行状态、stop reason、消息外的模型错误、已知／未知 usage 与输入 Git commit 单列；程序的预算耗尽提示不冒充模型答案。
- 优化已观察到的调查路径；同题比较速度／代价时，对齐已核实的局部进展；成功／失败、到达／未到达的差异仍可学习。
- 保留证据充分时停止、避免重复核验、以更短路径取得所需证据的有效策略。
- 不同题侧重共性与适用条件；多个条件同时变化时，区分观察与原因，保留有用观察和候选解释。
- 按需查询同 query 的 Git 历史，检查旧知识是否实际应用、是否减少重建、是否出现新反例。
- 沿用 Hermes 学习／维护准则；先改已有知识，保留证据、版本和适用边界。
- 自己提交知识修改与当前 traces；无知识变化也提交本轮证据。
- 离线 reviewer 仅维护上层选择的知识；skills 沿用 lint 反馈，memory 沿用原生添加与修删审批规则。pending 提案由程序保存，不进入 $`K`$、Git 知识或模型视图。

材料目录与权限统一列在下方「Agent 沙箱文件」；`batch_i` 是一个训练 step，`minibatch_j` 是该 reviewer 自己的组。

`history.json` 仅列同题历史的 commit、输入 commit 与 trace 路径；无匹配为 `[]`，按需读取。当前知识在选中的组件目录，输入／历史版本通过完整 Git 读取，不在每条 forward 复制全库快照。原生运行日志和成本留在宿主，不复制成 reviewer 的 sources、Markdown view、HTTP 树或多层索引。

可选 online／auto 在 forward 内产生的实际知识差分和自动复盘事件保留在 trace 的可选字段，区别于共同输入 $`K_t`$。旧离线材料未提供生成 Git 身份时，`input_commit` 明确未知，保留一致的生成知识指纹；不把当前 review 目标知识冒充原输入。

**Merge：**

1. 从正式 checkpoint $`K_t`$ 创建真实辅助 commit $`W`$（`Working knowledge`）：唯一 parent 为 $`K_t`$，去掉旧 traces，知识不变；$`W`$ 在 reviewer／merger 的 Git 中可见。
2. 每个 reviewer 提交的唯一 parent 为 $`W`$，程序原样导入；全部 $`B`$ 个提交完成后，交给一个 Merger 从 $`W`$ 整合。
3. Merger 使用 Git 合并，并审阅重复内容、适用边界、引用与结构改进空间；没有文本冲突时也审阅，按需重组知识。审阅后无需进一步调整即可交回。
4. 程序校验提交、知识格式与权限、原始 traces 完整性，再将完整树压平发布 $`K_{t+1}`$；正式历史一 step 一个 commit，唯一 parent 为 $`K_t`$。

$`W`$、reviewer／Merger 的提交，以及程序复制 Merger 最终树后生成的 `Integrated reviewer updates`，均属私有辅助历史，不进入正式 $`K_t\to K_{t+1}`$ 链。
每个正式 commit 保存新知识和本步全部 traces；旧 traces 留在历史，不计入 $`K`$ 或 loss。

## Loss

$`\Delta(K,K')`$：跨本次选中知识文件抵消相同非空原文行，保留重复次数；剩余内容分别估计新增、删除 tokens。loss 取各 reviewer 合并前的更新量均值：

```math
\widehat L_t=\frac1B\sum_{j=1}^{B}\Delta(K_t,K_{t,j}).
```

- 合并后的净变化另记；traces 和运行文件不计入。
- 未完成或不可计量的 review 不作零更新；主均值记 NA。
- loss 观察更新幅度；val 衡量质量与实际成本。
- 完整行搬移、文件拆并／改名、段落换序、空白行变化不计；改写、缩进、重新折行仍可能计入文本差异。

## 验证

**固定 $`K`$：** 同轮各 checkpoint 使用同一批 query、相同 repeats 和输出预算 $`C`$；每次新对话，online／auto 关闭。

**提交评测：** eval 为传入的现成 Agent 后装载专用 hook，再执行 forward。提示词为：

> You have a limited budget. During your investigation, use `submit_to_judge` to submit workarounds or root causes in natural language to a rigorous judge. The judge will objectively determine your score based on everything you submit.

`submit_to_judge(text)` 只记录自然语言，可多次提交及修正，不即时判分、不返回得分、不结束调查。程序按序保存每次提交及当时累计 forward 输出 tokens、未知用量和时间；被截断而未执行的调用不算提交。

**Judge：** 同一 query 的所有 checkpoint／repeat forward 共用一个只读文件沙箱、一个 Judge 会话，共享原题与唯一一份冻结有界参考答案。各 forward 的实际任务要求和有序全部提交独立存放。使用匿名 forward 编号，程序单独保存 checkpoint／题目／repeat 映射。

Judge 的 system 说明布局和评分规则，只有 `read_file`，按需读取本题文件，不将大正文重复塞入 system。它逐 forward 独立输出严格整题 Q 与 workaround／root_cause 结论、证据和提交 ID；程序仍逐 forward 计算 Q/U/D，再按 repeats 和题目聚合。没有提交的 final answer、reasoning、工具结果、skills、原始 trace 和别题文件均不挂载；不自动展开参考路径指向的 raw 文件。参考不足记 unassessed／NA，不另作调查、轨迹比较、优化建议或加权总分。Judge 实际模型请求、读取工具次数、tokens 与耗时按 query 会话单列一次，不向多个 forward 重复分摊费用；API 失败重试沿用原机制。

任务要求由组件的只读 `task_requirements` 或调用方 `agent.metadata["task_requirements"]` 声明，并核对确实存在于实际 system／developer 原文。标准 TaskPrompt 自动提供实际注入的目标和报告者环境限定；自定义上下文未声明时在首请求前报配置错误，显式空列表表示调用方确认没有额外任务要求。不向 Judge 携入完整 system、skills 索引或工具说明；旧轨迹不补造提交。

每次提交的 token／时间是实际记录，最终成果引用早期 ID 不代表当时已经完整成立，本版不推算首次有效成本。参考不提供给 forward，forward 原有在线工具保持。

checkpoint $`t`$、题 $`i`$、重复 $`r`$ 分别记录：

```math
q_{tir}\in\{0,1,\mathrm{NA}\},\qquad
u_{tir}=T^{\mathrm{out}}_{tir}/C_{tir},\qquad
D_{tir}=\text{用户交付秒数}.
```

交付耗时保留原口径：调用方逐 Agent 记录复制／快照及运行沙箱准备之后的组件构造耗时，eval 的后装载 hook 耗时单列并计入本 Agent 构造；再加实际 query 执行耗时，得到 $`D_{tir}`$。批量预构造后的排队、其它 Agent 和 Judge 耗时不计入。各分量分别留存，支持重算。

三项指标先在题内求 repeats 均值，再对题等权：

```math
\widehat Q_t=\operatorname{mean}_i\operatorname{mean}_r q_{tir},\qquad
\widehat U_t=\operatorname{mean}_i\operatorname{mean}_r u_{tir},\qquad
\widehat D_t=\operatorname{mean}_i\operatorname{mean}_r D_{tir}.
```

- 成功与失败均记录真实 output 和时间；reasoning 计一次，已知 retry 纳入实际消费。
- 缺失不填零；同时报告有效样本数，基础设施异常和 judge 开销单列。
- 三项并列，不合成总分；程序不自动选择或回滚 checkpoint。
- val 的题数、repeats、频率与预算由运行配置给定；test 不参与选择。

用户持续学习实验另设：每题新对话，共享更新的 $`K`$，online／auto 开启；普通基线每题空 $`K`$、无学习。

训练阶段按 `StageConfig.evaluation` 的显式时机和 checkpoint 列表自行装配独立、关闭 online／auto 的 Agent，再调用 `csi.eval.run_fixed`；
持续学习对照实验由 `train/online.py` 单独组合。

## 训练 pipe

~~~text
learn_memory = False

one_minibatch(queries, K, C, view, knowledge, t, j):
    completed = parallel(forward(q, copy(K), C) for q in queries)
    # All reviewers share a real private W (parent Kt) and commit directly on W.
    return opt.review_group(view, knowledge=knowledge,
                            batch_id=f"batch_{t}/minibatch_{j}", sources=completed)

for t in range(T):
    alphas = stratified_alpha(B, t, T, seed)
    minibatches = [sample(b, k, alpha, beta) for alpha in alphas]
    view, knowledge = train.prepare_step_view(K, learn_memory)
    reviews = parallel(one_minibatch(Q, K, C[t], view, knowledge, t, j) for j, Q in enumerate(minibatches))
    # Merger integrates every group; the published commit has sole parent Kt.
    opt.merge_reviews(view, reviews, knowledge=knowledge)
    K = train.collect_knowledge(view, knowledge)
    validate(K) if scheduled
~~~

train 等该 minibatch 的 forward 完成后，将记录交给 opt；opt 在该 reviewer 的私有沙箱只导出一次极简材料。离线 caller 也可交现成的 `forward_*/trace.json`；旧 source JSON 输入仅作兼容读取，不恢复旧复制／转写链。离线更新原地维护显式选中的 Skills／Memory 组件目录，通过已有 `.git/` 读取历史。整 step 合入后再进入下一步，合入失败保留原 $`K`$。
训练 pipe 自己管理持久目录、初始模板、每 step 的输入与输出选择；opt 不管理这些目录。
评测仍负责新 query 的执行、judge 与结果聚合；调用方提供独立 Agent，并负责其交付后快照和关闭。

源码按 [48 仓清单](experiments/source_catalog.json) 普通 clone 到 `repos/<owner>/<repo>/`，每个远端一份；工作树与本地 refs 保持 clone 所得状态。

| 层次／用途 | 仓库 |
| --- | --- |
| 推理／模型 | `vllm-project/vllm-ascend`、`vllm-project/vllm`、`huggingface/transformers`、`mlc-ai/xgrammar` |
| 调度／KV 缓存 | `ray-project/ray`、`LMCache/LMCache`、`LMCache/LMCache-Ascend`、`kvcache-ai/Mooncake`、`ModelEngine-Group/unified-cache-management`、`Ascend/memcache`、`openeuler/GVirt`、`openeuler/yuanrong-datasystem` |
| Torch／NPU 适配 | `pytorch/pytorch`、`Ascend/pytorch`、`Ascend/op-plugin` |
| 图编译／执行 | `Ascend/torchair`、`cann/graph-autofusion`、`mindspore/dvm`、`cann/ge`、`cann/metadef` |
| 算子库 | `MinghuasLab/flash-attention-npu`、`flashserve/flash-linear-attention-npu`、`cann/ascend-transformer-boost`、`cann/catlass`、`cann/opbase`、`cann/ops-math`、`cann/ops-nn`、`cann/ops-transformer`、`openeuler/kvcache-ops` |
| 算子语言／编译工具链 | `triton-lang/triton-ascend`、`triton-lang/triton`、`cann/asc-devkit`、`Ascend/AscendNPU-IR`、`llvm/torch-mlir`、`Ascend/torch-mlir`、`llvm/llvm-project`、`Ascend/llvm-project` |
| 通信／内存传输 | `cann/hccl`、`cann/hcomm`、`cann/hixl`、`cann/shmem`、`Ascend/memfabric_hybrid` |
| 运行时／驱动／容器 | `cann/runtime`、`cann/driver`、`Ascend/mind-cluster`（`component/ascend-docker-runtime`） |
| 诊断（跨层） | `Ascend/msprobe`、`Ascend/msserviceprofiler`、`cann/oam-tools` |

Stage 1 从 Graphub 根启动，`config.source_repos=v6/code-skill-issue/repos`；forward／reviewer 共享这份只读源码，启动时不 clone／fetch。仓内 docs 保留；NPU 独立文档存放方式待定。

| Agent | 工具 |
| --- | --- |
| forward | `github`、`gitcode`、`search_files`；`web_search`、`web_fetch`；terminal套件；`skills_list`、`skill_view` |
| reviewer | terminal套件；`github`、`gitcode` |
| merge agent | terminal套件 |
| judge | `read_file`，仅读本 query 的题目、各次提交与唯一参考材料 |

terminal套件：`bash`、`terminal_open`、`terminal_send`、`terminal_read`、`terminal_signal`、`terminal_close`、`terminal_list`、`job_output`、`job_list`、`job_kill`（以上 10 个为 DSH 原生工具），以及 `execution_output` 桥接补充。

Stage 1 使用 `vllm-ascend` 工具集，`research` 关闭；val 另行配置。

### Agent 沙箱文件

所有 agent 的 cwd 统一为 `/workspace`，业务文件均在其下；RO 为只读，RW 为可写。`harness/` 对应训练对象；当前默认仅训练 skills。

**Forward**

~~~text
/workspace/
├── repos/                     RO  共享源码仓，保留各仓 .git
├── .agents/
│   └── skills/                RO  输入 K 的技能文件
└── scratch/                   RW  本条 forward 的脚本、下载与中间结果
~~~

技能索引注入 system prompt；`skills_list`、`skill_view` 提供检索与阅读。脚本、附件路径指向 `/workspace/.agents/skills/`，供终端直接使用。

**Reviewer**

~~~text
/workspace/
├── harness/                   RO  训练对象的 Git 工作树根
│   ├── skills/                RW  待更新知识
│   ├── .git/                  RW  查询完整历史、提交修改
│   └── traces/                RO  本 minibatch 的原始轨迹
│       └── batch_i/
│           └── minibatch_j/
│               └── forward_k/
│                   ├── trace.json
│                   └── history.json
├── scratch/                   RW  轨迹分析脚本、中间结果与修改草稿
└── repos/                     RO  共享源码仓，用于核验
~~~

`trace.json` 保留原始 messages；`history.json` 提供同题旧轨迹的 Git 位置。Reviewer 从 $`K_t`$ 去除旧 `traces/` 后开始，加入本 minibatch 的轨迹，修改知识并自行提交。

**Merger**

~~~text
/workspace/
├── harness/                   RO  训练对象的 Git 工作树根
│   ├── skills/                RW  合并知识、调整结构
│   ├── .git/                  RW  完整历史及本 step 全部 reviewer 提交
│   └── traces/                RW  起始为空；合入本 step 的全部原始轨迹
└── scratch/                   RW  临时合并脚本与草稿
~~~

Reviewer／Merger 使用 `git -C harness …` 操作 Git。Merger 同样从 $`K_t`$ 去除旧 `traces/` 后开始，查看并合入各 reviewer 提交。合并校验结果直接反馈到对话。正式历史每个 step 保留一个 commit，保存合并后的知识与本 step 全部原始轨迹。

各 agent 的 `/workspace/scratch/` 独占，跨工具调用保留，不跨 agent 继承、不进入知识 Git。开启 `learn_memory` 时，Reviewer／Merger 的 `harness/` 增加 `memories/MEMORY.md` 并纳入知识提交；Forward 的记忆由组件注入 system prompt。

**Judge** — 仅 `read_file`。

~~~text
/workspace/
└── judge/                     RO  同一 query 的评分材料
    ├── query.json             用户问题
    ├── reference.json         唯一参考答案与证据
    ├── forward-0001.json      一次 forward 的任务要求与全部有序提交
    └── forward-0002.json …
~~~

每份 forward 独立评分，共用同一份参考材料。宿主运行日志、计费与评分映射留在沙箱外。

按 [运行说明](README.md#运行)准备组件环境；在 Graphub 根目录准备题池、固定语义缓存和公开 workspace 后启动：

```sh
PYTHONPATH=v6/code-skill-issue uv run --no-project --python v6/code-skill-issue/.venv/bin/python \
  -m experiments.prepare_repos \
  --catalog v6/code-skill-issue/experiments/source_catalog.json --output v6/code-skill-issue/repos \
  --env-file ../../.env
uv run --no-project --python v6/code-skill-issue/.venv/bin/python python - <<'PY'
import json
from pathlib import Path

stage = json.loads(Path("v6/code-skill-issue/examples/stage1-training.json").read_text())
stage["config"].update(steps=100, semantic_cache="training-semantics.json")
Path("stages.json").write_text(json.dumps([stage], ensure_ascii=False, indent=2) + "\n")
PY
uv run --no-project --python v6/code-skill-issue/.venv/bin/python \
  -m csi.train run-stages tasks.json \
  --stages stages.json --workspace /absolute/path/to/public-workspace \
  --output /absolute/path/to/new-training-run --env-file ../../.env
```

示例 $`T=100`$；`stages.json` 保留 [Stage 1 配置](examples/stage1-training.json)的预算、工具和学习设置。`config` 中的 `batch_size` 为 $`b`$，`reviewers` 为 $`B`$，`candidates` 为 $`k`$。

Web 后端写在初始模板 `config.yaml`：普通抓取用 Firecrawl，`llm_extract=true` 和 `web_search` 用 DeepSeek。

- 密钥：`FIRECRAWL_API_KEY` 用于普通抓取，`NANO_HERMES_WEB_API_KEY` 用于搜索／模型提取。
- 凭据优先级：进程环境 > profile 的 `.env` > 本机组件目录 `.env.web`。主机文件仅补密钥，新训练与续训共用。
- 显式加载：`--env-file`、`--web-env-file PATH`；已有进程变量优先，两文件以前者优先。
- 密钥留在主机；初始模板只含公开配置，显式 profile 使用自身配置。
