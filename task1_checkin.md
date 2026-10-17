# Task1 打卡 · 注意力计算与执行分派

> 微信群昵称: （发到群里前补上）
> GitHub ID: TobinZuo
> 环境：Mac（本机无 NVIDIA GPU，CPU 机制测试）
> 系统：macOS
> AI 占比 vs 人工占比：80:20
> 任务项编号 4.1 + 4.2 + 4.3（Part 02 · 04 MHA / GQA，Part 02 · 20 FlashAttention / SDPA，Part 02 · INF-EXECUTION-DISPATCH）
> 当前卡点：本机没有 NVIDIA GPU。4.2 的耗时和峰值显存、4.3 的 CUDA kernel 名都还没有本机实测。下面给出实验口径，以及没有实测时怎么区分三种情况。

学习来源：DataWhale 社区《大模型算法与系统教程》https://github.com/datawhalechina/llm-algo-leetcode ，对应 Part 02 · 04、Part 02 · 20 与 Part 02 · INF-EXECUTION-DISPATCH。

## 1. MHA 与 GQA 的 KV cache 有什么区别？

MHA 里每个 Query 头都有自己的 K、V 头，头数相同。GQA 仍保留全部 Query 头，但让一组 Query 头共用一对 K、V。本题是 32 个 Query 头对 8 个 KV 头，每 4 个 Query 头共享一组 K、V。

Cache 里只保存投影出来的原始 KV，形状是 `[B, H_kv, S, D]`。前向时如果要和 Query 头对齐，只在当次计算里把 KV 头临时重复，重复结果不写回 cache。所以 GQA 省下的是长期驻留的历史状态，而不是把计算时的临时张量也按 MHA 的头数存下来。

单层 K、V 占用：

`字节数 = 2 × B × S × H_kv × D × 每元素字节数`

前面的 2 是 K 和 V 各一份。FP16 每个元素 2 字节。代入 B=1、S=4096、D=128：

- MHA，H_kv=32：`2 × 1 × 4096 × 32 × 128 × 2 = 67,108,864` 字节 = 64 MiB
- GQA，H_kv=8：`2 × 1 × 4096 × 8 × 128 × 2 = 16,777,216` 字节 = 16 MiB

GQA 这一层的 KV cache 是 MHA 的 1/4，少 48 MiB。这里只算 K、V 张量本身，不含权重、激活和分配器额外占用。1 MiB 按 2^20 字节计。

解码时每个新 token 的 Query 长度是 1，算力很小，时间主要花在从显存读出全部历史 K、V。GQA 把要读的 KV 头数从 32 降到 8，同一层每次解码从 HBM 搬进计算单元的 KV 字节数也降到原来的 1/4。写入新 token 的 K、V 同样按 8 个头而不是 32 个头增长。这就是它能减轻解码访存的原因。收益不会自动变成 4 倍吞吐：如果实现里先把 8 个头显式复制成 32 个头再读，临时流量会把一部分节省吃回去；其它算子、采样和调度也可能成为新的瓶颈。

## 2. FlashAttention 为什么能够降低显存占用？

普通 Attention 先算出完整分数 `S = QK^T / sqrt(d)`，再算出 `P = softmax(S)`，最后 `O = PV`。自注意力里 S 和 P 的形状是 `[B, H, S, S]`，序列一长就按平方增长，而且每一步都要把这块中间结果写回显存再读出来。

FlashAttention 仍计算同一种稠密注意力，但按 token 块处理：一块 Q 配上一块 K、V，局部分数留在片上存储里更新。HBM 上保留的是 Q、K、V、输出，以及每行的最大值和指数和，不再驻留完整的 S、P。

三件事各管一块：

- 分块：片上存储放不下整段序列的分数矩阵。切块后，当前块的工作集由块大小决定，不随总长度平方变大。
- 在线 Softmax：一行的归一化要用到该行所有可见 key，不能把各块 softmax 之后直接相加。实现上为每行记下当前最大值、指数和，以及按这个分母归一化后的输出。新块到来时，用新旧最大值的指数差把历史统计量和历史输出重标定，再并入当前块。这样不用保存整行分数，数值上也避免直接对很大的分数做 exp。
- 算子融合：局部分数、mask、指数、乘 V 放在同一次遍历里完成，中间块不写回 HBM，也少了多次 kernel 启动。

数学结果：同样的 Q、K、V、缩放和 mask 下，实数运算与标准稠密 Attention 相同。浮点里累加顺序不同，会有舍入差，比较时用误差容限，不要求逐位相同。

计算复杂度：每个 query 仍要看所有允许的 key，算术量仍是 `O(B · H · S² · D)`，固定 B、H、D 时仍是 `O(S²)`。变少的是 HBM 读写和 N² 中间矩阵的驻留，不是乘法次数。它没有把稠密注意力变成线性注意力。

和上一问的关系：GQA 减少的是跨解码步留下来的 KV；FlashAttention 减少的是这一次注意力计算的临时矩阵和读写。两者解决的不是同一块显存。

## 代码测试

在 macOS、CPU 上跑通教程机制测试：

- Part 02 · 04：MHA / GQA / MQA 输出形状、因果 mask、KV cache 解码与完整前向末 token 一致、非法 head 配置被拒绝。
- Part 02 · 20：分块在线 softmax 与完整注意力在非因果、因果、float64 下数值对齐；大分数输出保持有限；固定块大小下局部分数元素数不随序列长度平方增长。

测试输出见 `task1_test_output.txt`。

![机制测试通过](https://raw.githubusercontent.com/TobinZuo/llm-algo-leetcode-study/master/task1_test_pass.png)

## 4.2 公平的 Attention 性能对比

### 1. 怎样才算公平

两条路径必须吃同一份 Q、K、V：同一个 device、dtype、形状、因果 mask 和随机种子。只改注意力怎么执行，显式物化分数矩阵是基线，SDPA 或 FlashAttention 内核是候选。batch、头数、头维度、序列长度档位事先定死。这里沿教程的机制实验口径：batch=1、8 个头、头维度 64、长度 512/1024/2048/4096，精度用 GPU 实际支持的 BF16 或 FP16。

比速度之前先看数值。低精度允许舍入差，用固定的 atol、rtol 看输出是否接近，再看最大绝对误差和相对 L2。没通过这一关的结果不拿来谈加速。

### 2. 预热、同步、耗时和峰值显存

正式计时前先空跑若干次，把 kernel 加载、缓存和频率爬升排除掉。GPU 调用是异步的，CPU 侧的 `time.time()` 会在计算没做完时就返回。用 CUDA Event 包住每次调用，结束事件 `synchronize` 之后再读毫秒数。重复多次，记中位数和 P95，不拿单次最小值当结论。

峰值显存每次测量前 `reset_peak_memory_stats`。看的是这一路自己的 `max_memory_allocated` 和 `max_memory_reserved`，两条路分开清零，避免前一路的峰值算到后一路头上。Profiler 再记实际算子名，用来确认 SDPA 走到了哪一个内核。

本机是 Mac，没有 CUDA，所以上面这组延迟和峰值显存没有实测数字。

### 3. 算子变快，请求为什么不一定同等变快

一次请求的时间不只有注意力。预填充在序列很长时，分数矩阵的读写确实可能占大头，这时算子加速比较容易反映到首 token 时间上。解码时每个新词的 Query 长度是 1，注意力主要在读历史 KV；旁边还有前馈、词表投影、采样和调度。注意力内核即使快几倍，这些部分不动，整次请求就快不了那么多。

序列很短时，分数矩阵本来就不大，分块内核的启动开销可能把收益吃掉。显存省下来以后，服务端可能加大 batch，吞吐上去了，单条请求的延迟却不一定按同样比例下降。

## 4.3 SDPA 不等于已经跑了 FlashAttention

### 1. 调用 SDPA 只是进入分派

`scaled_dot_product_attention` 按设备、dtype、头维度、mask、dropout 和当前 PyTorch 构建，在几条后端里选一条。常见的有 FlashAttention 内核、memory-efficient 内核、cuDNN，以及显式的 math 实现。Flash 内核通常要求 CUDA、FP16 或 BF16，以及它支持的头维度。FP32、某些 mask、CPU，都会走到别的实现。所以代码里写了 SDPA，只能说明调用了这个接口。

### 2. 性能没提升时看什么

先分清三件事，证据不要混用。

后端不可用：机器没有 CUDA，或者当前 dtype、头维度、GPU 架构不在 Flash 内核的支持范围里，或者这版 PyTorch 根本没编进该内核。`flash_sdp_enabled()` 一类能力开关是 false，日志里会有不支持的提示。Profiler 里看不到 flash 的 kernel，看到的是 matmul 和 softmax。

执行路径未命中：能力开关是开的，但这一次调用没走进去。dropout、自定义 mask、被 `torch.compile` 拆碎的图，都可能让同一次 SDPA 退回 math 或 memory-efficient。这时要看这次运行的算子名，而不是看开关。命中 Flash 时，记录里应出现 `_scaled_dot_product_flash_attention`、`_flash_attention_forward` 或对应的 `fmha` kernel。没有这些名字，就是没走到。

已命中但收益有限：算子名已经是 Flash 内核，输出也和显式注意力对得上，可是延迟差不多。短序列、小 batch，或者时间主要花在前馈和读 KV 上，都会这样。加速比接近 1，同时 kernel 名已经命中，就属于这一类，而不是没走到。

CPU 上可以先验证分派规则本身：长模式优先匹配，dtype、设备、序列长度、动态形状各自成为拒绝原因，全部后端都不满足时退回 eager，并留下每个后端的拒绝理由。这部分测试已通过，见下方输出。GPU 上的 kernel 名仍然要等有 CUDA 的环境再采。
