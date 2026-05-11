---
name: vllm-profiler
description: >
  采集并分析 vLLM Ascend 推理服务的 NPU profiling 数据。
  当用户提到 profiling、性能分析、trace、NPU 流分析、性能采集、双流并行验证，
  或需要确认某个特性（如双流并行）在图模式中是否生效时，使用此 skill。
---

# vLLM Ascend Profiling 采集器

从正在运行的 vLLM Ascend 服务中采集 NPU profiling trace，使用
`torch_npu.profiler.analyse()` 解析，通过 SQLite 数据库分析流和算子行为，
并将解析结果打包以便分享。

## 整体流程

```
[步骤1: 环境检查] → [步骤2: 配置请求] → [步骤3: 确认并预热]
→ [步骤4: 采集数据] → [步骤5: 解析数据] → [步骤6: 分析] → [步骤7: 打包]
```

每步都需要和用户确认后才能继续，**不能跳过或自行假设**。

---

## 步骤1: 环境检查

采集前检查三项：

### 1.1 服务存活

```bash
curl -s http://<server_url>/v1/models
```

如果失败，告知用户并停止，请用户先启动服务。

默认 `server_url`: `http://127.0.0.1:7000`，允许用户覆盖。

### 1.2 Profiling 配置

读取服务启动脚本（询问用户路径，或检查常见位置如 `start_test.sh`）。
查找 `--profiler-config` 参数。

确认 profiler 类型为 `"torch"`，记录 `torch_profiler_dir` 的值
（通常为 `"./vllm_prof"`），profiling 输出将落在此目录。

如果没有 `--profiler-config`，警告用户 profiling 不会产生数据。

### 1.3 特性开关

如果用户要验证特定特性（如双流并行），检查 `--additional-config`：

```bash
grep "additional-config" <启动脚本>
```

输出当前状态：`"dsa_dual_stream": true` 或 `"dsa_dual_stream": false`。

---

## 步骤2: 配置推理请求

向用户展示将在 profiling 期间执行的推理请求。

### 2.1 默认请求模板

```json
{
  "model": "deepseek_v4",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "max_tokens": <N>,
  "temperature": 0
}
```

### 2.2 可自定义参数

向用户确认：
- **Model**: 默认 `deepseek_v4`
- **Prompt**: 默认 `"你好，请简单的介绍你自己。"`
- **Max tokens**: 默认 `128`
- **其他参数**（如 headers、stream 等）

打印最终 curl 命令，等待用户确认。

---

## 步骤3: 确认并预热

### 3.1 确认采集计划

总结展示：

```
采集计划:
  服务地址:    http://127.0.0.1:7000
  模型:        deepseek_v4
  请求:        "你好，请简单的介绍你自己。" (max_tokens=128)
  输出目录:    ./vllm_prof/
  特性状态:    dsa_dual_stream=true
```

请用户确认。

### 3.2 预热（可选）

询问用户：**"是否需要在正式采集前先 warmup？"**

如果需要，在**不启动 profiler** 的情况下执行同一条推理请求 1~3 次：

```
预热: curl <同一条请求>  →  丢弃输出
预热: curl <同一条请求>  →  丢弃输出
预热: curl <同一条请求>  →  丢弃输出
```

如果不需要，直接进入步骤4。

---

## 步骤4: 采集 Profiling 数据

### 4.1 清理旧数据

```bash
rm -rf <profiler_dir>/*
```

### 4.2 启动 profiling

```bash
curl -s -X POST http://<server>/start_profile -H "Content-Type: application/json"
```

检查 HTTP 200。如果失败，告知用户 profiling API 可能不可用。

### 4.3 执行推理请求

执行步骤2中确认的 curl 命令。截断输出内容展示，确认请求正常完成。

### 4.4 停止 profiling

```bash
curl -s -X POST http://<server>/stop_profile -H "Content-Type: application/json"
```

### 4.5 定位数据

在 `<profiler_dir>/` 下查找各 rank 的目录：

```bash
ls <profiler_dir>/dp0_pp0_tp0_dcp0_ep0_rank0_*
```

每个 rank 有两个目录：
- `*_ascend_pt/` 含 `FRAMEWORK/` + `PROF_*/` 子目录（原始 NPU trace）
- `*_ascend_pt/` 仅含 `profiler_info_*.json`（元数据）

含有 `FRAMEWORK` 子目录的才是实际 profiling 数据目录。

---

## 步骤5: 解析数据

### 5.1 询问解析范围

询问用户：**"是否需要解析所有 rank 的数据，还是只解析指定 rank？"**

- **所有 rank**: 遍历每个含 `FRAMEWORK/` 的 rank 目录
- **指定 rank**: 默认 rank0（对流分析已足够）

### 5.2 执行 analyse()

对每个选中的 rank 目录：

```python
from torch_npu.profiler.profiler import analyse

analyse("<profiler_dir>/<rank_dir>")
```

这会在 rank 目录下创建 `ASCEND_PROFILER_OUTPUT/`，包含：
- `ascend_pytorch_profiler.db`（或 `ascend_pytorch_profiler_0.db`）— 主 trace 数据库
- `trace_view.json` — Chrome trace 可视化数据
- `operator_details.csv` — 逐算子明细
- `kernel_details.csv` — 逐 kernel 明细
- `analysis.db` — 通信分析
- `op_statistic.csv` — 算子统计

注意：`analyse()` 每个 rank 约需 3~5 分钟。

### 5.3 验证解析完成

检查 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db` 存在且非空。

---

## 步骤6: 分析数据

解析完成后，询问用户需要哪些分析：

### 可选分析项：

1. **流分布** — NPU 流有哪些，各自的 task 数量、wall time、compute time。
   这是验证双流并行的主要手段。
   ```sql
   SELECT streamId, COUNT(*) FROM TASK GROUP BY streamId ORDER BY COUNT(*) DESC
   ```

2. **DSA 算子分布** — QuantBatchMatmul、DynamicQuant、ScatterNdUpdate、
   Rotary、QLI 在各流上的计数。可判断 weights_proj 是否在独立子流上执行。
   ```sql
   SELECT t.streamId, COUNT(*) FROM TASK t
   JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId = cti.globalTaskId
   JOIN STRING_IDS s ON cti.name = s.id
   WHERE s.value LIKE '%<operator>%' GROUP BY t.streamId
   ```

3. **流同步** — 统计 `aclrtStreamWaitEvent` 调用次数，判断跨流同步是否发生。

4. **自定义分析** — 用户指定 SQL 查询或分析目标。

分析结果以清晰表格呈现。

### 双流并行验证对照表

| 场景 | 预期 |
|---|---|
| c4 双流开启 | DSA 算子分布在主流+子流，约 2:1 比例 |
| c4 双流关闭 | 所有 DSA 算子仅在主流图中，无独立子流 |
| c128 层 | 无索引器算子 |

---

## 步骤7: 打包

### 7.1 询问输出位置

**"将解析后的数据打包到哪个目录下？"** 默认目录：`./prof_docs/`

### 7.2 打包（仅 ASCEND_PROFILER_OUTPUT）

**只打包解析后的数据**，不包含原始 FRAMEWORK/PROF 数据：

```bash
tar czf <output_dir>/vllm_prof_<描述>_rank<N>.tar.gz \
  -C <profiler_dir>/<rank_dir> \
  ASCEND_PROFILER_OUTPUT
```

### 7.3 输出

打印文件路径和大小：

```
已打包: <output_dir>/vllm_prof_*.tar.gz (XXX MB)
```

---

## 默认参数配置

| 参数 | 默认值 |
|---|---|
| 服务地址 | `http://127.0.0.1:7000` |
| Profiler 输出目录 | `./vllm_prof/`（来自 `torch_profiler_dir`） |
| 模型 | `deepseek_v4` |
| Prompt | `"你好，请简单的介绍你自己。"` |
| Max tokens | `128` |
| Temperature | `0` |
| 默认 rank | `rank0` |
| 打包输出目录 | `./prof_docs/` |

用户可在每一步覆盖以上默认值。

---

## 常用分析模式

### 验证双流并行已生效

```
预期: DSA 算子（QBM, DQ, SND, Rotary, QLI）分布在至少2条流上，约 2:1 比例
关键流: streamId=X（主流）+ streamId=Y（子流）
同步: aclrtStreamWaitEvent 计数 > 0
```

### 验证双流并行已关闭

```
预期: 所有 DSA 算子在 1~2 条主流图上，无承载 DSA 算子的独立子流
关键流: streamId=A + streamId=B（均为主流，不同图尺寸）
同步: aclrtStreamWaitEvent 可能为 multistream_overlap_shared_expert 产生
```

### 查看算子热点

查询 `operator_details.csv`，按总耗时排序找出热点算子。
