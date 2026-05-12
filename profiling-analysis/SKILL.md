---
name: profiling-analysis
description: >
  分析已解析的 NPU profiling 数据（torch_npu.profiler.analyse() 输出）。
  本 skill 专门用于 INTERPRETING（解读）和 UNDERSTANDING（理解）已经采集并
  解析好的 profiling 数据。它不负责采集或打包——采集请使用 vllm-profiler skill。
  当用户已有解析好的 profiling 数据（ascend_pytorch_profiler.db），想要了解
  NPU 流分布、算子放置、特性验证或任何 trace 中可见的行为时，使用此 skill。
  也适用于用户请求"分析"或"检查"已有 profiling 数据时。
---

# Profiling 数据分析

分析已解析的 NPU profiling 数据（`ascend_pytorch_profiler*.db`），帮助用户
理解流分布、算子行为、特性验证，或回答与 NPU trace 相关的任何问题。

## 核心原则：先确认再下结论

Profiling 分析充满陷阱：错误的算子假设、错认的流角色、不正确的时间偏移、
混淆的概念。在下结论之前，**始终与用户确认你的理解**。

**最重要的规则：不确定就问。** 先问清楚再继续分析，可以避免数小时的无效工作。

## 用户交互方式

使用 `question` 工具（而非自由文本）向用户寻求确认。尽可能提供结构化选项。

---

## 分析流程

```
[Phase 1: 理解需求]
  ├── 明确用户的问题和分析目标
  ├── 确认服务配置（哪些特性开启）
  └── 与用户确认

[Phase 2: 代码 → 算子映射]
  ├── 确定相关的代码路径和算子
  ├── 列出你的理解
  └── 查数据前先与用户确认

[Phase 3: 全量扫描]
  ├── 枚举所有流及其算子
  ├── 报告各流的时间窗口
  └── 让用户判断哪些流重要

[Phase 4: 定向钻取]
  ├── 基于用户指引运行定向查询
  ├── 用时序/重叠分析验证
  └── 逐步展示发现

[Phase 5: 报告]
  ├── 用表格清晰展示发现
  └── 让用户决定下一步
```

---

## Phase 1: 理解需求

查数据前，先理解用户需要什么。

### 1.1 明确分析目标

使用 `question` 工具缩小范围：

```python
question(questions=[{
    "header": "分析目标",
    "question": "你想从 profiling 数据中了解什么？",
    "options": [
        {"label": "流分布概览", "description": "查看所有 NPU 流上有哪些算子和任务分布"},
        {"label": "特定特性验证", "description": "验证某个功能（如双流并行、MTP 等）是否生效"},
        {"label": "特定算子定位", "description": "查看某个算子/代码路径在哪些流上执行"},
        {"label": "性能热点分析", "description": "找出耗时最长的算子和热点路径"},
        {"label": "其他", "description": "我来描述具体需求"}
    ]
}])
```

根据用户选择确定分析重点。

### 1.2 确认服务配置

检查启动脚本中影响 profiling 行为的配置参数。重点关注：
- `--additional-config` 中的特性开关（如 `dsa_dual_stream`、
  `multistream_overlap_shared_expert` 等）
- `--compilation-config` 选项（如 `cudagraph_mode`）
- `--profiler-config` 设置

展示发现并用 `question` 工具确认：

```python
question(questions=[{
    "header": "确认配置",
    "question": "以上配置是否正确？",
    "options": [
        {"label": "配置正确", "description": "按此配置继续分析"},
        {"label": "配置有误", "description": "我来提供正确的配置信息"}
    ]
}])
```

配置上下文是正确解读 profiling 结果的关键。

---

## Phase 2: 代码 → 算子映射

这是最常见的错误来源。**不要猜测** NPU 算子与代码路径的对应关系。

### 2.1 确定相关代码路径

询问用户或阅读代码，找到与分析目标相关的代码路径。

### 2.2 呈现算子映射并确认

列出你的理解，用 `question` 工具确认：

```python
question(questions=[{
    "header": "确认算子映射",
    "question": "以下算子映射关系是否正确？",
    "options": [
        {"label": "映射正确", "description": "继续分析"},
        {"label": "部分不对", "description": "我来纠正"}
    ]
}])
```

示例呈现（根据实际分析目标调整）：

```
我的算子映射理解：
  weights_proj    → MatMulV2 (FP16 矩阵乘)
  kv_compressor   → Compressor_xxx
  kv_scatter      → ScatterNdUpdateV2_xxx
  q_quant         → DynamicQuant_xxx
  QLI             → QuantLightningIndexer_xxx
  QKV 投影        → QuantBatchMatmulV3_xxx
```

**不要假设**算子到代码的映射。同一个算子名在不同上下文中可能代表不同的
功能。通过 `question` 工具先问清楚。

### 2.3 确定目标流

如果用户已经知道要看哪些流，直接使用。否则从全量扫描开始（Phase 3）。

---

## Phase 3: 全量扫描

### 3.1 全流算子查询

查询所有流的算子分布。具体查询哪些算子取决于分析目标。

通用查询模板：

```sql
SELECT t.streamId,
  COUNT(*) as total_tasks,
  SUM(CASE WHEN s.value LIKE '%<算子1>%' THEN 1 ELSE 0 END) as op1,
  SUM(CASE WHEN s.value LIKE '%<算子2>%' THEN 1 ELSE 0 END) as op2,
  ...
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId = cti.globalTaskId
JOIN STRING_IDS s ON cti.name = s.id
GROUP BY t.streamId ORDER BY t.streamId
```

### 3.2 报告时间窗口

```sql
SELECT streamId, MIN(startNs)/1e6, MAX(startNs)/1e6,
       (MAX(startNs)-MIN(startNs))/1e6 as dur_ms,
       COUNT(*) as tasks
FROM TASK GROUP BY streamId ORDER BY streamId
```

### 3.3 让用户判断

展示概览后，用 `question` 工具让用户判断：

```python
question(questions=[{
    "header": "确认流角色",
    "question": "根据以上数据，你能判断各流的角色吗？",
    "options": [
        {"label": "我已判断，继续钻取", "description": "指定要分析的流和算子"},
        {"label": "不清楚，需要更多信息", "description": "我来补充分析方向"},
        {"label": "流分布异常", "description": "发现问题需要讨论"}
    ]
}])
```

没有用户确认，不要自行下结论。

---

## Phase 4: 定向钻取

根据用户在 Phase 3 中指引的方向，执行特定分析。

### 4.1 时序重叠分析

检查两条流是否并行执行：

```sql
SELECT COUNT(*) FROM (
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <A> AND <条件_A>
  INTERSECT
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <B> AND <条件_B>
)
```

### 4.2 特定时间窗口的详细时序

```sql
SELECT startNs/1e6, streamId,
  CASE WHEN s.value LIKE '%X%' THEN '标签_x'
       WHEN s.value LIKE '%Y%' THEN '标签_y'
       ELSE '其他' END as tag
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId=cti.globalTaskId
JOIN STRING_IDS s ON cti.name=s.id
WHERE streamId IN (<流列表>)
AND startNs BETWEEN <t0> AND <t1>
ORDER BY startNs
```

### 4.3 时间偏移量确认

当用户提到相对时间（如"从 694ms 开始"），必须用 `question` 工具
确认参考点：

```python
question(questions=[{
    "header": "确认时间偏移",
    "question": "你说的 {N}ms 是从哪个起点算的？",
    "options": [
        {"label": "从流{X}起点", "description": "流{X}的 startNs + {N}ms"},
        {"label": "从流{Y}起点", "description": "流{Y}的 startNs + {N}ms"},
        {"label": "绝对时间戳", "description": "我提供具体的绝对时间戳"}
    ]
}])
```

不同流的 start 时间可能相差 13ms 以上，选错参考点会导致完全错误的分析。

### 4.4 进一步迭代

展示钻取结果后，用 `question` 工具确认下一步：

```python
question(questions=[{
    "header": "下一步",
    "question": "需要继续深入分析吗？",
    "options": [
        {"label": "继续分析其他方面", "description": "指定新的分析方向"},
        {"label": "已完成，生成报告", "description": "总结发现"},
        {"label": "打包数据", "description": "将解析后的数据打包"}
    ]
}])
```

---

## Phase 5: 报告

展示分析结果时：

1. **用清晰的对比表格**展示各流的算子计数
2. **注明时间窗口**和流间重叠时长
3. **明确陈述结论** — 数据说明了什么，意味着什么
4. **承认不确定性** — 例如"profiler 将所有算子合并到同一条流上，但这可能是图编译的产物"
5. **用 `question` 确认**分析结果是否满足用户需求

```python
question(questions=[{
    "header": "报告确认",
    "question": "以上分析是否回答了你的问题？",
    "options": [
        {"label": "已满足需求", "description": "分析完成"},
        {"label": "还需补充", "description": "说明需要补充的内容"}
    ]
}])
```

---

## 常见陷阱

### ❌ 错误：假设算子到代码的映射
务必与用户确认映射关系。同一个算子名在不同模型/配置下意义可能完全不同。

### ❌ 错误：忽略非明显的流
任务数少的流可能反而是关键流。全量扫描时检查**所有**流。

### ❌ 错误：从错误基线计算相对偏移
不同流的 start 时间不同。计算偏移前先和用户确认参考点。

### ❌ 错误：从缺失数据得出否定结论
"profiler 中看不到"不等于"没生效"。ACL 图运行时可能合并流或优化掉中间操作。

### ❌ 错误：混淆图实例与并行流
两条流包含相同算子集合时，通常是不通的 batch size 的 captured graph，
而不是功能上独立的并行流。

### ❌ 错误：未经用户方向就钻取数据
先确认分析目标和算子映射，再执行 SQL 查询。避免在无关方向上浪费时间。
