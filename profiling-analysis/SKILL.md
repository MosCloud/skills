---
name: profiling-analysis
description: >
  Analyze parsed NPU profiling data (torch_npu.profiler.analyse() output) from
  vLLM Ascend inference services. This skill is specifically for INTERPRETING
  and UNDERSTANDING profiling data that has already been collected and parsed.
  It is NOT for collecting or packaging profiling data — use the vllm-profiler
  skill for that. Use this skill whenever the user has parsed profiling data
  (ascend_pytorch_profiler.db) and wants to understand stream distribution,
  operator placement, dual-stream parallelism, or any behavior visible in the
  NPU trace. Also use this skill when the user asks to "analyze" or "check"
  something in profiling data they already have.
---

# Profiling Data Analysis

Analyze parsed NPU profiling data from `ascend_pytorch_profiler*.db` to help
users understand stream distribution, operator behavior, feature verification,
or any question they have about their NPU trace data.

## Core Principle: Confirm Before Conclude

Profiling analysis is full of traps: wrong operator assumptions, misidentified
stream roles, incorrect time offsets, conflated concepts. Before drawing any
conclusion, ALWAYS confirm your understanding with the user.

**The single most important rule: if you are not sure about something, ask
before analyzing further.** It saves hours of wasted work.

## How to Interact with the User

Use the `question` tool (not free-text) to ask for input and confirmation.
Provide structured options when possible.

---

## Analysis Workflow

```
[Phase 1: Understand Requirements]
  ├── Clarify the user's question and analysis goal
  ├── Understand the service config (which features are on/off)
  └── Confirm with user

[Phase 2: Know the Code → Operator Mapping]
  ├── Identify which code paths and operators are relevant
  ├── List your understanding of the mapping
  └── CONFIRM WITH THE USER before querying data

[Phase 3: Survey the Data]
  ├── Enumerate all streams and their operators
  ├── Report time windows per stream
  └── Let the user identify what matters

[Phase 4: Targeted Drill-down]
  ├── Based on user guidance, run targeted queries
  ├── Verify specific conclusions with timing/overlap analysis
  └── Present findings incrementally

[Phase 5: Report]
  ├── Summarize findings in clear table format
  └── Let the user decide next steps
```

---

## Phase 1: Understand Requirements

Before touching any SQL query, understand what the user needs.

### 1.1 Clarify the analysis goal

Use the `question` tool to narrow down:

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

Based on the user's choice, proceed with the appropriate focus.

### 1.2 Confirm the service configuration

Check the startup script for configuration parameters that affect profiling
behavior. Look for:
- Feature toggles in `--additional-config` (e.g., `dsa_dual_stream`,
  `multistream_overlap_shared_expert`, etc.)
- `--compilation-config` options (e.g., `cudagraph_mode`)
- `--profiler-config` settings

Present findings and use `question` tool to confirm:

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

The config context is critical for interpreting profiling results correctly.

---

## Phase 2: Know the Code → Operator Mapping

This is the MOST COMMON SOURCE OF ERROR. Do NOT guess which NPU operator
corresponds to which code path.

### 2.1 Identify relevant code paths

Ask the user or read the code to find which code paths are relevant to
their analysis goal.

### 2.2 Present operator mapping for confirmation

List your understanding and use the `question` tool:

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

Example presentation (adjust based on actual analysis goal):

```
My understanding of the operator mappings:
  weights_proj    → MatMulV2 (FP16 matmul)
  kv_compressor   → Compressor_xxx
  kv_scatter      → ScatterNdUpdateV2_xxx
  q_quant         → DynamicQuant_xxx
  QLI             → QuantLightningIndexer_xxx
  QKV projection  → QuantBatchMatmulV3_xxx
```

DO NOT assume operator-to-code mappings. The same operator name can serve
different purposes in different contexts. ASK FIRST via the `question` tool.

### 2.3 Identify relevant streams

If the user already knows which streams to look at, use that. If not,
start with a complete stream survey (Phase 3).

---

## Phase 3: Survey the Data

### 3.1 Full stream survey

Query ALL streams for operator distribution. The specific operators to
check depend on the analysis goal from Phase 1.

General pattern:

```sql
SELECT t.streamId,
  COUNT(*) as total_tasks,
  SUM(CASE WHEN s.value LIKE '%<op1>%' THEN 1 ELSE 0 END) as op1,
  SUM(CASE WHEN s.value LIKE '%<op2>%' THEN 1 ELSE 0 END) as op2,
  ...
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId = cti.globalTaskId
JOIN STRING_IDS s ON cti.name = s.id
GROUP BY t.streamId ORDER BY t.streamId
```

### 3.2 Report time windows

```sql
SELECT streamId, MIN(startNs)/1e6, MAX(startNs)/1e6,
       (MAX(startNs)-MIN(startNs))/1e6 as dur_ms,
       COUNT(*) as tasks
FROM TASK GROUP BY streamId ORDER BY streamId
```

### 3.3 Use `question` tool to get user's interpretation

After showing the overview, let the user interpret:

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

Do NOT jump to conclusions about stream roles without the user's input.

---

## Phase 4: Targeted Drill-down

Based on the user's guidance from Phase 3, run specific analyses.

### 4.1 Timing overlap analysis

To check if two streams execute in parallel:

```sql
SELECT COUNT(*) FROM (
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <A> AND <condition_A>
  INTERSECT
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <B> AND <condition_B>
)
```

### 4.2 Detailed timeline for a time window

```sql
SELECT startNs/1e6, streamId,
  CASE WHEN s.value LIKE '%X%' THEN 'label_x'
       WHEN s.value LIKE '%Y%' THEN 'label_y'
       ELSE 'other' END as tag
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId=cti.globalTaskId
JOIN STRING_IDS s ON cti.name=s.id
WHERE streamId IN (<streams>)
AND startNs BETWEEN <t0> AND <t1>
ORDER BY startNs
```

### 4.3 Time offset clarity check

When the user mentions a relative time (e.g., "from 694ms"), ALWAYS
clarify the reference point using the `question` tool:

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

### 4.4 Further iteration

After showing drill-down results, use the `question` tool:

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

## Phase 5: Report

When presenting findings:

1. **Use clear comparison tables** showing operator counts per stream
2. **Note the total time window** and overlap duration for each stream
3. **State the conclusion clearly** — what the data says and what it means
4. **Acknowledge uncertainty** — e.g., "the profiler sees all ops on one
   stream, but this may be a graph compilation artifact"
5. **Use `question` to confirm** if the findings address the user's need

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

## Common Pitfalls to Avoid

### ❌ Wrong: Assuming operator-to-code mappings
Always confirm mappings with the user. The same operator name can have
different meanings in different models/configurations.

### ❌ Wrong: Ignoring non-obvious streams
A stream with few tasks may still be the critical one. Check ALL streams
when doing initial survey.

### ❌ Wrong: Computing relative offset from the wrong baseline
Different streams have different start times. Clarify the reference point
before computing time offsets.

### ❌ Wrong: Drawing negative conclusions from missing data
"Not visible in profiling" does not mean "not working." The ACL graph
runtime may merge streams or optimize away intermediate operations.

### ❌ Wrong: Confusing graph instances with parallel streams
Two streams each with the same set of operators are likely different
captured graph sizes (different batch sizes), not separate functional
streams for parallel execution.

### ❌ Wrong: Diving into data without user direction
Always clarify the analysis goal and confirm mappings before running
SQL queries. This prevents wasted effort on irrelevant analysis.
