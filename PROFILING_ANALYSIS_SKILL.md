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

Analyze parsed NPU profiling data from `ascend_pytorch_profiler*.db` to
understand stream distribution, operator behavior, feature verification
(e.g., dual-stream parallelism), and performance characteristics.

## Core Principle: Confirm Before Conclude

Profiling analysis is full of traps: wrong operator assumptions, misidentified
stream roles, incorrect time offsets, conflated concepts. Before drawing any
conclusion, ALWAYS confirm your understanding with the user.

**The single most important rule: if you are not sure about something, ask
before analyzing further.** It saves hours of wasted work.

---

## Analysis Workflow

```
[Phase 1: Understand Requirements]
  ├── Clarify the user's question
  └── Understand the service config (which features are on/off)

[Phase 2: Know the Code → Operator Mapping]
  ├── Identify which code paths and operators are relevant
  ├── List your understanding of the mapping
  └── CONFIRM WITH THE USER before querying data

[Phase 3: Survey the Data]
  ├── Enumerate all streams and their operators
  ├── Identify each stream's role
  └── Present the overview, let the user correct

[Phase 4: Targeted Drill-down]
  ├── Based on user confirmation, run targeted queries
  ├── Verify specific conclusions with timing analysis
  └── Present findings incrementally

[Phase 5: Package & Report]
  ├── Summarize findings in clear table format
  └── Let the user decide next steps
```

---

## Phase 1: Understand Requirements

Before touching any SQL query, understand:

### 1.1 What is the user trying to verify?

Common analysis goals:
- **Dual-stream parallelism**: Is weights_proj running on a separate stream
  from compressor/scatter/q_quant/QLI?
- **Stream distribution**: What are all the NPU streams and what runs on each?
- **Operator placement**: Where does a specific operator execute?
- **Performance**: What are the hotspots?
- **Feature verification**: Confirm a feature is working as designed.

### 1.2 What config was the service running with?

Check the startup script for:
- `dsa_dual_stream`: true or false
- `multistream_overlap_shared_expert`: true or false
- `cudagraph_mode`: FULL_DECODE_ONLY or other
- capture batch sizes

These directly affect stream assignment in the profiler.

**Always present this config info to the user and confirm** before proceeding.
Different configs produce radically different stream patterns.

---

## Phase 2: Know the Code → Operator Mapping

This is the MOST COMMON SOURCE OF ERROR. Do NOT guess which NPU operator
corresponds to which code path. Instead:

### 2.1 Identify relevant code paths

Ask the user or read the code to find which code path is being analyzed:
- e.g., `_kv_compressor_forward` → involves `compressor`, `scatter`,
  `weights_proj`, `q_quant`, `QLI`
- e.g., `_indexer_qkv_prepare` → involves `rotary`, `rmsnorm`

### 2.2 Map code to operator names

**List your understanding explicitly and get confirmation:**

```
My understanding of the operator mappings:
  weights_proj    → MatMulV2 (FP16 matmul)
  kv_compressor   → Compressor_xxx
  kv_scatter      → ScatterNdUpdateV2_xxx
  q_quant         → DynamicQuant_xxx
  QLI             → QuantLightningIndexer_xxx
  QKV projection  → QuantBatchMatmulV3_xxx

Is this correct?
```

DO NOT assume. The `QuantBatchMatmulV3` operator could be QKV attention or
weights_proj depending on the model. ASK FIRST.

### 2.3 Identify relevant stream IDs

If the user already knows which streams to look at, use that. If not, start
with a complete stream survey (Phase 3).

---

## Phase 3: Survey the Data

### 3.1 Basic survey query

Query ALL streams for DSA-relevant operators:

```sql
SELECT t.streamId,
  SUM(CASE WHEN s.value LIKE '%QuantBatchMatmul%' THEN 1 ELSE 0 END) as QBM,
  SUM(CASE WHEN s.value LIKE '%MatMulV2%' THEN 1 ELSE 0 END) as MatMulV2,
  SUM(CASE WHEN s.value LIKE '%ScatterNdUpdate%' THEN 1 ELSE 0 END) as SND,
  SUM(CASE WHEN s.value LIKE '%Compressor%' THEN 1 ELSE 0 END) as Compressor,
  SUM(CASE WHEN s.value LIKE '%DynamicQuant%' THEN 1 ELSE 0 END) as DQ,
  SUM(CASE WHEN s.value LIKE '%Rotary%' THEN 1 ELSE 0 END) as Rotary,
  SUM(CASE WHEN s.value LIKE '%QuantLightningIndexer%' THEN 1 ELSE 0 END) as QLI
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId = cti.globalTaskId
JOIN STRING_IDS s ON cti.name = s.id
GROUP BY t.streamId ORDER BY t.streamId
```

### 3.2 Present findings to the user

Show the result table and let the user interpret it before you do. Ask:
- "Which of these streams do you think is the main DSA stream?"
- "Do you see a stream that only has weights_proj?"
- "Are any streams unexpected?"

**Do NOT jump to conclusions about which stream is "main" vs "sub" without
the user's input.** The user knows their configuration and architecture better
than you do.

### 3.3 Report time windows

```sql
SELECT streamId, MIN(startNs)/1e6, MAX(startNs)/1e6,
       (MAX(startNs)-MIN(startNs))/1e6 as dur_ms
FROM TASK GROUP BY streamId ORDER BY streamId
```

This reveals whether streams run in the same time window (overlap) or in
distinct phases (sequential).

---

## Phase 4: Targeted Drill-down

Once the user confirms the stream mapping, run targeted queries.

### 4.1 Timing overlap analysis

To check if two streams execute in parallel:

```python
# Find 1ms buckets where both streams have active tasks
# This only makes sense AFTER confirming which operators/streams to check
SELECT COUNT(*) FROM (
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <sub> AND <operator_condition>
  INTERSECT
  SELECT FLOOR(startNs / 1000000) FROM TASK
  WHERE streamId = <main> AND <operator_condition>
)
```

### 4.2 Detailed timeline

For a specific time window, show the exact sequence of operators on each
stream. This helps verify concurrent execution.

```sql
SELECT startNs/1e6, streamId,
  CASE WHEN s.value LIKE '%A%' THEN 'label_a'
       WHEN s.value LIKE '%B%' THEN 'label_b'
       ELSE 'other' END as tag
FROM TASK t
JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId=cti.globalTaskId
JOIN STRING_IDS s ON cti.name=s.id
WHERE streamId IN (<streams>)
AND startNs BETWEEN <t0> AND <t1>
ORDER BY startNs
```

### 4.3 Time offset critical check

When the user says "from 694ms", ALWAYS clarify the reference point:
- From the start of which stream?
- Absolute or relative timestamp?
- Confirm the baseline before running the query

**Common mistake**: computing relative time from the wrong stream's startNs.
The same relative offset can mean different absolute times for different
streams.

### 4.4 Stream sync verification

Cross-stream synchronization:
```sql
SELECT COUNT(*) FROM CANN_API ca
JOIN STRING_IDS s ON ca.name=s.id
WHERE s.value = 'aclrtStreamWaitEvent'
```

---

## Phase 5: Report

When presenting findings:

1. **Use a clear comparison table** showing operator counts per stream
2. **Note the total time window** and overlap duration for each stream
3. **State the conclusion clearly** — whether dual-stream is ON or OFF
4. **Explain WHY** based on the data, not just the conclusion
5. **Acknowledge uncertainty** — e.g., "the profiler sees all ops on one
   stream, but this may be a graph compilation artifact"

### Example report structure

```
## Conclusion: DSA Dual-Stream IS Active

Evidence:
- Stream 63 has 630 MatMulV2 (weights_proj) and 0 scatter/QLI → sub-stream
- Stream 64 has 1140 SND + 600 Compressor + 3150 QBM → main stream
- 300/812 1ms buckets have both streams active simultaneously → 37% overlap
- aclrtStreamWaitEvent: 667 cross-stream sync calls

## Conclusion: DSA Dual-Stream NOT Active

Evidence:
- Stream 61 has ALL DSA ops (QBM, DQ, SND, Rotary, QLI) → single stream
- Stream 64 also has ALL DSA ops → different graph size, not sub-stream
- No stream has weights_proj alone without scatter/QLI
```

---

## Common Pitfalls to Avoid

### ❌ Wrong: Assuming operator-to-code mappings
"QuantBatchMatmul must be weights_proj" — WRONG. It could be QKV attention.
Always confirm mappings with the user.

### ❌ Wrong: Ignoring non-DSA streams
A stream with Muls + MatMulV2 (stream 63) may be the real weights_proj stream.
Check ALL streams, not just the ones with obvious DSA operators.

### ❌ Wrong: Computing relative offset from the wrong baseline
The user says "694ms" — from which stream's start? Different streams have
different start times (sometimes 13+ms apart). Clarify before computing.

### ❌ Wrong: Drawing negative conclusions from missing data
"No separate sub-stream visible" does not mean "dual-stream not working."
Graph compilation may merge streams. The user knows their config.

### ❌ Wrong: Confusing graph instances with parallel streams
Two streams each with all DSA ops (QBM, SND, QLI) are likely different
captured graph sizes, NOT main+sub streams.

### ✅ Right: Incremental presentation
Show a stream overview first. Let the user interpret it. Then drill down.
Don't jump to conclusions.
