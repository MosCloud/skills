---
name: vllm-profiler
description: >
  Collect and analyze NPU profiling data for vLLM Ascend inference services.
  Use this skill whenever the user wants to profile, benchmark, trace, verify
  dual-stream parallelism, or analyze NPU stream/operator performance on
  Ascend hardware. Triggers on mentions of profiling, profiler, trace, npu
  stream analysis, performance capture, or when they want to verify whether
  a feature (like dual-stream) is active in the compiled graph.
---

# vLLM Ascend Profiling Collector

Collect NPU profiling traces from a running vLLM Ascend service, parse them via
`torch_npu.profiler.analyse()`, analyze stream and operator behavior from the
SQLite database, and package results for sharing.

## Workflow Overview

```
[Step 1: ENV CHECK] → [Step 2: CONFIGURE REQUEST] → [Step 3: CONFIRM & WARMUP]
→ [Step 4: COLLECT PROFILING] → [Step 5: PARSE DATA] → [Step 6: ANALYZE]
→ [Step 7: PACKAGE]
```

Each step prompts the user for confirmation or input before proceeding. Do NOT
skip steps or make assumptions about what the user wants.

---

## Step 1: Environment Check

Before any profiling, verify three things:

### 1.1 Service health

```bash
curl -s http://<server_url>/v1/models
```

If this fails, tell the user and stop. Ask them to restart the service.

Default `server_url`: `http://127.0.0.1:7000`. Let the user override it.

### 1.2 Profiling config in start script

Read the service startup script (ask the user for its path, or check common
locations like `start_test.sh`, `start.sh`). Look for `--profiler-config` in
the arguments.

Confirm the profiler type is `"torch"` and note the `torch_profiler_dir` value
(usually `"./vllm_prof"` or similar). This is where profiling output will land.

If `--profiler-config` is missing, warn the user that profiling won't produce
output without it.

### 1.3 Additional config (for targeted analysis)

If the user wants to verify a specific feature (e.g. dual-stream), check the
`--additional-config` for the relevant toggle:

```bash
grep "additional-config" <start_script>
```

Report the current state: `"dsa_dual_stream": true` or `"dsa_dual_stream": false`.

---

## Step 2: Configure the Inference Request

Show the user the inference request that will run during profiling. This is
what drives the GPU/NPU computation that gets traced.

### 2.1 Default request template

```json
{
  "model": "deepseek_v4",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "max_tokens": <N>,
  "temperature": 0
}
```

### 2.2 Let the user customize

Ask the user:
- **Model name**: default `deepseek_v4`
- **Prompt content**: default `"你好，请简单的介绍你自己。"`
- **Max tokens**: default `128`
- **Any other request parameters** (e.g. extra headers, streaming, etc.)

Print the final curl command that will be used, and wait for confirmation.

---

## Step 3: Confirm and (Optionally) Warmup

### 3.1 Confirm the collection plan

Summarize what will happen:

```
Profiling plan:
  Server:    http://127.0.0.1:7000
  Model:     deepseek_v4
  Request:   "你好，请简单的介绍你自己。" (max_tokens=128)
  Output:    ./vllm_prof/
  Feature:   dsa_dual_stream=true
```

Ask the user to confirm.

### 3.2 Warmup (optional, ask first)

Ask: "是否需要在正式采集前先 warmup？"

If yes, run the same inference request 1-3 times WITHOUT starting the profiler.
This ensures the model is fully loaded and compiled graphs are ready.

```
Warmup: curl <same request>  →  discard output
Warmup: curl <same request>  →  discard output
Warmup: curl <same request>  →  discard output
```

If no, skip to Step 4.

---

## Step 4: Collect Profiling

### 4.1 Clear old data

```bash
rm -rf <profiler_dir>/*
```

### 4.2 Start profiling

```bash
curl -s -X POST http://<server>/start_profile -H "Content-Type: application/json"
```

Check HTTP 200. If it fails, the profiling API may not be available — tell the user.

### 4.3 Run the inference request

Execute the curl command from Step 2. Print the truncated output to confirm
the request completed normally.

### 4.4 Stop profiling

```bash
curl -s -X POST http://<server>/stop_profile -H "Content-Type: application/json"
```

### 4.5 Locate the profiling data

Under `<profiler_dir>/`, find the rank-specific directories:

```bash
ls <profiler_dir>/dp0_pp0_tp0_dcp0_ep0_rank0_*
```

Each rank has two directories:
- `*_ascend_pt/` with `FRAMEWORK/` + `PROF_*/` subdirectories (raw NPU trace)
- `*_ascend_pt/` with `profiler_info_*.json` (metadata only)

The directories with `FRAMEWORK` subdirectories contain the actual profiling
data for that rank.

---

## Step 5: Parse Data

### 5.1 Ask: parse all ranks or specific rank?

Ask: "是否需要解析所有 rank 的数据，还是只解析指定 rank？"

- **All ranks**: iterate over each rank directory with `FRAMEWORK/` present
- **Specific rank**: the user picks one (default: rank0 for stream analysis)

### 5.2 Run analyse()

For each selected rank directory:

```python
from torch_npu.profiler.profiler import analyse

analyse("<profiler_dir>/<rank_dir>")
```

This creates `ASCEND_PROFILER_OUTPUT/` inside the rank directory containing:
- `ascend_pytorch_profiler.db` (or `ascend_pytorch_profiler_0.db`) — main trace DB
- `trace_view.json` — Chrome trace for visualization
- `operator_details.csv` — per-operator breakdown
- `kernel_details.csv` — per-kernel breakdown
- `analysis.db` — comm analysis
- `op_statistic.csv` — operator statistics

Note: `analyse()` may take 3-5 minutes per rank.

### 5.3 Verify parsing success

Check that `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db` exists and is
non-empty for each parsed rank.

---

## Step 6: Analyze

After parsing, ask the user what analysis they want:

### Default analysis options to offer:

1. **Stream distribution** — Which NPU streams exist, their task counts, wall
   time, and compute time. This is the primary tool for verifying dual-stream
   parallelism.
   ```sql
   SELECT streamId, COUNT(*) FROM TASK GROUP BY streamId ORDER BY COUNT(*) DESC
   ```

2. **DSA operator distribution** — QuantBatchMatmul, DynamicQuant,
   ScatterNdUpdate, Rotary, QLI counts per stream. Reveals whether
   weights_proj is correctly on a sub-stream.
   ```sql
   SELECT t.streamId, COUNT(*) FROM TASK t
   JOIN COMPUTE_TASK_INFO cti ON t.globalTaskId=cti.globalTaskId
   JOIN STRING_IDS s ON cti.name=s.id
   WHERE s.value LIKE '%<operator>%' GROUP BY t.streamId
   ```

3. **Stream synchronization** — Count `aclrtStreamWaitEvent` calls to
   verify cross-stream sync is happening.

4. **Custom analysis** — The user can specify their own SQL queries or
   analysis goals.

Present findings in a clear table format. For dual-stream verification,
compare against the expected pattern:
- c4 dual-stream: DSA ops on main stream + sub-stream with ~2:1 ratio
- c4 dual-stream OFF: all DSA ops on main graph streams only
- c128: no indexer ops at all

---

## Step 7: Package

### 7.1 Ask for output location

Ask: "将解析后的数据打包到哪个目录下？"

Default: the current working directory or `./prof_docs/`.

### 7.2 Create tar.gz (ASCEND_PROFILER_OUTPUT only)

Only package the parsed output, not the raw FRAMEWORK/PROF data:

```bash
tar czf <output_dir>/vllm_prof_<description>_rank<N>.tar.gz \
  -C <profiler_dir>/<rank_dir> \
  ASCEND_PROFILER_OUTPUT
```

### 7.3 Report

Print the archive path and size:

```
Packaged: <output_dir>/vllm_prof_*.tar.gz (XXX MB)
```

---

## Default Configuration Reference

| Parameter | Default Value |
|---|---|
| Server URL | `http://127.0.0.1:7000` |
| Profiler dir | `./vllm_prof/` (from `torch_profiler_dir`) |
| Model | `deepseek_v4` |
| Prompt | `"你好，请简单的介绍你自己。"` |
| Max tokens | `128` |
| Temperature | `0` |
| Default rank | `rank0` |
| Output dir | `./prof_docs/` |

All of these can be overridden by the user at each step.

---

## Common Analysis Patterns

### Verify dual-stream is ON

```
Expected: DSA ops (QBM, DQ, SND, Rotary, QLI) on at least 2 streams with ~2:1 ratio
Key stream: streamId=X (main) + streamId=Y (sub)
Sync: aclrtStreamWaitEvent count > 0
```

### Verify dual-stream is OFF

```
Expected: All DSA ops on 1-2 main graph streams, NO standalone sub-stream with DSA ops
Key streams: streamId=A + streamId=B (both main, different graph sizes)
Sync: aclrtStreamWaitEvent may be present (from multistream_overlap_shared_expert)
```

### Check operator hot spots

Query `operator_details.csv` for top operators by total duration.
