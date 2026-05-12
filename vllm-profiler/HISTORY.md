# vllm-profiler 版本历史

## v1.0 (2026-05-12)

初始版本。基于会话中的 profiling 采集流程提炼，包含完整的采集工作流。

### 功能
- 环境检查（服务存活、profiler 配置）
- 配置推理请求并确认
- 可选 warmup
- 采集（start → curl → stop）
- 解析全部或指定 rank
- 一键打包 ASCEND_PROFILER_OUTPUT 为 tar.gz
