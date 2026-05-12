# vllm-profiler 版本历史

## v1.1 (2026-05-12)

步骤6 增加 skill 联动：分析时先尝试加载 `profiling-analysis` skill，
成功则委托给专业分析 skill，失败则走独立分析 fallback。

### 变更
- 步骤6.1：新增 `skill("profiling-analysis")` 加载逻辑
- 步骤6.2：原分析内容降为 fallback

## v1.0 (2026-05-12)

初始版本。基于会话中的 profiling 采集流程提炼，包含完整的采集工作流。

### 功能
- 环境检查（服务存活、profiler 配置）
- 配置推理请求并确认
- 可选 warmup
- 采集（start → curl → stop）
- 解析全部或指定 rank
- 一键打包 ASCEND_PROFILER_OUTPUT 为 tar.gz

---

### 版本归档

更新版本时，将当前版本的以下文件复制到 `archive/` 目录，再修改 SKILL.md：

- `archive/v{N}.0_SKILL.md` — 当前版本的 SKILL.md 快照
- `archive/v{N}.0.skill` — 当前版本的 .skill 打包文件

然后在 HISTORY.md 中追加新版本记录，提交并打 tag。
