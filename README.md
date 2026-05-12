# MosCloud Skills

vLLM Ascend 开发相关的 Claude Code / Cowork skills。

## 安装

将 `.skill` 文件放入 `~/.agents/skills/` 目录：

```bash
cp */<skill-name>.skill ~/.agents/skills/
```

重新加载后 skills 自动生效。

## Skills

### [vllm-profiler](vllm-profiler/)

采集 vLLM Ascend 推理服务的 NPU profiling 数据。

**功能**:
- 环境检查（服务存活、profiler 配置）
- 配置推理请求并确认
- 可选 warmup
- 采集（start → curl → stop）
- 解析全部或指定 rank
- 一键打包 `ASCEND_PROFILER_OUTPUT` 为 tar.gz

### [profiling-analysis](profiling-analysis/)

分析已解析的 NPU profiling 数据（`ascend_pytorch_profiler.db`）。

**核心原则**: 先确认再下结论。算子映射、流角色、时间偏移量都需要和用户确认。

**功能**:
- 全流扫描与算子分布分析
- 跨流时序重叠分析
- 特定特性验证（双流并行等）
- 性能热点分析

## 开发

添加新 skill：

```bash
mkdir <skill-name>/
# 编写 SKILL.md
# 生成 .skill 打包文件
python3 -m scripts.package_skill <skill-path>

# 提交
git add <skill-name>/
git commit -s -m "feat: add <skill-name> skill"
git push
```
