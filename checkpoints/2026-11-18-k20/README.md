# CSI training 2026-11-18-k20

CSI（code-skill-issue）从空 K0 经 20 次 `full_random` 更新得到的分层知识 profile。训练完成于北京时间 **2026-11-18 10:14:35**。

- [训练设计、恢复与终点评测](TRAINING.md)
- [vLLM-Ascend 排障 Skill](profile/skills/vllm-ascend-bug-report-triage/SKILL.md)及其[分层参考](profile/skills/vllm-ascend-bug-report-triage/references/)
- [配置、正式知识提交与逐文件校验值](manifest.json)
- [训练快照身份](provenance/K20-snapshot.json)与[执行源码身份](provenance/execution-source.json)

本版本共有 91 个知识文件（1,650,900 字节），含 1 个 Skill 入口、75 篇参考、14 个工具脚本和 1 个报告模板。技能入口按调查需要引向工作流、症状、组件和仓库证据。

## 使用与校验

在本版本目录执行：

```sh
sha256sum -c SHA256SUMS
```

将完整的 [`profile/`](profile/) 复制到任务目录，向已安装的 CSI 传入 `--state /absolute/path/to/copied-profile`。该路径包含 `config.yaml` 和 `skills/`。模型与工具凭据由 CSI 的运行环境提供。
