# CSI training 2026-11-02-k20

CSI（code-skill-issue）训练产出的累计 K20 知识 profile。

- 训练完成时间：`2026-11-02T01:47:34+08:00`（Asia/Shanghai）。目录和 tag 使用训练完成日期。
- 正式知识提交：`39088928ad0fda9ca49950d126fb6d3f6f447e99`。
- 知识树：`4542280e02aae0597d51628240c525b4ca18ca66`。
- 内容：2 个 skill 入口、44 篇专题参考文档、1 个查询脚本，共 47 个知识文件、1,056,331 字节；另保存原 profile 配置。
- 版本包含 20 次学习更新及原有初始知识；K10 的人工修正单独记录在 manifest 中。

## 内容

- [问题排查主入口](profile/skills/vllm-ascend-issue-triage/SKILL.md)
- [专题参考知识](profile/skills/vllm-ascend-issue-triage/references/)
- [Quay 镜像版本查询](profile/skills/quay-image-versions/SKILL.md)
- [原 profile 配置](profile/config.yaml)
- [版本身份和逐文件校验值](manifest.json)
- [原始 K20 快照身份记录](provenance/K20-snapshot.json)
- [训练源码身份记录](provenance/execution-source.json)

## 使用

将完整 `profile/` 复制到自己的任务目录，向已安装的 CSI 传入 `--state /absolute/path/to/copied-profile`。该参数使用 profile 根目录；它包含 `config.yaml` 和 `skills/`。模型与工具凭据按 CSI 的运行环境单独配置。

## 校验

在本版本目录执行：

```sh
sha256sum -c SHA256SUMS
```

导出时，47 个知识文件逐一核对了原始 SHA-256 和正式知识提交中的 Git blob，配置与原 profile 逐字节一致。
