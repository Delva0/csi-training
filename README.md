# CSI Training

保存 CSI（code-skill-issue）训练生成的知识版本。每个版本按训练完成日期（Asia/Shanghai）和 checkpoint 命名，并使用同名 Git tag 固定。

## 重要文档

[TRAINING.md](TRAINING.md)：CSI 当前训练设计，包含数据划分、采样、Forward、Reviewer／Merge、Loss 与验证流程。

文中的相对源码路径以 CSI 工程目录为基准。各知识版本的实际配置和来源记录见对应的 `manifest.json`。

## 训练版本

| 训练版本 | Checkpoint | 训练完成时间（北京时间） | 原始知识提交 |
| --- | --- | --- | --- |
| [2026-11-02-k20](checkpoints/2026-11-02-k20/) | K20 | 2026-11-02 01:47:34 | `39088928ad0fda9ca49950d126fb6d3f6f447e99` |

## 目录约定

每个 `checkpoints/<日期>-<checkpoint>/` 包含可加载的 `profile/`、版本说明、`manifest.json`、原始身份记录和 `SHA256SUMS`。后续训练版本使用新目录和新 tag 保存。

当前版本的入口见 [K20 版本说明](checkpoints/2026-11-02-k20/README.md)。
