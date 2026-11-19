# CSI Training

保存 CSI（code-skill-issue）训练生成的知识版本。每个版本按训练完成日期（Asia/Shanghai）和 checkpoint 命名，并使用同名 Git tag 固定。

## 训练版本

| 训练版本 | 知识结构 | 训练完成时间（北京时间） | 原始知识提交 | 训练说明 |
| --- | --- | --- | --- | --- |
| [2026-11-18-k20](checkpoints/2026-11-18-k20/) | 分层专题 | 2026-11-18 10:14:35 | `36de135c08352aa6928c4d46208655728cb70097` | [TRAINING.md](checkpoints/2026-11-18-k20/TRAINING.md) |
| [2026-11-02-k20](checkpoints/2026-11-02-k20/) | 扁平参考 | 2026-11-02 01:47:34 | `39088928ad0fda9ca49950d126fb6d3f6f447e99` | [TRAINING.md](checkpoints/2026-11-02-k20/TRAINING.md) |

## 目录约定

每个 `checkpoints/<日期>-<checkpoint>/` 包含可加载的 `profile/`、版本说明、`manifest.json`、来源记录和 `SHA256SUMS`。训练说明引用的源码路径以 CSI 工程目录为基准。
