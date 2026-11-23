# CSI Training

给 Codex Desktop 安装一份用于 vLLM-Ascend 排障的技能。面对日志时，Codex 可以从已选版本中检索相关源码入口、观测方法和调查记录。默认提供最近一次完整发布的 **K76**，也可以在相同问题上试空基线 K0 或早期版本。

**[下载 Windows 安装器 Install-CSI.cmd](https://github.com/Delva0/csi-training/releases/download/2026-11-22-mainline/Install-CSI.cmd)** · [下载与 SHA-256 校验值](https://github.com/Delva0/csi-training/releases/tag/2026-11-22-mainline) · [查看训练与版本来源](checkpoints/2026-11-22-mainline/TRAINING.md)

## 安装和切换

1. 双击安装器，选择 Codex **运行 agent 的环境**。在 Windows 本机运行选 `1`，使用 WSL 运行选 `2`，然后选择对应发行版。安装器只操作所选环境当前用户的技能目录。[Codex 的 Windows 与 WSL 设置](https://learn.chatgpt.com/docs/windows/windows-app)
2. 按回车安装 K76，或输入下面的版本名。安装完成会显示版本号。
3. 在 Codex **新建对话**，输入 `$csi-vllm-ascend` 并附上实际首问、日志和当时知道的版本。若技能列表尚未刷新，重启 Codex。[Codex 技能的发现与调用](https://learn.chatgpt.com/docs/build-skills)

| 输入 | 得到什么 |
| --- | --- |
| `K76` | 2026 年 11 月 22 日完整发布的主线终点 |
| `K1` 至 `K75` | 同一训练链的早期版本，包含自己的 `K20` |
| `legacy-K20` | 11 月 18 日旧分层版，原名 `2026-11-18-k20` |
| `flat-K20` | 11 月 2 日旧扁平版，原名 `2026-11-02-k20` |
| `K0` | 移除本安装器的活动技能，回到空基线 |

更换版本时再运行同一个安装器。比较两个版本时，每次切换后新建对话，使用同一份首问和相同的源码条件。K0 直接发送首问。方便的话，可私下反馈哪一步取证最有帮助，或哪一步把你带偏。现场日志可以留在本地。

### 已在 Linux／WSL 终端

以下命令适用于 x86-64。ARM64 将下载地址末尾的 `amd64` 改成 `arm64`。以后可直接用下载的程序切换版本，例如 `./csi-install K0`。

```sh
curl -fL https://github.com/Delva0/csi-training/releases/download/2026-11-22-mainline/csi-install-linux-amd64 -o csi-install && chmod 700 csi-install && ./csi-install K76
```

## 本地会安装什么

Codex 的用户技能目录 `~/.agents/skills/csi-vllm-ascend/` 中只有一个活动入口。它将训练时的 `skill_view`、`/workspace/repos` 等用法对应到桌面可用的文件读取、终端、工作区 Git 仓库和已连接的 GitHub 工具。已选公开快照按 SHA-256 校验后缓存在 `~/.csi-training/`。全栈源码和训练机的专用工具需由使用环境提供。同名技能已由用户创建或修改时，安装器会保留它并报告冲突。

[79 个版本及快照校验值](checkpoints/2026-11-22-mainline/manifest.json) · [旧分层 K20](checkpoints/2026-11-18-k20/) · [旧扁平 K20](checkpoints/2026-11-02-k20/) · [安装器源码](installer/)
