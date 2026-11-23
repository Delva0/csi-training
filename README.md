# CSI Training

给 Codex 装一份用于 vLLM-Ascend 排障的技能。把首问、日志和当时知道的版本交给 Codex，它可以沿技能中的线索去查源码和现场证据。默认版本为 **K76**。[阅读 K76 技能正文](checkpoints/2026-11-22-mainline/skills/vllm-ascend-triage/SKILL.md) · [按版本浏览](checkpoints/2026-11-22-mainline/)

## Windows PowerShell

如果 Codex 的 agent 在 Windows 本机运行，复制这条命令安装。Git 会创建所需目录，下载整条训练链上的版本。

```powershell
git clone --branch codex --single-branch https://github.com/Delva0/csi-training.git "$env:USERPROFILE\.agents\skills\csi-vllm-ascend"
```

在 Codex **新建对话**，输入 `$csi-vllm-ascend` 并附上首问。若技能列表尚未刷新，重启 Codex。[Codex 技能文档](https://learn.chatgpt.com/docs/build-skills)

要换版本，只需切到已有的标签。下面这条命令选择主线 K20，切换无需联网。

```powershell
git -C "$env:USERPROFILE\.agents\skills\csi-vllm-ascend" switch --detach codex-K20
```

| 标签 | 版本 |
| --- | --- |
| `codex-K76` | 最近一次完整发布的主线终点 |
| `codex-K1` 至 `codex-K75` | 同一训练链上的任一步，包括主线 K20 |
| `codex-legacy-K20` | 11 月 18 日的旧分层 K20 |
| `codex-flat-K20` | 11 月 2 日的旧扁平 K20 |
| `codex-K0` | 空基线，移除本仓的活动技能入口 |

## WSL 或 Linux

如果 Windows 上的 Codex 使用 WSL agent，请在它使用的发行版终端中安装。下列两行分别是安装 K76 和切换到主线 K20。[查看 Windows 与 WSL 设置](https://learn.chatgpt.com/docs/windows/windows-app)

```sh
git clone --branch codex --single-branch https://github.com/Delva0/csi-training.git "$HOME/.agents/skills/csi-vllm-ascend"
git -C "$HOME/.agents/skills/csi-vllm-ascend" switch --detach codex-K20
```

## 对照和已有安装

每次切换后新建对话，复制同一份首问，保持源码条件一致。选 K0 时直接发送首问。愿意的话，私下告诉我哪一步取证最有帮助，或哪一步把你带偏。现场日志可以留在本地。

Git 安装只放入一个技能及其资料。资料以 Markdown 为主，部分版本附有可选的参考脚本。安装和切换都由已有的 Git 完成，无需运行下载的程序、注册工具、配置 MCP 或连接 GitHub 账号。文中的 `skill_view` 是训练时对读取技能文件的叫法。

<details>
<summary>已有同名目录或用过旧安装器</summary>

已经用上面的 Git 命令安装过，直接使用切换命令。若同名目录已有其他内容，Git 会停止并保留原文件。

确认要迁移旧版 CSI 技能时，先把安装命令的目标换成用户主目录下一个新的临时目录，等克隆成功，再把原来的 `csi-vllm-ascend` 整个移到 `.agents/skills` 之外留存，最后把新目录移到原位置并命名为 `csi-vllm-ascend`。原来的文件和修改都随备份保留。完成迁移后即可用 Git 切换。

</details>

[浏览技能分支](https://github.com/Delva0/csi-training/tree/codex) · [训练与版本来源](checkpoints/2026-11-22-mainline/TRAINING.md) · [79 个版本的清单](checkpoints/2026-11-22-mainline/manifest.json) · [上一代安装器](https://github.com/Delva0/csi-training/releases/tag/2026-11-22-mainline)
