# Podcast Radar macOS 版

> 面向研究人员的本地播客雷达：发现信源、核对节目身份、转录音视频、阅读中文全文并播放译文。

## 下载与授权

- 最新内测版：[GitHub Releases](https://github.com/qihaoshen857-ship-it/Podcast-Radar/releases/latest)
- 问题与建议：[GitHub Issues](https://github.com/qihaoshen857-ship-it/Podcast-Radar/issues)
- 当前仓库为公开源码预览，允许个人非商业学习和评估；商业使用、再分发和打包销售需取得单独授权，详见 `LICENSE`。
- API Key、Chrome Cookie、下载音频、转录稿、译文和个人资料库均不会提交到 GitHub。

这是从原 Windows 便携包适配出的 macOS 版本，用来完成：

1. 自动搜索 AI、天气、农业领域重要播客更新，或从 YouTube / 小宇宙读取节目列表并下载音频。
2. 将下载后的音频自动转录为同名 TXT。
3. 选择本地音频或视频文件单独转录。
4. 在阅读页按需把英文完整转录逐段翻译成中文，并用本机 Qwen3-TTS 分段合成为可播放的译文音频。

转录可在设置页选择 `本地优先` 或 `极速云端`。长音频会先通过 Silero VAD 在本地切分；
`极速云端` 会默认用 4 路阿里云百炼 `qwen3-asr-flash` 并发处理，再按原顺序去重合并。
没有 Key 或个别云端片段失败时，本机 `faster-whisper medium` 会补齐缺失片段。

## 快速开始

首次使用建议双击一个入口：

```text
run_macos.command
```

它会在首次运行时自动安装项目本地运行环境，之后直接启动。

也可以分开执行，首次使用先双击：

```text
setup_macos.command
```

安装完成后双击：

```text
launch_macos.command
```

### 阶段2：本地 app 分发（你当前目标）

执行：

```text
./build_macos_app.command
```

生成目录：`release/` 下会出现以下文件（版本号与时间戳会变化）：

- `PodcastRadar-<版本>-<时间戳>.dmg`：双击安装。
- `PodcastRadar-<版本>-<时间戳>.zip`：解压后直接交付同事使用。
- `.../release/<版本包>/update-manifest.json`：便于你后续补充“自动更新”元数据。

说明：

- 打包会带上精简的本地媒体运行时（`ffmpeg/ffprobe/deno` 及必需动态库），避免同事端环境缺失，同时避免完整 Python 运行时拖慢首次启动。
- APP 的显示名称为 `Podcast Radar`。为直接沿用历史资料，配置仍落在 `~/Library/Application Support/ResearchPodcastRadar`。

详细安装、Cookie、常见问题和导出说明见：

```text
README_MAC.md
```

如果你要填写自己的 API Key、替换本地 Whisper 模型，或接入其他云端 / 自建 ASR 服务，请看：

- [使用与模型配置指南](docs/使用与模型配置指南.md)

## 本机使用

- 打开应用后会自动读取 `双层雷达`：AI、天气、农业播客 RSS，精选 AI / 养生 YouTube 官方频道源，以及小宇宙精选节目页，并分成 `AI 播客` / `天气农业` / `养生` 区域展示。
- 点击 `AI`：只刷新 AI 播客更新。
- 点击 `天气农业`：只刷新天气、ENSO、气候和农业市场相关播客更新。
- 点击播客卡片：打开详情页，自动生成中文沉浸式翻译，并保留英文原文；右下角 `选择` 按钮才负责勾选。
- 在 `手动链接` 中粘贴 YouTube / 小宇宙链接后点击 `读取`：保留原来的指定来源下载流程。
- 勾选 `下载后自动转录`：下载完成后自动进入本地转录队列。
- 选中卡片后点击 `转录整理选中`：先下载到隐藏缓存，再按设置的 `本地优先` / `极速云端` 模式生成转录，并在有 Key 时生成中文 Markdown 研究笔记。
- 打开英文 TXT 后点击 `看译文`：按完整原文分段翻译并缓存到资料库；原文变化、模型变化或翻译规则升级时会自动重建。
- 点击 `播放译文`：优先复用已有译文音频；首次使用会调用本机 Qwen3-TTS 分段生成一个连续 WAV，长文不会只读前 1800 字。
- 官方 Transcript 只在节目标题一致时复用；标题错期、缺失或不可验证时，会改走当前节目的音频/视频下载与 ASR。
- `极速云端` 默认 4 路并发、最多 8 路；每片完成即写入断点缓存，重试时只补缺失片段。缺少 Key 或云端失败时会自动交给本地 Whisper。
- 如果某次只能基于 RSS 简介生成轻量稿，下次整理仍会重试真实媒体，不会被旧轻量稿永久卡住。
- 点击 `标为精选` / `取消精选` 管理本地精选库；点击 `下载精选` 才把精选节目正式下载到本地下载目录。

## 主要文件

- `setup_macos.command`：双击安装运行环境。
- `launch_macos.command`：双击启动工具。
- `run_macos.command`：自动判断是否需要安装，然后启动。
- `check_system.command`：双击检查运行环境、Key、设置、Chrome。
- `doctor_macos.sh`：检查本机运行环境。
- `setup_macos.sh`：终端版安装脚本。
- `launch_macos.sh`：终端版启动脚本。
- `export-portable-macos.sh`：导出新的 Mac 便携 zip。
- `build_macos_app.command`：生成面向同事的本地 app 分发包（阶段2）。
- `main.py`：桌面工具入口。
- `app/transcription.py`：本地音频切分与转录逻辑。
- `app/research_digest.py`：转录文本的中文翻译、摘要和研究重点整理。
