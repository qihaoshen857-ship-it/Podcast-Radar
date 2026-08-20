# Podcast Radar macOS 安装与使用

这是一版从 Windows 便携包改出的 macOS 方案，目标是让工具能在 Mac 上完成：

1. 自动搜索 AI 产业、AI 创业、天气农业和科学养生领域的重要播客更新，或读取 YouTube / 小宇宙节目列表并下载音频。
2. 下载后自动转录，或选择本地 `mp3`、`m4a`、`wav`、`mp4` 转录。
3. 可选 `本地优先` 或 `极速云端`；极速云端会切片后并发调用阿里云，失败片段由本机 `faster-whisper medium` 补齐。
4. 英文转录可按需生成中文全文译文，并用本机 Qwen3-TTS 播放整篇译文。

当前公开版本为 `v0.4.42`。人物监控访谈已经接入与今日雷达相同的转录整理流程；英文标题、阅读排版、天气 ENSO 权重和任务进度反馈也已同步更新。

## 首次安装

需要 macOS 和网络环境。纯本地 TXT 转录不强制要求 `DASHSCOPE_API_KEY`；极速云端和中文纪要需要 Key。

为避免牺牲 YouTube 下载能力，默认安装要求：

- Python 3.10+，并带 tkinter。
- 最新 `yt-dlp` 支持的运行环境。
- ffmpeg / ffprobe。
- Deno，用于 `yt-dlp` 处理 YouTube JS Challenge。

当前 macOS 系统自带 Python 3.9.6 可以运行部分 Python 代码，但最新版
`yt-dlp` 已不支持 Python 3.9。因此默认不使用 Python 3.9 做完整功能环境。

无需安装 Homebrew。默认会在项目目录下创建本地运行环境 `.runtime`，并安装：

- Python 3.13
- Tk
- ffmpeg / ffprobe
- Deno
- Python 依赖

首次双击：

```text
run_macos.command
```

或者分开双击：

```text
setup_macos.command
launch_macos.command
```

安装脚本会下载一个项目本地 micromamba，并通过 conda-forge 创建完整运行环境。所有环境文件都在当前目录内，不会改系统 Python，也不需要管理员密码。

也可以在终端运行：

```bash
cd /path/to/podcast-transcriber-mac
./run_macos.sh
```

## 设置转录模式与 API Key

打开工具后进入 `设置` 页：

1. 选择 `本地优先` 或 `极速云端`；长播客希望明显提速时选 `极速云端`。
2. 极速云端或中文纪要需要填写 `DASHSCOPE_API_KEY`；只做本地 TXT 可以留空。
3. 保持默认切分参数和 4 路并发，或按需调整；程序安全上限为 8 路。
4. 点击 `保存设置`。

API Key 会保存到本机 `.env` 文件中。便携包导出脚本会排除 `.env`。

## 研究播客自动更新

应用打开后会自动读取 `双层雷达`，覆盖 AI 产业、AI 创业、天气农业与科学养生播客 RSS，并补充精选 YouTube 频道源和小宇宙节目页，解析真实音频地址或视频入口和封面。新增的 `AI创业` 主题聚焦创始人故事、产品从 0 到 1、获客和商业化，会过滤纯模型新闻、估值争论和产业快讯。主界面可单独点击 `AI`、`AI创业`、`天气` 或 `养生` 分主题刷新。当前保留的手动链接模式仍可用于 YouTube、频道、播放列表和小宇宙页面。

`人物监控` 页按 `今日雷达` 的结构统一了 52px 标题栏、右侧人物切换、操作按钮、蓝色来源摘要、栏目标题、真实节目封面和横向节目卡片。Apple Podcasts 使用节目封面，RSS 优先使用单集图片并回退到频道图片，Elon Musk 公开访谈档案会使用档案页图片或原视频缩略图；图片进入本地缓存，网络失败时才显示人物占位图。`档案已核验`、`优质固定源`、`跨节目发现` 会分层展示。

新的个人研究工作流：

1. 在今日雷达选中感兴趣的播客卡片，或在人物监控卡片上直接点击 `转录`。
2. 点击卡片主体可以打开详情页，工具会自动把节目标题/简介翻译成中文，并保留英文原文。
3. 点击 `转录整理选中`，确认后会先把音频放进隐藏缓存 `.cache/research/audio`，再按当前转录模式处理。
4. 极速云端会把片段并发交给阿里云 ASR，按原顺序去重合并；失败片段由本地 Whisper 补齐。配置 Key 后还会继续生成中文 Markdown 研究笔记，保存到 `.cache/research/digests`。
5. 打开英文 TXT 后点击 `看译文`，程序会逐段翻译完整原文并缓存；点击 `播放译文` 会分段生成一个连续的本地 WAV。
6. 如果内容值得保留，点击 `标为精选`；如果想正式留存音频，点击 `下载精选` 或 `下载已选`。
7. 精选、摘要、译文、朗读音频路径和原链接会写入 `research_library.json`，后续可直接作为上云前的网站数据雏形。

转录来源规则：

- 官方 Transcript 必须从文稿内提取出节目标题，并与当前 RSS 节目通过一致性校验。
- 错期稿会自动隔离为带时间戳的 `identity-mismatch` 备份，不再参与中文纪要生成。
- 没有同期官方稿时，程序继续下载当前播客音频或 YouTube 视频音轨，并按设置的转录模式处理。
- `本地优先` 先用 `faster-whisper medium`，失败时调用阿里云；`极速云端` 直接并发调用 `qwen3-asr-flash`，缺少 Key 或云端片段失败时改由本地补齐。
- RSS 简介轻量稿是最后一层应急结果，会明确标注“非完整转录”；后续再次整理时依然重试真实媒体。

## Cookie 与 YouTube

`浏览器登录态` 建议：

- `auto`：Mac 上会依次尝试已安装的 Chrome、Edge、Firefox。
- `chrome` / `edge` / `firefox`：只尝试指定浏览器登录态。
- `none`：不使用浏览器登录态。
- `cookies.txt`：如果你手动导出过 cookies，可以在输入框中选择。

如果本机没有 Edge，工具会自动优先使用 Chrome。选择 `edge` 时，工具会尝试通过调试端口刷新 YouTube Cookie。这个过程会先请求关闭当前 Edge，再重新启动 Edge 导出 Cookie；如果 Edge 里有未保存内容，先手动保存。

## 已做的 Mac 适配

- 修复 Windows zip 里的反斜杠路径问题，目录结构可以在 macOS 正常使用。
- 新增 `setup_macos.sh` / `setup_macos.command`。
- 新增 `launch_macos.sh` / `launch_macos.command`。
- 新增 `run_macos.sh` / `run_macos.command`，作为本机使用的单入口。
- 新增 `check_system.sh` / `check_system.command`，检查运行环境、Key、设置、Chrome。
- 新增 `doctor_macos.sh` 环境预检脚本。
- 新增 `export-portable-macos.sh`。
- Edge 自动 Cookie 刷新从 Windows `powershell/taskkill` 改为 macOS 可用的 `osascript/pkill` 与进程终止逻辑。
- 浏览器登录态会按本机实际安装情况选择；这台机器没有 Edge 时会用 Chrome。

## 阶段2：本地 app 分发（同事内测版）

如果你现在要做“同事可装、版本可控”的第2级，直接执行：

```bash
./build_macos_app.command
```

产物放在 `release/` 下：

- `PodcastRadar-<版本>-<时间戳>.dmg`：点开即可安装。
- `PodcastRadar-<版本>-<时间戳>.zip`：解压后发给同事。
- `release/<版本包>/update-manifest.json`：你后续要做更新提醒时可替换为真实下载地址。

这个脚本会把精简媒体运行时（`ffmpeg/ffprobe/deno` 及必需动态库）打进 `*.app` 的 `Contents/Resources/.runtime`，减少同事端环境漂移导致的缺链路，也避免完整 Python 运行时的大量小文件拖慢 macOS 首次启动。打包时还会为应用设置 `OPENSSL_CONF=/dev/null`，避免新签名应用在首次读取 OpenSSL 默认配置时被 macOS 文件检查长时间卡住。

应用显示名称为 `Podcast Radar`。为直接沿用改名前的数据，配置、缓存和摘要继续落在：

- `~/Library/Application Support/ResearchPodcastRadar`

这样不会和可执行文件同目录污染，也更容易做升级替换。

## 常见问题

### 双击脚本提示无法打开

在终端执行一次：

```bash
chmod +x run_macos.command run_macos.sh setup_macos.command launch_macos.command setup_macos.sh launch_macos.sh doctor_macos.sh export-portable-macos.sh build_macos_app.command build_macos_app.sh
```

### 缺少 Python 3.10+、ffmpeg、ffprobe 或 Deno

为了不牺牲功能，默认安装脚本不会降级跳过这些依赖。重新运行：

```bash
./setup_macos.sh
```

### Python 依赖安装失败

优先确认 Python 版本和依赖：

```bash
./doctor_macos.sh
```

然后重新运行：

```bash
./setup_macos.sh
```

### YouTube 仍然提示机器人校验

先在 Chrome 或 Edge 中手动登录 YouTube，再回到工具里选择对应浏览器登录态。如果浏览器 Cookie 被占用，可以关闭浏览器后重试，或手动导出 `cookies.txt`。

## 导出新的 Mac 便携包

执行：

```bash
./export-portable-macos.sh
```

生成的 zip 位于 `release` 目录，不包含 `.venv`、`.env`、音频下载、转录文本、cookies、设置和缓存。
