#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
RELEASE_DIR="$ROOT_DIR/release"
DIST_DIR="$ROOT_DIR/dist"
WORK_DIR="$ROOT_DIR/.pyinstaller_work"
APP_DISPLAY_NAME="Podcast Radar"
EXECUTABLE_NAME="PodcastRadar"
PACKAGE_PREFIX="PodcastRadar"
APP_BUNDLE="$APP_DISPLAY_NAME.app"
ICON_PATH="$ROOT_DIR/assets/ResearchPodcastRadar.icns"
VERSION="$( \
  "$RUNTIME_DIR/bin/python" - <<'PY'
import os
import re
from pathlib import Path

text = Path("main.py").read_text(encoding="utf-8")
match = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
print(match.group(1) if match else "0.4.0")
PY
)"
STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_NAME="${PACKAGE_PREFIX}-${VERSION}-${STAMP}"
PACKAGE_DIR="$RELEASE_DIR/$PACKAGE_NAME"
ZIP_PATH="$RELEASE_DIR/${PACKAGE_NAME}.zip"
DMG_PATH="$RELEASE_DIR/${PACKAGE_NAME}.dmg"

if [[ ! -x "$RUNTIME_DIR/bin/python" ]]; then
  echo "未找到本地运行时，请先执行 setup_macos.command。"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
  echo "主程序不存在：$ROOT_DIR/main.py"
  exit 1
fi

mkdir -p "$RELEASE_DIR"
rm -rf \
  "$WORK_DIR" \
  "$DIST_DIR/.PodcastRadar-normalized.app" \
  "$DIST_DIR/$EXECUTABLE_NAME" \
  "$DIST_DIR/$EXECUTABLE_NAME.app" \
  "$DIST_DIR/$APP_BUNDLE" \
  "$DIST_DIR/ResearchPodcastRadar" \
  "$DIST_DIR/ResearchPodcastRadar.app"

"$RUNTIME_DIR/bin/python" -m pip install --disable-pip-version-check --no-input -r "$ROOT_DIR/requirements-build.txt"

echo "正在打包应用：$APP_DISPLAY_NAME v$VERSION ..."
PYINSTALLER_ICON_ARGS=()
if [[ -f "$ICON_PATH" ]]; then
  PYINSTALLER_ICON_ARGS=(--icon "$ICON_PATH")
fi
"$RUNTIME_DIR/bin/python" -m PyInstaller --clean --noconfirm --windowed --onedir --distpath "$DIST_DIR" --workpath "$WORK_DIR" --name "$EXECUTABLE_NAME" --osx-bundle-identifier "com.shenqihao.ResearchPodcastRadar" --add-data "$ROOT_DIR/modules/person_monitor:modules/person_monitor" "${PYINSTALLER_ICON_ARGS[@]}" "$ROOT_DIR/main.py"

GENERATED_PLIST_PATH="$(find "$DIST_DIR" -maxdepth 5 -type f -path '*/Contents/Info.plist' -print -quit)"
if [[ -z "$GENERATED_PLIST_PATH" ]]; then
  echo "未检测到 PyInstaller 输出的应用包。"
  exit 1
fi
GENERATED_APP_PATH="${GENERATED_PLIST_PATH%/Contents/Info.plist}"

APP_PATH="$DIST_DIR/$APP_BUNDLE"
# PyInstaller 版本升级后可能把真正的 .app 再包在同名目录中。
# 先归一化到临时路径，再改成对用户可见的应用名，避免产生嵌套 app。
NORMALIZED_APP_PATH="$DIST_DIR/.PodcastRadar-normalized.app"
rm -rf "$NORMALIZED_APP_PATH"
ditto "$GENERATED_APP_PATH" "$NORMALIZED_APP_PATH"
rm -rf "$APP_PATH"
mv "$NORMALIZED_APP_PATH" "$APP_PATH"

set_plist_value() {
  local plist_path=$1
  local key=$2
  local value=$3
  if /usr/libexec/PlistBuddy -c "Print :$key" "$plist_path" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Set :$key $value" "$plist_path" || true
  else
    /usr/libexec/PlistBuddy -c "Add :$key string $value" "$plist_path" || true
  fi
}

if [[ -f "$APP_PATH/Contents/Info.plist" ]]; then
  set_plist_value "$APP_PATH/Contents/Info.plist" CFBundleShortVersionString "$VERSION"
  set_plist_value "$APP_PATH/Contents/Info.plist" CFBundleVersion "$VERSION"
  set_plist_value "$APP_PATH/Contents/Info.plist" CFBundleName "$APP_DISPLAY_NAME"
  set_plist_value "$APP_PATH/Contents/Info.plist" CFBundleDisplayName "$APP_DISPLAY_NAME"
  if [[ -f "$ICON_PATH" ]]; then
    VERSIONED_ICON_NAME="ResearchPodcastRadar-v${VERSION//./_}.icns"
    cp "$ICON_PATH" "$APP_PATH/Contents/Resources/$VERSIONED_ICON_NAME"
    set_plist_value "$APP_PATH/Contents/Info.plist" CFBundleIconFile "$VERSIONED_ICON_NAME"
  fi
  if ! /usr/libexec/PlistBuddy -c "Print :LSEnvironment" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" "$APP_PATH/Contents/Info.plist"
  fi
  if /usr/libexec/PlistBuddy -c "Print :LSEnvironment:OPENSSL_CONF" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Set :LSEnvironment:OPENSSL_CONF /dev/null" "$APP_PATH/Contents/Info.plist"
  else
    /usr/libexec/PlistBuddy -c "Add :LSEnvironment:OPENSSL_CONF string /dev/null" "$APP_PATH/Contents/Info.plist"
  fi
fi

mkdir -p "$PACKAGE_DIR"
cp -R "$APP_PATH" "$PACKAGE_DIR/"

if [[ -d "$RUNTIME_DIR" ]]; then
  EMBEDDED_RUNTIME="$PACKAGE_DIR/$APP_BUNDLE/Contents/Resources/.runtime"
  mkdir -p "$EMBEDDED_RUNTIME/bin" "$EMBEDDED_RUNTIME/lib"
  for tool_name in ffmpeg ffprobe deno; do
    if [[ -x "$RUNTIME_DIR/bin/$tool_name" ]]; then
      cp "$RUNTIME_DIR/bin/$tool_name" "$EMBEDDED_RUNTIME/bin/"
    fi
  done
  # ffmpeg/ffprobe use @loader_path/../lib. Only their top-level shared
  # libraries are needed; copying the whole Python runtime adds ~47k files
  # and can make a newly signed macOS app spend minutes in first-launch scan.
  rsync -a --include='*.dylib' --exclude='*' "$RUNTIME_DIR/lib/" "$EMBEDDED_RUNTIME/lib/"
fi

# Info.plist and bundled runtime are added after PyInstaller's first signing pass.
# Re-sign the final bundle so macOS validates the artifact the user actually opens.
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$PACKAGE_DIR/$APP_BUNDLE"
fi

cat > "$PACKAGE_DIR/update-manifest.json" <<EOF
{
  "app_name": "Podcast Radar",
  "platform": "macos",
  "current_version": "${VERSION}",
  "release_date": "$(date +%Y-%m-%d)",
  "download_url": "https://github.com/qihaoshen857-ship-it/Podcast-Radar/releases/latest",
  "release_notes": [
    "新增机器人独立栏目，聚焦具身智能、人形本体、模型、零部件与量产应用。",
    "7 个官方视频频道与中英文长访谈筛选共同发现节目，保留真实来源与发布日期。",
    "机器人采用独立研究评分，区分量产计划、来源自述的交付数据，过滤纯表演与连续直播。",
    "六个栏目按钮支持窄窗口换行，黑色选中态与刷新滑板恐龙即时反馈。",
    "同步改进长英文标题两行显示、英文副标题、简介去重与完整标题悬停提示。",
    "人物监控卡片新增‘转译’入口，直接复用首页的下载、Whisper／阿里云 ASR 和中文研究纪要队列。",
    "人物 RSS 与 Apple Podcasts 条目保留真实音频地址、时长和简介，避免把节目网页误当成音频。",
    "Elon Musk 已核验档案优先读取完整公开 Transcript，并保留原始音视频作为备用。",
    "人物监控刷新新增逐信源进度、实时耗时和完成回执，点击后可直接确认正在检索。",
    "人物访谈改为发布时间优先排序，新节目不再被较旧的已核验档案压到列表后方。",
    "修复 Dwarkesh Podcast 失效的中转 RSS，Dylan Patel 和 Elon Musk 均切换到官方 Substack RSS。",
    "刷新完成后自动回到列表顶部，并明确显示本轮新发现条数或暂无更新。",
    "英文节目标题现在会在后台自动汉化，首页、资料库和阅读页共用同一份中文标题缓存。",
    "资料库会自动回填历史英文标题，已生成的转录、纪要、译文和朗读音频无需重做。",
    "资料库表格、提示区以及阅读页正文、译文与朗读操作区统一改为圆角容器。",
    "天气栏目新增厄尔尼诺、拉尼娜和 ENSO 专项权重，相关节目获得 2.3 分额外投资价值加成。",
    "关键数据与案例升级为逐段完整台账，覆盖全部数值、口径、主体、定性信号、对比和案例。",
    "研究整理输出上限提升至 8192 tokens，若仍触顶会明确失败并保留旧稿，避免静默截断。",
    "点击首页转译后，小恐龙从准备阶段立即出现，不再等待首个非零进度。",
    "阅读页看译文新增独立恐龙进度条，等待模型时往返滑行，获得分段进度后按百分比前进。",
    "中文研究整理新增 120 秒请求超时和一次自动重试，网络或模型服务异常不会长期卡住。",
    "等待模型时底部踩滑板小恐龙会持续往返，并明确显示生成中或重试中。",
    "整理失败后保留具体错误和失败计数，不再被后续卡片翻译状态覆盖。",
    "RSS 简介兜底稿禁止补写外部背景、财务口径或虚构信源，并持续标明非完整转录。",
    "阅读模式改为深读结构：全文主线、完整核心要点、关键数据与案例、中文整理稿和后续追踪。",
    "核心要点不再限制条目数或做价值分级；移除‘与我的研究相关’，后续追踪最多保留 3 项。",
    "新增页内导航和窄栏深读排版，改善字号、行距、段距、标题层级和内容对比度。",
    "Markdown 阅读器现在保留列表层级并合并续行，换行后自动与条目正文对齐。",
    "旧纪要无需重新生成：打开时会自动转换旧栏目、隐藏无用模块并将追踪问题限制为前 3 项。",
    "YouTube 音频下载恢复自动选择可用客户端，解决 web 客户端只返回分镜图而无可下载音频的问题。",
    "底部刷新与任务进度条升级为圆角胶囊轨道，蓝色进度头不再显示生硬方块。",
    "刷新时新增踩荧光滑板的像素小恐龙往返动画，腿部与滑板轮子带双帧动态。",
    "下载、转录和整理任务中，小恐龙会跟随真实进度沿圆角蓝条前进。",
    "今日雷达分类按钮新增明确选中态；切换 AI、AI创业、天气或养生后，当前按钮立即变黑。",
    "栏目刷新期间新增顶部状态标识和流动进度条，成功、失败及本地缓存回退均有清晰提示。",
    "今日雷达标题同步显示当前分类，避免把天气、AI 与综合雷达混淆。",
    "人物监控卡片补齐真实节目封面，与今日雷达使用相同的圆角图片规格和本地缓存。",
    "Apple Podcasts、RSS 单集/频道图片、YouTube 缩略图与访谈档案页图片按优先级自动回填。",
    "封面后台并发下载，不阻塞人物页；图片不可用时保留 EM/DP 人物占位图。",
    "人物监控页按今日雷达统一标题栏、人物切换、操作区、蓝色来源摘要与横向节目卡片。",
    "移除人物页四块大指标卡，收录、信源、已核验档案和误报统计合并为紧凑摘要。",
    "人物页按钮改用首页同款蓝色、黑色与白色圆角按钮，解决 macOS 原生灰色按钮割裂感。",
    "人物监控页升级为首页同款圆角卡片、轻阴影、人物主题与分层来源标签。",
    "Elon Musk 新增公开访谈档案源，补齐 The Economist、Hannity、SpaceX 技术访谈等最新内容。",
    "马斯克访谈按已核验档案、优质固定源、目录候选分层排序，最新访谈优先显示。",
    "新增“AI创业”独立雷达：聚焦创始人故事、0 到 1、获客与商业化，并过滤纯 AI 产业快讯。",
    "新增“极速云端”转录模式：长音频切片后默认 4 路并发调用阿里云 ASR。",
    "云端片段按原顺序合并，边界保留 1.5 秒重叠并自动去除重复文字。",
    "每个成功片段立即保存断点；云端失败时仅由本地 Whisper 补齐缺失片段。",
    "设置页可随时切换“本地优先”和“极速云端”，并发安全上限为 8 路。",
    "英文完整转录新增“看译文”，按原文逐段生成中文全文并缓存，适合深度阅读。",
    "新增“播放译文”，本机 Qwen3-TTS 会分段合成连续音频，长文不再只朗读开头。",
    "译文、译文音频进入资料库；切换文章时同步切换播放缓存，避免误播上一篇。",
    "应用名称正式统一为 Podcast Radar；原有资料库、设置、下载和系统权限继续沿用。",
    "Dylan Patel 扩展至 15 个优质固定 RSS，并加入 Apple Podcasts 跨节目发现。",
    "Elon Musk 扩展至 10 个优质固定 RSS，并补入 Dwarkesh、JRE、All-In、Moonshots 等来源。",
    "人物监控刷新改为并发抓取，展示上限从 5 条提升至 40 条并保留历史。",
    "优质固定源优先显示；目录搜索结果标为待核验候选，避免第三方提及冒充本人发言。",
    "Summer 耳机增加金色端点接收天线和轻量信号波纹。",
    "雷达指针升级为发光轴心、定向扫描针、外圈落点和半透明扫描扇面。",
    "首页链接输入改为按需展开，减少低频功能占用空间。",
    "侧栏新增“我的精选”；转录任务详情收纳到“更多”菜单。",
    "播客列表卡片加入适度圆角、细描边和轻阴影。",
    "启用 Summer 白企鹅新图标，加入耳机、麦克风和播客广播环。",
    "本地优先模式继续使用 faster-whisper；本地失败且有 Key 时调用阿里云补齐。",
    "未配置阿里云 Key 仍可完成本地 TXT 转录；中文纪要会明确跳过。"
  ],
  "signature": ""
}
EOF

cp "$ROOT_DIR/README.md" "$PACKAGE_DIR/"
cp "$ROOT_DIR/README_MAC.md" "$PACKAGE_DIR/"
mkdir -p "$PACKAGE_DIR/docs"
cp "$ROOT_DIR/docs/EMBODIED_INTELLIGENCE.md" "$PACKAGE_DIR/docs/"

if command -v zip >/dev/null 2>&1; then
  (cd "$PACKAGE_DIR" && zip -qr "$ZIP_PATH" "$APP_BUNDLE" update-manifest.json README.md README_MAC.md docs)
else
  echo "系统未找到 zip 命令，跳过 zip 打包。"
fi

if command -v hdiutil >/dev/null 2>&1; then
  hdiutil create -ov -volname "$APP_DISPLAY_NAME v$VERSION" -fs HFS+ -srcfolder "$PACKAGE_DIR" "$DMG_PATH"
else
  echo "系统未找到 hdiutil，跳过 DMG 打包。"
fi

echo "已完成："
echo "  应用目录: $PACKAGE_DIR"
echo "  ZIP:      $ZIP_PATH"
echo "  DMG:      $DMG_PATH"
