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

if [[ -d "$DIST_DIR/$EXECUTABLE_NAME.app" ]]; then
  GENERATED_APP_PATH="$DIST_DIR/$EXECUTABLE_NAME.app"
elif [[ -d "$DIST_DIR/$EXECUTABLE_NAME/$EXECUTABLE_NAME.app" ]]; then
  GENERATED_APP_PATH="$DIST_DIR/$EXECUTABLE_NAME/$EXECUTABLE_NAME.app"
else
  echo "未检测到 PyInstaller 输出的应用包。"
  exit 1
fi

APP_PATH="$DIST_DIR/$APP_BUNDLE"
mv "$GENERATED_APP_PATH" "$APP_PATH"

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

if command -v zip >/dev/null 2>&1; then
  (cd "$PACKAGE_DIR" && zip -qr "$ZIP_PATH" "$APP_BUNDLE" update-manifest.json README.md README_MAC.md)
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
