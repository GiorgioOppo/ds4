#!/bin/sh
# Assemble DwarfStar.app from the SwiftPM release build.
#
# Produces build/DwarfStar.app with:
#   Contents/MacOS/DwarfStar          the release executable
#   Contents/Info.plist               bundle metadata
#   Contents/Resources/metal/*.metal  Metal kernel sources (REQUIRED at runtime)
#   Contents/Resources/bin/ds4*       helper binaries, if already built (optional)
#   Contents/Resources/download_model.sh, speed-bench/   optional helpers
#
# The app is ad-hoc code-signed so it runs locally. For distribution, re-sign
# with a Developer ID identity and notarize (see README).
set -eu

# Resolve paths relative to this script: PKG=DS4-gui/packaging, GUI=DS4-gui, ROOT=project.
PKG_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GUI_DIR=$(CDPATH= cd -- "$PKG_DIR/.." && pwd)
ROOT_DIR=$(CDPATH= cd -- "$GUI_DIR/.." && pwd)

APP_NAME=DwarfStar
BUILD_DIR="$GUI_DIR/build"
APP="$BUILD_DIR/$APP_NAME.app"
SIGN_IDENTITY="${DS4_SIGN_IDENTITY:--}"   # default: ad-hoc

echo "==> Building engine static library"
( cd "$GUI_DIR" && make engine )

echo "==> Building SwiftPM release"
( cd "$GUI_DIR" && swift build -c release --product "$APP_NAME" )

EXE="$GUI_DIR/.build/release/$APP_NAME"
if [ ! -x "$EXE" ]; then
    echo "error: release executable not found at $EXE" >&2
    exit 1
fi

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin"
cp "$EXE" "$APP/Contents/MacOS/$APP_NAME"
cp "$PKG_DIR/Info.plist" "$APP/Contents/Info.plist"

# Harden the bundle Info.plist: a SwiftUI macOS app launches but never shows a
# window without NSPrincipalClass=NSApplication. The packaging/Info.plist source
# is sometimes reformatted (keys dropped), so set the essentials here too.
PB=/usr/libexec/PlistBuddy
PLIST="$APP/Contents/Info.plist"
set_key() { $PB -c "Set :$1 $2" "$PLIST" 2>/dev/null || $PB -c "Add :$1 string $2" "$PLIST"; }
set_key NSPrincipalClass NSApplication
set_key CFBundleExecutable "$APP_NAME"
set_key CFBundleName "$APP_NAME"
set_key CFBundleIdentifier org.ds4.dwarfstar
set_key LSMinimumSystemVersion 14.0
$PB -c "Set :NSHighResolutionCapable true" "$PLIST" 2>/dev/null \
    || $PB -c "Add :NSHighResolutionCapable bool true" "$PLIST"

# Required: the Metal kernel sources, compiled at runtime by the engine. These
# are vendored inside the DS4-gui project (GUI_DIR/metal), so the bundle does
# not depend on the upstream tree.
cp -R "$GUI_DIR/metal" "$APP/Contents/Resources/metal"

# Optional app icon.
if [ -f "$PKG_DIR/AppIcon.icns" ]; then
    cp "$PKG_DIR/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

# Optional: bundle helper binaries (for the server/bench/diagnostics panels).
for b in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
    if [ -x "$ROOT_DIR/$b" ]; then
        cp "$ROOT_DIR/$b" "$APP/Contents/Resources/bin/$b"
    fi
done

# Optional: model downloader and the benchmark prompt corpus.
[ -f "$ROOT_DIR/download_model.sh" ] && cp "$ROOT_DIR/download_model.sh" "$APP/Contents/Resources/download_model.sh"
if [ -f "$ROOT_DIR/speed-bench/promessi_sposi.txt" ]; then
    mkdir -p "$APP/Contents/Resources/speed-bench"
    cp "$ROOT_DIR/speed-bench/promessi_sposi.txt" "$APP/Contents/Resources/speed-bench/"
fi

echo "==> Code signing ($SIGN_IDENTITY)"
# Sign nested binaries first, then the app.
find "$APP/Contents/Resources/bin" -type f -perm +111 -exec \
    codesign --force --timestamp=none --sign "$SIGN_IDENTITY" {} \; 2>/dev/null || true
codesign --force --deep --timestamp=none --sign "$SIGN_IDENTITY" "$APP"

echo "==> Verifying"
codesign --verify --verbose "$APP" || true

echo "Done: $APP"
