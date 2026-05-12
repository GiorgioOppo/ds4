#!/usr/bin/env bash
# Regenerate DeepSeekV4Pro.xcodeproj from project.yml.
#
# Run this once after `brew install xcodegen`, and any time you edit
# project.yml or rename / add / remove source files outside of the
# Sources/DeepSeekUI tree XcodeGen picks up automatically.

set -euo pipefail

if ! command -v xcodegen >/dev/null 2>&1; then
    cat >&2 <<'EOF'
xcodegen is not on PATH. Install it with:

    brew install xcodegen

If you don't have Homebrew, see https://brew.sh — or build from source:

    git clone https://github.com/yonaskolb/XcodeGen.git
    cd XcodeGen && make install

EOF
    exit 1
fi

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

xcodegen generate

cat <<'EOF'

DeepSeekV4Pro.xcodeproj generated. Open it with:

    open DeepSeekV4Pro.xcodeproj

then select the "DeepSeekUI" scheme and ⌘R.

Re-run this script after editing project.yml or after restructuring
the source tree (XcodeGen scans Sources/DeepSeekUI but caches it in
the pbxproj — a rerun keeps the project in sync).

EOF
