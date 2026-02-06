#!/bin/bash
# Auto-update yt-dlp: upgrade, commit, push, and restart if changed
set -euo pipefail

cd "$(dirname "$0")/.."

uv lock --upgrade-package yt-dlp

if git diff --quiet uv.lock; then
    echo "yt-dlp is already up to date."
    exit 0
fi

uv sync

VERSION=$(uv run yt-dlp --version)
git add uv.lock
git commit -m "Update yt-dlp to ${VERSION}"
git push
pm2 restart omni-transcriber

echo "yt-dlp updated to ${VERSION}, bot restarted."
