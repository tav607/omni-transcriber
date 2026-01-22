#!/bin/bash
# Run bot with messagebus group permissions to read Local Bot API Server files
cd /home/paladinllq/omni-transcriber
export PATH="$HOME/.local/bin:$PATH"
exec sg messagebus -c "$HOME/.local/bin/uv run python -m src.main"
