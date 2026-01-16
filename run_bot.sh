#!/bin/bash
# Run bot with messagebus group permissions to read Local Bot API Server files
cd /home/paladinllq/omni-transcriber
exec sg messagebus -c "uv run python -m src.main"
