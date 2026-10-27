#!/bin/sh
# Rebuild the course zip friends download (everything needed to do workbooks locally).
set -e
cd "$(dirname "$0")/.."
mkdir -p deploy/download && rm -f deploy/download/ai-from-scratch.zip
zip -qr deploy/download/ai-from-scratch.zip index.html assets lessons workbooks data course requirements.txt README.md \
  -x '*/__pycache__/*' '*.pyc' 'data/tinygpt*'
ls -lh deploy/download/ai-from-scratch.zip
