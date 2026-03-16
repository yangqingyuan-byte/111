#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_TEX="Thesis.tex"
OUTPUT_DIR="Tmp"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Compiling $MAIN_TEX ..."
xelatex --output-directory="$OUTPUT_DIR" "$MAIN_TEX"
xelatex --output-directory="$OUTPUT_DIR" "$MAIN_TEX"

echo "Done."
echo "PDF: $ROOT_DIR/$OUTPUT_DIR/Thesis.pdf"
