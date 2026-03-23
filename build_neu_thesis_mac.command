#!/bin/bash

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
LOG_PREFIX="[build-mac]"

DEFAULT_PROJECT_DIR="$SCRIPT_DIR"
if [[ ! -f "$DEFAULT_PROJECT_DIR/Thesis.tex" ]]; then
    DEFAULT_PROJECT_DIR="$SCRIPT_DIR/docs/NEU-Thesis-main/NEU-Thesis-main"
fi

if [[ $# -ge 1 ]]; then
    INPUT_PATH="$1"
    if [[ -d "$INPUT_PATH" ]]; then
        PROJECT_DIR="$INPUT_PATH"
    else
        PROJECT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
    fi
else
    PROJECT_DIR="$DEFAULT_PROJECT_DIR"
fi

MAIN_TEX="$PROJECT_DIR/Thesis.tex"
OUTPUT_DIR="$PROJECT_DIR/Tmp"
BACKUP_TEX="$PROJECT_DIR/Thesis.tex.bak.macscript"
STYLE_FILE="$PROJECT_DIR/Style/artratex.sty"
STYLE_BACKUP="$PROJECT_DIR/Style/artratex.sty.bak.macscript"

if [[ ! -f "$MAIN_TEX" ]]; then
    echo "$LOG_PREFIX Could not find Thesis.tex in: $PROJECT_DIR" >&2
    echo "$LOG_PREFIX Drag this script into Terminal and press Enter." >&2
    echo "$LOG_PREFIX Or run: \"$0\" \"/path/to/NEU-Thesis-main\"" >&2
    exit 1
fi

for cmd in xelatex bibtex; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "$LOG_PREFIX Missing command: $cmd" >&2
        echo "$LOG_PREFIX Please install MacTeX or TeX Live first." >&2
        exit 1
    fi
done

# This template ships with a Windows font preset by default.
# On macOS we swap it once so XeLaTeX uses the system CJK fonts.
if grep -q 'fontset=windows' "$MAIN_TEX"; then
    if [[ ! -f "$BACKUP_TEX" ]]; then
        cp "$MAIN_TEX" "$BACKUP_TEX"
    fi
    perl -0pi -e 's/fontset=windows/fontset=mac/g' "$MAIN_TEX"
    echo "$LOG_PREFIX Switched Thesis.tex fontset from windows to mac."
fi

# Some copies of this template point to Tinos/Arimo, which are often missing on macOS.
if [[ -f "$STYLE_FILE" ]] && grep -q 'Tinos' "$STYLE_FILE"; then
    if [[ ! -f "$STYLE_BACKUP" ]]; then
        cp "$STYLE_FILE" "$STYLE_BACKUP"
    fi
    perl -0pi -e 's/\\setmainfont\[NFSSFamily=entextrm,ItalicFont=Tinos Italic\]\{Tinos\}%/\\setmainfont[NFSSFamily=entextrm,BoldFont=Times New Roman Bold,ItalicFont=Times New Roman Italic,BoldItalicFont=Times New Roman Bold Italic]{Times New Roman}%/g; s/\\setsansfont\[NFSSFamily=entextsf,ItalicFont=Arimo Italic\]\{Arimo\}%/\\setsansfont[NFSSFamily=entextsf,BoldFont=Helvetica Bold,ItalicFont=Helvetica Oblique,BoldItalicFont=Helvetica Bold Oblique]{Helvetica}%/g' "$STYLE_FILE"
    echo "$LOG_PREFIX Switched artratex.sty English fonts to macOS defaults."
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -e "$OUTPUT_DIR/Biblio" ]]; then
    ln -s ../Biblio "$OUTPUT_DIR/Biblio"
fi
cd "$PROJECT_DIR"

echo "$LOG_PREFIX Compiling Thesis.tex ..."
xelatex -interaction=nonstopmode -file-line-error --output-directory="$OUTPUT_DIR" Thesis.tex
(
    cd "$OUTPUT_DIR"
    bibtex Thesis
)
xelatex -interaction=nonstopmode -file-line-error --output-directory="$OUTPUT_DIR" Thesis.tex
xelatex -interaction=nonstopmode -file-line-error --output-directory="$OUTPUT_DIR" Thesis.tex

PDF_PATH="$OUTPUT_DIR/Thesis.pdf"
if [[ -f "$PDF_PATH" ]]; then
    echo "$LOG_PREFIX Done: $PDF_PATH"
else
    echo "$LOG_PREFIX Compilation finished, but PDF was not found." >&2
    exit 1
fi
