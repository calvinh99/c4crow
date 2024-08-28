#!/bin/bash
set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: $0 <source_directory>"; exit 1; }

src=$(realpath "$1")
out="claude_projects_upload"

[ -d "$src" ] || { echo "Error: Source directory '$src' does not exist."; exit 1; }

rm -rf "$out/*"
mkdir -p "$out"

find "$src" -type f -name "*.py" ! -name "__init__.py" | while read -r file; do
    rel=${file#$src/}
    new=$(echo "${rel%.*}" | tr '/' '.')
    cp "$file" "$out/$new.py"
    echo "Copied: $rel -> $new.py"
done

[ "$(ls -A "$out")" ] && echo "Export complete. Files in '$out'." || echo "No .py files found."