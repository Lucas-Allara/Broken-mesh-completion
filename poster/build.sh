#!/bin/sh
# Builds poster.pdf (two passes: the remember picture nodes need them).
cd "$(dirname "$0")"
export PATH="$HOME/.local/share/TinyTeX/bin/x86_64-linux:$PATH"
lualatex -interaction=nonstopmode -file-line-error poster.tex >/dev/null
lualatex -interaction=nonstopmode -file-line-error poster.tex >/dev/null
grep -E '^\./poster.tex:[0-9]+:|Overfull|undefined' poster.log
rm -f poster.aux poster.log poster.out
