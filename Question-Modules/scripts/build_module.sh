#!/usr/bin/env bash
set -euo pipefail
m=${1:?module directory required}
for f in tutorial slides discussion_questions project_ideas understanding_check; do
  (cd "$m" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$f.tex" >/tmp/dls-latex-${f}.log)
done
