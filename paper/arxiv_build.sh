#!/usr/bin/env bash
# Build a clean, self-contained arXiv submission tarball from paper/.
# arXiv compiles by running pdflatex; the bibliography is inline (\thebibliography),
# so no .bib/.bbl is needed. Only the 6 figures actually \includegraphics'd are bundled.
# Usage:  bash paper/arxiv_build.sh   ->   paper/arxiv_gpufsm.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/arxiv_gpufsm"
rm -rf "$OUT" && mkdir -p "$OUT/figures"
cp "$HERE/gpufsm.tex" "$HERE/IEEEtran.cls" "$OUT/"
for f in fig_costmodel_fit fig_throughput_vs_states fig_worklist_speedup \
         fig_memory_ablation fig_abstraction_regret fig_dfa_memory_bound; do
  cp "$HERE/figures/$f.pdf" "$OUT/figures/"
done
# clean-room compile check (twice for refs)
( cd "$OUT" && pdflatex -interaction=nonstopmode -halt-on-error gpufsm.tex >/dev/null 2>&1 \
            && pdflatex -interaction=nonstopmode -halt-on-error gpufsm.tex >/dev/null 2>&1 ) \
  && echo "clean-room compile OK ($(cd "$OUT" && pdfinfo gpufsm.pdf 2>/dev/null | awk '/Pages/{print $2" pages"}'))" \
  || { echo "ERROR: clean-room compile failed"; exit 1; }
# strip aux/log/pdf from the tarball (arXiv regenerates them)
( cd "$OUT" && rm -f gpufsm.aux gpufsm.log gpufsm.out gpufsm.pdf )
tar -C "$OUT" -czf "$HERE/arxiv_gpufsm.tar.gz" .
echo "wrote $HERE/arxiv_gpufsm.tar.gz"
tar -tzf "$HERE/arxiv_gpufsm.tar.gz"
