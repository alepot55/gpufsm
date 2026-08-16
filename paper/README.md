# paper/ — the first paper ("Two Faces of Abstraction Regret")

Two builds of the same work, both self-contained (no `\input`, inline
`\thebibliography`), both compiling with `pdflatex` alone:

| File | What it is |
|---|---|
| `gpufsm.tex` | the 8-page full version; the arXiv/journal source |
| `gpufsm_hpec.tex` | the 6-page IEEE HPEC cut (four figures dropped, Nsight/SOTA inlined) |
| `IEEEtran.cls` | the IEEE class. Duplicated in `paper2/` on purpose: each tree has to be a self-contained tarball for arXiv/HotCRP, so neither can point at the other |
| `figures.py` | regenerates every figure from the committed CSVs in `data/` |
| `arxiv_build.sh` | builds a clean-room arXiv tarball and verifies it compiles |
| `bibliography.bib` | reference data only. No live `.tex` reads it — both use an inline bibliography — but it is kept while the papers are still being revised. `paper2/` has its own `refs.bib` |

The second paper (the "cure" line, ASPLOS) lives in `paper2/`.

## Build

```bash
python paper/figures.py                 # regenerate figures from data/*.csv
pdflatex -interaction=nonstopmode paper/gpufsm.tex   # twice, for the references
bash paper/arxiv_build.sh               # or: tarball + clean-room compile check
```

## Reproducibility

Figures are regenerated **only** from the versioned CSVs under `data/`, never from a
live GPU run, so a checkout reproduces the paper's plots without hardware. The CSVs
themselves come from `scripts/` and `experiments/`; the claim-to-command map is in
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

⚠️ The CSV filenames carry `rtx4070` because that is where the numbers were measured.
Re-running a measurement on another GPU currently writes to the same filename — check
`docs/REPRODUCIBILITY.md` before overwriting anything under `data/`.
