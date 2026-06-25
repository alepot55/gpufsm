# paper/

LaTeX sources and figure pipeline for the publication.

**Status:** migrated from the prior `triton_vs_cuda_fsm` 3-domain study; being
rewritten for the FSM-only, memory-centric contribution (see `../docs/METHODOLOGY.md`).

- `conference_101719.tex`, `sections/`, `bibliography.bib`, `IEEEtran.cls` — paper source.
- `data/benchmarks_final_publication.csv` — prior FSM benchmark results (kept as reference).
- `generate_figures.py` — legacy figure generator (3-domain). To be rewritten to consume the
  versioned `gpufsm sweep` CSV schema so figures rebuild deterministically from committed data only.

Build (once rewritten):

```bash
python paper/generate_figures.py
latexmk -pdf paper/conference_101719.tex
```
