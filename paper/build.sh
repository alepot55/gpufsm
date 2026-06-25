#!/bin/bash
# Build script for the conference paper
# Usage: ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building conference paper ==="

# Generate figures first
echo "1. Generating figures..."
python3 generate_figures.py

# Compile LaTeX (run twice for references)
echo "2. Compiling LaTeX..."
pdflatex -interaction=nonstopmode conference_101719.tex > /dev/null 2>&1 || true
bibtex conference_101719 > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode conference_101719.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode conference_101719.tex > /dev/null 2>&1 || true

# Check if PDF was created
if [ -f "conference_101719.pdf" ]; then
    echo "3. Success! PDF generated: conference_101719.pdf"
    ls -lh conference_101719.pdf
else
    echo "Error: PDF not generated"
    exit 1
fi

# Clean auxiliary files
echo "4. Cleaning auxiliary files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

echo "=== Done ==="
