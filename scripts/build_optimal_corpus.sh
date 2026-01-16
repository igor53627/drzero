#!/bin/bash
# Build optimal ChromaDB corpus for Dr. Zero training
# Target: 95-99% retrieval accuracy

set -e

INPUT_DIR="${1:-/Users/user/pse/000-research/iO-papers/pdfs}"
OUTPUT_DIR="${2:-/Users/user/pse/000-research/iO-papers/optimal_db}"
COLLECTION="papers"

echo "=== Dr. Zero Optimal Corpus Builder ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check dependencies
echo "Checking dependencies..."
pip install -q chromadb sentence-transformers pymupdf tiktoken tqdm

# Backup old DB if exists
if [ -d "$OUTPUT_DIR" ]; then
    BACKUP="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing DB to: $BACKUP"
    mv "$OUTPUT_DIR" "$BACKUP"
fi

# Find PDF directory
if [ ! -d "$INPUT_DIR" ]; then
    # Try to find pdfs subdirectory
    PARENT=$(dirname "$INPUT_DIR")
    if [ -d "$PARENT/pdfs" ]; then
        INPUT_DIR="$PARENT/pdfs"
        echo "Using PDF directory: $INPUT_DIR"
    else
        echo "Error: Cannot find PDF directory"
        exit 1
    fi
fi

# Count PDFs
PDF_COUNT=$(find "$INPUT_DIR" -name "*.pdf" | wc -l)
echo "Found $PDF_COUNT PDF files"

# Build corpus
echo ""
echo "Building corpus with optimal settings..."
python scripts/rechunk_corpus.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --collection-name "$COLLECTION" \
    --test

echo ""
echo "=== Done ==="
echo "New corpus: $OUTPUT_DIR"
echo ""
echo "To use with Dr. Zero on Modal:"
echo "  modal run modal_train.py --action upload --corpus-path $OUTPUT_DIR"
