#!/bin/sh
# Fetch the corpora used by the analysis. Run from the repo root:
#   sh scripts/fetch_data.sh
set -e
mkdir -p data
cd data

# Zandbergen-Landini transliteration of the Voynich MS (EVA, IVTFF 2.0)
# Source and documentation: http://www.voynich.nu/transcr.html
curl -sS -O "http://www.voynich.nu/data/ZL3b-n.txt"

# Takahashi transliteration (optional cross-check corpus)
curl -sS -O "http://www.voynich.nu/data/IT2a-n.txt"

# English control text: Culpeper, "The Complete Herbal" (1653),
# Project Gutenberg #49513 (public domain)
curl -sS -o culpeper.txt "https://www.gutenberg.org/files/49513/49513-0.txt"

echo "done:"
ls -la ZL3b-n.txt IT2a-n.txt culpeper.txt
