#!/bin/bash
# Downloads the OpenCommand per-pitch targets (Tom Kim, open-command.com, CC BY-NC-SA 4.0) for 2024-2026
# from Hugging Face into this folder. Double-click on the Mac; some networks refuse Hugging Face, so the pull
# runs here. Resumable: finished files are skipped.
cd "$(dirname "$0")"
mkdir -p data
for Y in 2024 2025 2026; do
  for F in targets.csv.gz pbp_info.csv.gz command_scores.csv; do
    OUT="data/${Y}_${F}"
    if [ -s "$OUT" ]; then echo "have $OUT"; continue; fi
    echo "pulling $Y/$F ..."
    curl -L --fail --retry 3 -o "$OUT" "https://huggingface.co/datasets/tomdoyo/open-command/resolve/main/$Y/$F" || { echo "FAILED $Y/$F"; rm -f "$OUT"; }
  done
done
echo; echo "done. files:"; ls -la data; shasum -a 256 data/* | cut -c1-16,65- > data/SHA16.txt; cat data/SHA16.txt
echo; echo "You can close this window."
