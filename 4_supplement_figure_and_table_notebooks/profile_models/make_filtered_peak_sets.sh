#!/bin/bash

set -e

ROOT="/users/kcochran/projects/domain_adaptation_nosexchr"

genomes=( "mm10" "hg38" )
tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )

for genome in "${genomes[@]}"; do
  for tf in "${tfs[@]}"; do
    raw_peaks="$ROOT/raw_data/${genome}/${tf}/mgps_out_${tf}.bed"
    processed_peaks="$ROOT/raw_data/${genome}/${tf}/all.all"
    filtered_peaks="$ROOT/data/${genome}/${tf}/filtered_peaks.bed"
    bedtools intersect -u -wa -a "$raw_peaks" -b "$processed_peaks" > "$filtered_peaks"

    grep "chr1	" "$filtered_peaks" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr1.bed"
    grep "chr2	" "$filtered_peaks" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr2.bed"
    cat "$ROOT/data/${genome}/${tf}/filtered_peaks_chr1.bed" "$ROOT/data/${genome}/${tf}/filtered_peaks_chr2.bed" > "$ROOT/data/${genome}/${tf}/tmp.bed"
    bedtools intersect -v  -a "$filtered_peaks" -b "$ROOT/data/${genome}/${tf}/tmp.bed" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr3toY.bed"
  done
done

