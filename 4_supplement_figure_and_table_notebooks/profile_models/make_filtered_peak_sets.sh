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

    # first, to figure out which peaks were filtered out, we overlap
    # the original peak file with the filtered window set
    bedtools intersect -u -wa -a "$raw_peaks" -b "$processed_peaks" > "$filtered_peaks"

    # separate out the test sets: chromosome 1 (validation) and chromosome 2 (test)
    grep "chr1	" "$filtered_peaks" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr1.bed"
    grep "chr2	" "$filtered_peaks" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr2.bed"

    # then figure out what is left for the training set by removing the test peaks from the whole peakset
    cat "$ROOT/data/${genome}/${tf}/filtered_peaks_chr1.bed" "$ROOT/data/${genome}/${tf}/filtered_peaks_chr2.bed" > "$ROOT/data/${genome}/${tf}/tmp.bed"
    bedtools intersect -v  -a "$filtered_peaks" -b "$ROOT/data/${genome}/${tf}/tmp.bed" > "$ROOT/data/${genome}/${tf}/filtered_peaks_chr3toY.bed"
  done
done

