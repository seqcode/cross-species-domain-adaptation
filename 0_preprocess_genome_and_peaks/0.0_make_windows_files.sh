#!/bin/bash

set -e

ROOT=$1  # the project directory (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"
genome=$2  # one of mm10, hg38
GENOME_FILE=$3  # the full path of your whole-genome fasta file for the genome you input as $2


# this directory structure was set up by ../setup_directories_and_download_files.sh
RAW_DATA_DIR="$ROOT/raw_data/${genome}"

# see ../setup_directories_and_download_files.sh for where this blacklist file is downloaded from
BLACKLIST_BED_FILE="$RAW_DATA_DIR/${genome}.blacklist.bed"

echo "Creating windows files for ${genome} genome."

# make bed-formatted file (example line: "chr1 0 500") of all windows, given length of chromosomes and window dimensions
echo "Writing windows to bed file..."

# This script assumes the chromosome sizes file chrom.sizes is in $RAW_DATA_DIR
# The output is a bed file called windows.unfiltered.bed
python _make_windows_bed.py "$RAW_DATA_DIR" 


echo "Getting genomic sequences for all regions to filter unresolved sequence regions..."
# This step...
# 1. Gets the sequence for each window in the bed file, returned in bed format
# 2. Filters out any lines with N (any bases in the genome that are unknown)
# 3. Removes the sequence info from each line (returning to 3-column bed format)
# 4. Sorts the lines in the file in standard order

bedtools getfasta -fi "$GENOME_FILE" -bed "$RAW_DATA_DIR/windows.unfiltered.bed" -bedOut | grep -v "n" | grep -v "N" | awk -v OFS="\t" '{print $1, $2, $3 }' | sort -k1,1 -k2,2n > "$RAW_DATA_DIR/windows.noN.bed" || exit 1


# filter out regions of windows.bed that are blacklisted
echo "Filtering out ENCODE-blacklisted regions from bed file..."

bedtools intersect -a "$RAW_DATA_DIR/windows.noN.bed" -b "$BLACKLIST_BED_FILE" -v > "$RAW_DATA_DIR/windows.bed" || exit 1

rm "$RAW_DATA_DIR/windows.unfiltered.bed"
rm "$RAW_DATA_DIR/windows.noN.bed"


# finally, we get the coverage of all the windows by mappability tracks
# windows with less than 80% of their sequence mappable by 36-mers are filtered out
# this is to prevent false-negative peak calls in regions a TF could bind to, but reads cannot map to
bedtools coverage -sorted -a "$RAW_DATA_DIR/windows.bed" -b "$RAW_DATA_DIR/k36.umap.bed" | awk -v OFS="\t" '$NF >= 0.8' > "$RAW_DATA_DIR/k36.umap.windows_gt0.8cov.bed"


echo "Done."

exit 0

