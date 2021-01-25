#!/bin/bash



genome=$1

if [ "$genome" = "mm10" ] ; then
  GENOME_FILE="/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta"
else
  # assuming hg38
  GENOME_FILE="/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
fi


RAW_DATA_DIR="/users/kcochran/projects/domain_adaptation/raw_data/${genome}/"
BLACKLIST_BED_FILE="$RAW_DATA_DIR/${genome}.blacklist.bed"

echo "Creating windows files for ${genome} genome."

# make bed-formatted file (example line: "chr1 0 500") of all windows, given length of chromosomes and window dimensions
echo "Writing windows to bed file..."

# This script assumes the chromosome sizes file chrom.sizes is in $RAW_DATA_DIR
# The output is a bed file called windows.unfiltered.bed
python make_windows_bed.py "$genome"


echo "Getting genomic sequences for all regions..."
# This step...
# 1. Gets the sequence for each window in the bed file, returned in bed format
# 2. Filters out any lines with N (any bases in the genome that are unknown)
# 3. Removes the sequence info from each line (returning to 3-column bed format)
# 4. Sorts the lines in the file in standard order

bedtools getfasta -fi "$GENOME_FILE" -bed "$RAW_DATA_DIR/windows.unfiltered.bed" -bedOut | grep -v "n" | grep -v "N" | awk -v OFS="\t" '{print $1, $2, $3 }' | sort -k1,1 -k2,2n > "$RAW_DATA_DIR/windows.noN.bed" || exit 1


# filter out regions of windows.bed that are blacklisted
echo "Filtering out blacklisted regions from bed file..."

bedtools intersect -a "$RAW_DATA_DIR/windows.noN.bed" -b "$BLACKLIST_BED_FILE" -v > "$RAW_DATA_DIR/windows.bed" || exit 1

#rm "$RAW_DATA_DIR/windows.unfiltered.bed"
#rm "$RAW_DATA_DIR/windows.noN.bed"


# finally, we will later need coverage of all the windows by mappability tracks
bedtools coverage -sorted -a "$RAW_DATA_DIR/windows.bed" -b "$RAW_DATA_DIR/k36.umap.bed" | awk -v OFS="\t" '$NF >= 0.8' > "$RAW_DATA_DIR/k36.umap.windows_gt0.8cov.bed"


echo "Done"
