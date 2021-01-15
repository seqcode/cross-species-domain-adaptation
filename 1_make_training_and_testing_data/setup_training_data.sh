#!/bin/bash

ROOT="/users/kcochran/projects/domain_adaptation"

### Run me first!!!!
### NOTE: this script needs to be run once for each species and TF.



### Parse args
# Arguments expected: TF (CTCF, CEBPA, Hnf4a, or RXRA) and genome (mm10, hg38)

tf=$1
genome=$2

if [[ -z "$tf" || -z "$genome" ]]; then
  echo "Missing an argument. Required: TF, species."
  exit 1
fi

echo "Prepping training datasets for $tf ($genome)."



### Check starting all.all file exists in correct dir
# all.all should be a file of all windows to be used for training/testing
# of models from a genome. This file is specific to each TF because it contains
# a column of binary binding labels (TF bound = 1, else 0). The format is TSV,
# with col 1-3 containing BED-format chromosome, start, stop info,
# and the final column containing binary binding labels for a given TF.
# This file is created by genomic window filtering scripts, so it does not
# contain regions filtered due to unmappability or the ENCODE blacklist.


DATA_DIR="$ROOT/data/$genome/$tf"
allfile="$DATA_DIR/all.all"

if [[ ! -f "$allfile" ]]; then
	echo "File all.all is missing from $DATA_DIR. Exiting."
	exit 1
fi

allbed="$DATA_DIR/all.bed"
awk -v OFS="\t" '{ print $1, $2, $3, $NF }' "$allfile" > "$allbed"

total_windows=`wc -l < "$allbed"`
echo "Total windows: $total_windows"




### Make chr1/2 validation and test sets
# This script creates a validation set of 1 million randomly sampled
# (without replacement) windows from chromosome 1, and a test set of all
# windows on chromosome 2.


chr1file="$DATA_DIR/chr1_random_1m.bed"
chr2file="$DATA_DIR/chr2.bed"

grep -F "chr1	" "$allbed" | shuf | head -n1000000 > "$chr1file"
grep -F "chr2	" "$allbed" > "$chr2file"

chr2_windows=`wc -l < "$chr2file" `
echo "Test set windows: $chr2_windows"

# Sanity checks

chr1_windows=`wc -l < "$chr1file" `
if [[ "$chr1_windows" != 1000000 ]]; then
	echo "Error: chr1 val set file only has $chr1_windows windows. Exiting."
	exit 1
fi

chr1_chroms=`awk '{ print $1 }' "$chr1file" | sort | uniq | wc -l `
if [[ "$chr1_chroms" != 1 ]]; then
	echo "Error: chr1 val set file contains mutliple chromosomes. Exiting."
	exit 1
fi

chr2_chroms=`awk '{ print $1 }' "$chr2file" | sort | uniq | wc -l `
if [[ "$chr2_chroms" != 1 ]]; then
	echo "Error: chr2 test set file contains mutliple chromosomes. Exiting."
	exit 1
fi



### Get training chromosomes, split into bound/unbound examples
# Here we divide the training examples (examples from chromosomes except 1, 2)
# into bound and unbound examples. In a later script we will sample balanced
# (half bound, half unbound) training datasets from these files.


grep -Ev "chr[12]	" "$allbed" | shuf > "$DATA_DIR/chr3toY_shuf.bed"
awk '$NF == 1' "$DATA_DIR/chr3toY_shuf.bed" | shuf > "$DATA_DIR/chr3toY_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/chr3toY_shuf.bed" | shuf > "$DATA_DIR/chr3toY_neg_shuf.bed"

total_windows=`wc -l < "$DATA_DIR/chr3toY_shuf.bed"`
bound_windows=`wc -l < "$DATA_DIR/chr3toY_pos_shuf.bed"`
unbound_windows=`wc -l < "$DATA_DIR/chr3toY_neg_shuf.bed"`

total=$(( $bound_windows + $unbound_windows ))
if [[ $total != $total_windows ]]; then
	echo "Error: bound + unbound windows does not equal total windows. Exiting."
	exit 1
fi

echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"

echo "Done!"

exit 0





