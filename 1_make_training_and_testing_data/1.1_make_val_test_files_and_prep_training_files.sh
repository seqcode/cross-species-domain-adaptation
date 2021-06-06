#!/bin/bash

### NOTE: this script needs to be run once for each species and TF.

### This script begins the process of making all the files that
# models will use for training, validation, and testing. Specifically,
# this script creates the validation and testing set files for a
# given TF and species, creates the binding training set's bound example files,
# and preps the files that will be used to create the rest of the
# training data: the binding training set's unbound example files
# and the species-background training data files.

# This script should be run before the rest of the scripts in this directory.
# This script should be run after the runall script in the 0_* directory.


### Arguments expected:
ROOT=$1  # the directory for the project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"
tf=$2  # one of CTCF, CEBPA, Hnf4a, or RXRA
genome=$3  # one of mm10, hg38

echo "Prepping training datasets for $tf ($genome)."


### Check starting all.all file exists in correct dir

# all.all should be a file of all windows to be used for training/testing
# of models from a genome. This file is specific to each TF because it contains
# a column of binary binding labels (TF bound = 1, else 0). The format is TSV,
# with col 1-3 containing BED-format chromosome, start, stop info,
# and the final column containing binary binding labels for a given TF.
# This file is created by genomic window filtering scripts, so it does not
# contain regions filtered due to unmappability or the ENCODE blacklist.
# Specifically, you should have run the scripts in the 0_* directory
# to create these files.


RAW_DATA_DIR="$ROOT/raw_data/$genome/$tf"
DATA_DIR="$ROOT/data/$genome/$tf"
allfile="$RAW_DATA_DIR/all.all"

if [[ ! -f "$allfile" ]]; then
	echo "File all.all is missing from $RAW_DATA_DIR. Exiting."
	exit 1
fi

# leftover from when we used to have additional columns in all.all
#allbed="$DATA_DIR/all.bed"
#awk -v OFS="\t" '{ print $1, $2, $3, $NF }' "$allfile" > "$allbed"

allbed=$allfile

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





