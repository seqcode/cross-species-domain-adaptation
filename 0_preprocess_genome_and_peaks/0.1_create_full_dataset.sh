#!/bin/bash


# This script uses the output from make_windows_file.sh as well as peak calls
# to generate the full dataset for a given species and TF.
# The output is a file called all.all in the raw_data directory.
# The all.all file contains all of the windows that have survived filtering,
# to potentially be part of the train/val/test data for the model.
# It is bed-formatted, with one additional column containing the binary label
# where 1 corresponds to "a peak overlapped with this window" and 0
# corresponds to "a peak did not overlap with this window".

# IMPORTANT -- once you run this script, if everything went okay, you need to
# move the all.all file from raw_data/${species}/${tf} to data/${species}/${tf}.
# I make you do this to avoid you accidentally overwriting your master dataset file.


# Expected arguments:
ROOT=$1  # the project directory (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"
genome=$2  # one of hg38, mm10
tf=$3  # CTCF, CEBPA, Hnf4a, or RXRA -- consistent with how you've named them elsewhere

DATA_DIR="$ROOT/raw_data/${genome}/${tf}"

# see setup_directories_and_download_files.sh on where this file came from
BLACKLIST_FILE="$ROOT/raw_data/${genome}/${genome}.blacklist.bed"

# the script 0.0_make_windows_files.sh created these files
WINDOWS_FILE="$ROOT/raw_data/${genome}/windows.bed"
UMAP_COV_FILE="$ROOT/raw_data/${genome}/k36.umap.windows_gt0.8cov.bed"


echo "Using genome ${genome} and TF ${tf}."

"Making TF labels..."
./_make_tf_labels.sh "$ROOT" "$genome" "$tf"

# output of the script above
tf_labels_file="$DATA_DIR/binding_labels.bed"


echo "Generating full dataset..."

# combine the bed-formatted columns for windows and the single column of binding labels into one file
paste -d "	" "$WINDOWS_FILE" "$tf_labels_file" > "$DATA_DIR/all.bedsort.tmp.all"
if [ ! -s "$DATA_DIR/all.bedsort.tmp.all" ]; then
  echo "Error: failed at paste command."
  exit 1
fi

# remove any windows that intersect with blacklist regions      ####### OR UNMAPPABLE REGIONS -- changed 5/30/19
bedtools intersect -v -a "$DATA_DIR/all.bedsort.tmp.all" -b "$BLACKLIST_FILE" > "$DATA_DIR/all.noBL.tmp.all"
if [ ! -s "$DATA_DIR/all.noBL.tmp.all" ]; then
  echo "Error: failed at blacklist intersect command."
  exit 1
fi

bedtools intersect -a "$DATA_DIR/all.noBL.tmp.all" -b "$UMAP_COV_FILE" -f 1 -wa > "$DATA_DIR/all.noUM.tmp.all"
if [ ! -s "$DATA_DIR/all.noUM.tmp.all" ]; then
  echo "Error: failed at umap intersect command."
  exit 1
fi

# finally, remove weird chromosomes and fix the file formatting
# no chrM, no chrEBV, and no scaffolds will be used
# specifically, this line removes a redundant set of bed-info columns (chr \t start \t stop) in a slightly hacky way
grep -E "chr[0-9XY]+" "$DATA_DIR/all.noUM.tmp.all" | sed -E 's/	/:/' | sed -E 's/	/-/' | sed -E 's/chr[0-9XY]+	[0-9]+	[0-9]+	//g' | sed -E 's/[:-]/	/g'> "$DATA_DIR/all.all"
if [ ! -s "$DATA_DIR/all.all" ]; then
  echo "Error: failed at final step."
  exit 1
fi

# cleanup -- delete tmp files
rm "$DATA_DIR/all.bedsort.tmp.all" "$DATA_DIR/all.noBL.tmp.all" "$DATA_DIR/all.noUM.tmp.all" "$tf_labels_file" "$WINDOWS_FILE"

lines=$(wc -l < "$DATA_DIR/all.all")

echo "Done! Whole genome file (all.all) contains ${lines} windows."

exit 0


