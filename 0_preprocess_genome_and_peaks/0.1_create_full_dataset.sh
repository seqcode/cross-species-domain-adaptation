#!/bin/bash


# This script uses the output from make_windows_file.sh as well as peak calls
# to generate the full dataset for a given species and TF.
# The output is a file called all.all in the raw_data directory --
# IMPORTANT -- once you run this script, if everything went okay, you need to
# move the all.all file from raw_data/${species}/${tf} to data/${species}/${tf}.
# This is to avoid you accidentally overwriting your master dataset file.




# Expected arguments:
genome=$1  # one of hg38, mm10
tf=$2  # CTCF, CEBPA, Hnf4a, or RXRA

DATA_DIR="/users/kcochran/projects/domain_adaptation/raw_data/${genome}/${tf}"

# see setup.sh to download this file
BLACKLIST_FILE="/users/kcochran/projects/domain_adaptation/raw_data/${genome}/${genome}.blacklist.bed"

# see make_windows_files.sh to create these files
WINDOWS_FILE="/users/kcochran/projects/domain_adaptation/raw_data/${genome}/windows.bed"
UMAP_COV_FILE="/users/kcochran/projects/domain_adaptation/raw_data/${genome}/k36.umap.windows_gt0.8cov.bed"

run_make_tf_labels=true  # can set to false if you've already run before


echo "Using genome ${genome} and TF ${tf}."


if [ $run_make_tf_labels = true ] ; then
  echo "Re-making TF labels..."
  # output of this will be a TFlabels.bed file
  ./make_tf_labels.sh $genome $tf || exit 1
fi

tf_labels_file="$DATA_DIR/binding_labels.bed"



echo "Generating full dataset..."

# combine bed windows and binding labels into one file
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

# fix file formatting
grep -E "chr[0-9XY]+" "$DATA_DIR/all.noUM.tmp.all" | sed -E 's/	/:/' | sed -E 's/	/-/' | sed -E 's/chr[0-9XY]+	[0-9]+	[0-9]+	//g' | sed -E 's/[:-]/	/g'> "$DATA_DIR/all.all"
if [ ! -s "$DATA_DIR/all.all" ]; then
  echo "Error: failed at final step."
  exit 1
fi

# cleanup
rm "$DATA_DIR/all.bedsort.tmp.all" "$DATA_DIR/all.noBL.tmp.all" "$DATA_DIR/all.noUM.tmp.all"

lines=$(wc -l < "$DATA_DIR/all.all")

echo "Done! Whole genome file (all.all) contains ${lines} windows."

exit 0


