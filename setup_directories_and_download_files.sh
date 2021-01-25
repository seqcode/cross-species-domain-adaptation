#!/bin/bash

# This script creates the directories that will be used by subsequent scripts.
# It also downloads a few files needed in the other scripts.
# Run this script first!


# this is the directory you want all of the project to be contained in
ROOT="/users/kcochran/projects/domain_adaptation"

# where data will be as it is being pre-processed
RAW_DATA_DIR="$ROOT/raw_data"

mkdir -p "$RAW_DATA_DIR"


TFS=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
GENOMES=( "mm10" "hg38" )

for genome in "${GENOMES[@]}"; do
  genome_dir="$RAW_DATA_DIR/$genome"

  mkdir -p "$genome_dir"

  ln -s "/users/kcochran/genomes/$genome.chrom.sizes" "$genome_dir/chrom.sizes"

  for tf in "${TFS[@]}"; do
    tf_dir="$genome_dir/$tf"
    mkdir -p "$tf_dir"
    
    # inside tf_dir, you should put the output of multiGPS, in "mgps_out/"

  done
done

# where data will be once it is processed, as it is being prepped for model training/testing
PROCESSED_DATA_DIR="$ROOT/data"

mkdir -p "$PROCESSED_DATA_DIR"

for genome in "${GENOMES[@]}"; do
  genome_dir="$PROCESSED_DATA_DIR/$genome"

  mkdir -p "$genome_dir"

  for tf in "${TFS[@]}"; do
    tf_dir="$genome_dir/$tf"
    mkdir -p "$tf_dir"
  done
done



# download and unzip ENCODE blacklist files and UMAP tracks

cd "$RAW_DATA_DIR/hg38"
wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/hg38-human/hg38.blacklist.bed.gz
gunzip hg38.blacklist.bed.gz

wget https://bismap.hoffmanlab.org/raw/hg38/k36.umap.bed.gz
gunzip k36.umap.bed.gz


cd "$RAW_DATA_DIR/mm10"
wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/mm10-mouse/mm10.blacklist.bed.gz
gunzip mm10.blacklist.bed.gz

wget https://bismap.hoffmanlab.org/raw/mm10/k36.umap.bed.gz
gunzip k36.umap.bed.gz

echo "Done."

exit 0

