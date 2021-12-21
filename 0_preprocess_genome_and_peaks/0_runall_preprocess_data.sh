#!/bin/bash

# This script pre-processes information from the genomes of both species
# and from the peak calls for each TF within each species to create a file
# of the genome-wide dataset for each TF-species combo. This file will then
# be used by scripts in the 1_* directory for model-specific training/val/test
# data preprocessing. See inside the scripts called here for more explanation.

ROOT=$1  # root directory for project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"


TFS=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
GENOMES=( "mm10" "hg38" )

for genome in "${GENOMES[@]}"; do

  # you'll need to replace these paths with paths to your own genome fastas!
  if [ "$genome" = "mm10" ] ; then
    GENOME_FILE="/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta"
  else
    # assuming hg38
    GENOME_FILE="/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
  fi

  ./0.0_make_windows_files.sh "$ROOT" "$genome" "$GENOME_FILE"

  for tf in "${TFS[@]}"; do
    tf_dir="$genome_dir/$tf"

    # inside tf_dir, you should have put the output of multiGPS peak calling, in "mgps_out/"
    # otherwise you'll get errors related to missing files

    ./0.1_create_full_dataset.sh "$ROOT" "$genome" "$tf"
    
  done
done


