#!/bin/bash

# This script will run all the adjacent scripts to set up datasets for model
# training and testing. See inside individual scripts for their purpose.

# NOTE: These scripts have specific directory structure expectations, and in
# particular, they require the "all.all" files created by earlier scripts.


tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
genomes=( "mm10" "hg38" )

ROOT=$1  # root directory for project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"

# One script is a special exception -- we need to create some repeat files
# for the human genome only, and the script needs an all.all file (produced by
# previous steps) to do it. So we can pick any TF's file to pass in.
random_tf="CTCF"  # this can be any TF you've got an all.all file for
allall_file="$ROOT/raw_data/hg38/${random_tf}/all.all"

# the script 1.4_make_noSINE_files_for_epochs.sh below requires the files made by
# this script
./1.0_make_repeat_files.sh "$ROOT" "$allall_file"


for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		echo "Setting up training data for ${tf} + ${genome}..."
		./1.1_make_val_test_files_and_prep_training_files.sh "$ROOT" "$tf" "$genome"  || exit 1
		./1.2_make_neg_window_files_for_epochs.sh "$ROOT" "$tf" "$genome"  || exit 1
	done

	# this script loops over the genomes internally,
    # because it needs to look at data from both at the same time
	./1.3_make_species_files_for_epochs.sh "$ROOT" "$tf"  || exit 1

    # this script only runs on the human genome data
	./1.4_make_noSINE_files_for_epochs.sh "$ROOT" "$tf"  || exit 1
done
