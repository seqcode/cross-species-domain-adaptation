#!/bin/bash

# This script will run all the adjacent scripts to set up datasets for model
# training and testing.

# Note that you will need to run make_repeat_files.sh yourself first --
# make_noSINE_files_for_epochs.sh relies on the output from that script.

# These scripts have specific directory structure expectations, and in
# particular, they require the "all.all" files created by earlier scripts.



tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
genomes=( "mm10" "hg38" )


for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		echo "$tf $genome"
		./setup_training_data.sh "$tf" "$genome"  || exit 1
		./make_neg_window_files_for_epochs.sh "$tf" "$genome"  || exit 1
	done
	# these scripts only run on the human genome
	./make_species_files_for_epochs.sh "$tf"  || exit 1
	./make_noSINE_files_for_epochs.sh "$tf"  || exit 1
done
