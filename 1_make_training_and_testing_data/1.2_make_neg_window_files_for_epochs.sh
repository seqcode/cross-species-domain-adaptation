#!/bin/bash


### This script picks up where 1.1_make_val_test_files_and_prep_training_files.sh
# leaves off. Here, using files made by the previous script, we
# shuffle the unbound sites that our training data can sample from, and then
# sample a distinct set of sites for training each epoch (no unbound site is
# repeated across multiple epochs). The bound sites trained on each epoch are
# the same. The number of unbound sites sampled each epoch is equal to the
# number of bound sites to be used in training (so each epoch's training data
# will be balanced).

# This process will create a file of unbound sites for training specific to
# each epoch. If you opt for too many epochs (or the size of your training 
# datasets is too large), this script may error out when it uses up all 
# possible unbound sites to sample from.

### NOTE: this script needs to be run once for each species and TF.

# Later, we train each model 5 times -- using a different random initialization,
# and using different randomly sampled training data, each time. This is to
# measure the reproducibility of downstream results. If you don't plan to do that,
# you can change the RUN parameter below to 1. Note that you'll need to change
# it everywhere in the code, including in script 1.3 and script 1.4.

# RUNS: the number of replicate model training runs to make data for.
RUNS=5

# EPOCHS: the maximum number of epochs the models will train for.
# Files will be generated for each epoch.
EPOCHS=15


### Arguments expected:
ROOT=$1  # the directory for the project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"
tf=$2  # one of CTCF, CEBPA, Hnf4a, or RXRA
genome=$3  # one of mm10, hg38

echo "Prepping shuffled negative example datasets for $tf ($genome)."

DATA_DIR="$ROOT/data/$genome/$tf"
# these two files were created by the previous script
POS_FILE="$DATA_DIR/chr3toY_pos_shuf.bed"
NEG_FILE="$DATA_DIR/chr3toY_neg_shuf.bed"

bound_windows=`wc -l < "$DATA_DIR/chr3toY_pos_shuf.bed"`
unbound_windows=`wc -l < "$DATA_DIR/chr3toY_neg_shuf.bed"`
echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"


tmp_shuf_file="$DATA_DIR/chr3toY_neg_shuf.tmp"  # reused each iteration

# for each replicate model training run...
for ((run=1;run<=RUNS;run++)); do
    # re-shuffle the data so it's in a different order for this run
	shuf "$NEG_FILE" > "$tmp_shuf_file"

    # for each epoch the model will train for...
	for ((epoch=1;epoch<=EPOCHS;epoch++)); do
        # calculate the # for the last line we want included in this epoch's batch
		head_line_num=$(( bound_windows * epoch ))
		epoch_run_filename="$DATA_DIR/chr3toY_neg_shuf_run${run}_${epoch}E.bed"

        # fetch the chunk of lines that ends at $head_line_num and is $bound_windows long, and write to file
		head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"

		# sanity check -- make sure the number of lines written to this epoch's file is what we expect
		lines_in_file=`wc -l < "$epoch_run_filename"`
		if [[ "$lines_in_file" != "$bound_windows" ]]; then
			echo "Error: incorrect number of lines ($lines_in_file) in file $epoch_run_filename (should be $bound_windows). Exiting."
			exit 1
		fi
	done
done

rm "$tmp_shuf_file"


echo "Done!"

exit 0





