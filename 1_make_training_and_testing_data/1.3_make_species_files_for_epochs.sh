#!/bin/bash

### NOTE: this script depends on the output from 1.1_make_val_test_files_and_prep_training_files.sh

### DA models need "species-background" training data for their discriminator
# half to train on. This script makes those files, in similar style to the
# 1.2_make_neg_window_files_for_epochs.sh script, which makes the unbound site
# half of the normal training set.

# This script first calculates how large the species-background datasets need
# to be. This will be equal to the size of the training set (bound + unbound
# examples) that is larger between the two species (during training, to keep
# the losses equally balanced, the network is shown the same # of examples of 
# each type of training data). Since this is equal to double the larger # of
# bound training examples, and the species-background training data is made up
# of 50% data from each species, then we can simplify: we need X training
# examples from each species, where X = max(# bound sites used in training)
# across the two species.

# Distinct examples will be used each epoch, to improve diversity of data seen
# by the models, so this script will create one new file for each epoch, for 
# each independent replicate model run.

# RUNS: the number of replicate model training runs to make data for.
RUNS=5
# EPOCHS: the number of epochs the models will train for.
# Files will be generated for each epoch.
EPOCHS=15

# Expecting 2 arguments: the root directory for the project,
#    and the TF to process data for (CTCF, CEBPA, Hnf4a, RXRA)
ROOT=$1
#ROOT="/users/kcochran/projects/domain_adaptation"
tf=$2

# which species is 1 vs. 2 does not matter
genome1="mm10"
genome2="hg38"

if [[ -z "$tf" ]]; then
  echo "Missing an argument. Required: TF."
  exit 1
fi

echo "Prepping shuffled species-background datasets for $tf."

DATA_DIR_s1="$ROOT/data/$genome1/$tf"
# these files were made by the script 1.1_make_val_test_files_and_prep_training_files.sh
TRAIN_FILE_s1="$DATA_DIR_s1/chr3toY_shuf.bed"
POS_FILE_s1="$DATA_DIR_s1/chr3toY_pos_shuf.bed"
bound_windows_s1=`wc -l < "$POS_FILE_s1"`

DATA_DIR_s2="$ROOT/data/$genome2/$tf"
# these files were made by the script 1.1_make_val_test_files_and_prep_training_files.sh
TRAIN_FILE_s2="$DATA_DIR_s2/chr3toY_shuf.bed"
POS_FILE_s2="$DATA_DIR_s2/chr3toY_pos_shuf.bed"
bound_windows_s2=`wc -l < "$POS_FILE_s2"`

### Created files need to be as large as the largest set of bound sites (of either species)
# So, we measure the size of both species' bound site sets
# and go with the larger number.

if [[ "$bound_windows_s1" -gt "$bound_windows_s2" ]]; then
	bound_windows="$bound_windows_s1"
else
	bound_windows="$bound_windows_s2"
fi


### For first species

# Process of getting distinct randomly selected examples for each epoch is the
# same as in the script make_neg_window_files_for_epochs.sh.

tmp_shuf_file="$DATA_DIR_s1/chr3toY_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
	shuf "$TRAIN_FILE_s1" > "$tmp_shuf_file"
	for ((epoch=1;epoch<=EPOCHS;epoch++)); do
		head_line_num=$(( bound_windows * epoch ))
		echo "For epoch $epoch, head ends at line $head_line_num"
		epoch_run_filename="$DATA_DIR_s1/chr3toY_shuf_run${run}_${epoch}E.bed"
		head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"
	done
done

rm "$tmp_shuf_file"


### For second species (code identical to first species)

tmp_shuf_file="$DATA_DIR_s2/chr3toY_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
    shuf "$TRAIN_FILE_s2" > "$tmp_shuf_file"
    for ((epoch=1;epoch<=EPOCHS;epoch++)); do
        head_line_num=$(( bound_windows * epoch ))
        echo "For epoch $epoch, head ends at line $head_line_num"
        epoch_run_filename="$DATA_DIR_s2/chr3toY_shuf_run${run}_${epoch}E.bed"
        head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"
    done
done

rm "$tmp_shuf_file"


echo "Done!"

exit 0


