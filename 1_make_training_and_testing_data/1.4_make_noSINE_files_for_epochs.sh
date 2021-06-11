#!/bin/bash

### NOTE: this script depends on files from 1.1_make_val_test_files_and_prep_training_files.sh
# and from 1.0_make_repeat_files.sh.

### This script makes files for training no-SINE models, in similar style to
# the scripts 1.2_make_neg_window_files_for_epochs.sh and 
# 1.3_make_species_files_for_epochs.sh. See either of those for more explanation.



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

# this script only needs to be run on human genome files
genome="hg38"


echo "Prepping shuffled negative example datasets for $tf ($genome)."

DATA_DIR="$ROOT/data/$genome/$tf"
# these files were created by 1.1_make_val_test_files_and_prep_training_files.sh
TRAIN_FILE="$DATA_DIR/chr3toY_shuf.bed"
POS_FILE="$DATA_DIR/chr3toY_pos_shuf.bed"
NEG_FILE="$DATA_DIR/chr3toY_neg_shuf.bed"

# this file is created by 1.0_make_repeat_files.sh
# this file contains intervals for all annotated SINEs in the genome.
SINES_FILE="$ROOT/data/$genome/sines.bed"


### Step 1: Filter out SINEs from existing training windows files.

ALL_NOSINES_FILE="$DATA_DIR/chr3toY_nosines_shuf.bed"
POS_NOSINES_FILE="$DATA_DIR/chr3toY_pos_nosines_shuf.bed"
NEG_NOSINES_FILE="$DATA_DIR/chr3toY_neg_nosines_shuf.bed"
bedtools intersect -a "$TRAIN_FILE" -b "$SINES_FILE" -wa -v > "$ALL_NOSINES_FILE" || exit 1
bedtools intersect -a "$POS_FILE" -b "$SINES_FILE" -wa -v > "$POS_NOSINES_FILE" || exit 1
bedtools intersect -a "$NEG_FILE" -b "$SINES_FILE" -wa -v > "$NEG_NOSINES_FILE" || exit 1



### Step 2: Make no-SINEs negative example files for each epoch and run
# Normally, when the model trains, it uses the same set of bound examples, plus
# a different random sample of unbound examples, each epoch. To replicate that
# but remove all examples with SINEs, we will create a file for each epoch 
# that contains unbound examples to train on, where none of those examples
# overlaps with a SINE. This is analogous to what happens in the script 
# 1.2_make_neg_window_files_for_epochs.sh.

bound_windows=`wc -l < "$POS_NOSINES_FILE"`

tmp_shuf_file="$DATA_DIR/chr3toY_neg_nosines_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
	shuf "$NEG_NOSINES_FILE" > "$tmp_shuf_file"
	for ((epoch=1;epoch<=EPOCHS;epoch++)); do
		head_line_num=$(( bound_windows * epoch ))
		epoch_run_filename="$DATA_DIR/chr3toY_neg_nosines_shuf_run${run}_${epoch}E.bed"
		head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"
	done
done

rm "$tmp_shuf_file"



### Step 3: Make no-sines background species files for each epoch and run
# For DA models, we also need to make "species-background" training data files.
# See make_species_files_for_epochs.sh (the code here is analogous).

other_genome="mm10"   # assuming the no-SINEs models are only ever trained in hg38
OTHER_POS_FILE="$ROOT/data/$other_genome/$tf/chr3toY_pos_shuf.bed"

other_genome_bound_windows=`wc -l < "$OTHER_POS_FILE"`

if [[ "$bound_windows" -gt "$other_genome_bound_windows" ]]; then
	max_bound_windows="$bound_windows"
else
	max_bound_windows="$other_genome_bound_windows"
fi

tmp_shuf_file="$DATA_DIR/chr3toY_nosines_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
    shuf "$ALL_NOSINES_FILE" > "$tmp_shuf_file"
    for ((epoch=1;epoch<=EPOCHS;epoch++)); do
        head_line_num=$(( max_bound_windows * epoch ))
        epoch_run_filename="$DATA_DIR/chr3toY_nosines_shuf_run${run}_${epoch}E.bed"
        head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$max_bound_windows" > "$epoch_run_filename"
    done
done

rm "$tmp_shuf_file"



echo "Done!"

exit 0





