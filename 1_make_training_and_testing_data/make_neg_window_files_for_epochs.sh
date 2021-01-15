#!/bin/bash


### This script picks up where setup_training_data.sh leaves off. Here, we
# shuffle the unbound sites that our training data can sample from, and then
# sample a distinct set of sites for training each epoch (no unbound site is
# repeated across multuple epochs). The bound sites trained on each epoch are
# the same. The number of unbound sites sampled each epoch is equal to the
# number of bound sites to be used in training (so each epoch's training data
# will be balanced).

# This process will create a file of unbound sites for training specific to
# each epoch. If you opt for too many epochs (or the size of your training 
# datasets is too large), this script may error out when it uses up all 
# possible unbound sites to sample from.




ROOT="/users/kcochran/projects/domain_adaptation"

# RUNS: the number of replicate model training runs to make data for.
RUNS=5
# EPOCHS: the number of epochs the models will train for.
# Files will be generated for each epoch.
EPOCHS=15



### NOTE: this script needs to be run once for each species and TF.

### Parse args
# Arguments expected: TF (CTCF, CEBPA, Hnf4a, or RXRA) and genome (mm10, hg38)

tf=$1
genome=$2

if [[ -z "$tf" || -z "$genome" ]]; then
  echo "Missing an argument. Required: TF, species."
  exit 1
fi

echo "Prepping shuffled negative example datasets for $tf ($genome)."

DATA_DIR="$ROOT/data/$genome/$tf"
POS_FILE="$DATA_DIR/chr3toY_pos_shuf.bed"
NEG_FILE="$DATA_DIR/chr3toY_neg_shuf.bed"

bound_windows=`wc -l < "$DATA_DIR/chr3toY_pos_shuf.bed"`
unbound_windows=`wc -l < "$DATA_DIR/chr3toY_neg_shuf.bed"`
echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"


tmp_shuf_file="$DATA_DIR/chr3toY_neg_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
	shuf "$NEG_FILE" > "$tmp_shuf_file"
	for ((epoch=1;epoch<=EPOCHS;epoch++)); do
		head_line_num=$(( bound_windows * epoch ))
		echo "For epoch $epoch, head ends at line $head_line_num"
		epoch_run_filename="$DATA_DIR/chr3toY_neg_shuf_run${run}_${epoch}E.bed"
		head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"

		# sanity check
		lines_in_file=`wc -l < "$epoch_run_filename"`
		echo "Lines in $epoch_run_filename: $lines_in_file"
		if [[ "$lines_in_file" != "$bound_windows" ]]; then
			echo "Error: incorrect number of lines ($lines_in_file) in file $epoch_run_filename (should be $bound_windows). Exiting."
			exit 1
		fi
	done
done

rm "$tmp_shuf_file"


echo "Done!"

exit 0





