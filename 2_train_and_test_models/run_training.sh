#!/bin/bash

set -e

ROOT=$1  # root directory for project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"
LOG_ROOT="$ROOT/logs"

# where all log files will be written to
# log files contain performance measurements made each epoch for the model
mkdir -p "$LOG_ROOT"
# log file naming just needs to be consistent with what the jupyter notebooks
# for downstream analysis are expecting.

# the python script below is also expecting a save directory for models to exist
models_dir="$ROOT/models"
mkdir -p "$models_dir"


### Training Combinations To Loop Over

tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )

# which source species to train the model with.
# e.g. if mm10, then the model will only see binding data from mouse
genomes=( "mm10" "hg38" )

# to measure reproducibility, we trained 5 models for every scenario (tf/species combo)
runs=( 1 2 3 4 5 )


for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		for run in "${runs[@]}"; do
			# this python script will train 1 model
			# data fetched will be for the given $tf and $genome (source species)
			# epoch-by-epoch performance metrics will be printed out to the log file
			echo "Training $tf model in $genome, run $run."
			python train.py "$tf" "$genome" "$run" > "$LOG_ROOT/BM_${genome}_${tf}_run${run}.log"
		done
	done
done
