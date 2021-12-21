#!/bin/bash

set -e

ROOT=$1  # root directory for project (same across all scripts)
#ROOT="/users/kcochran/projects/domain_adaptation"

# the python script below is also expecting a save directory for models to exist
models_dir="$ROOT/models"
mkdir -p "$models_dir"


### Training Combinations To Loop Over

tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )

# which source species to train the model with.
# e.g. if mm10, then the model will only see binding data from mouse
genomes=( "mm10" "hg38" )


for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		python train_final_model.py "$ROOT" "$genome" "$tf"
	done
done
