#!/bin/bash

ulimit -n 4096

LOG_ROOT="/users/kcochran/projects/domain_adaptation/logs/training"

tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
genomes=( "mm10" )
runs=( 1 2 3 4 5 )

for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		for run in "${runs[@]}"; do
			python DA_train.py "$tf" "$genome" "$run" > "$LOG_ROOT/DA_${genome}_${tf}_run${run}.log"
		done
	done
done
