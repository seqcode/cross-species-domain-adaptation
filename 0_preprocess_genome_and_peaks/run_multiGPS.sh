#!/bin/bash


# This script runs MultiGPS. See http://mahonylab.org/software/multigps
# for details on necessary files to input, settings, etc.
# Citation: Mahony et al. 2014


# Expected arguments:
# 1. mm10 or hg38 (which species to use data from)
# 2. tf name (as specified in design file) -- CTCF, CEBPA, Hnf4a, or RXRA
#        (must also match what is in READDB; case-sensitive)


genome=$1
tf=$2



DATA_ROOT="/storage/home/kxc1032/group/lab/kelly/experimental_data/"

cd "$DATA_ROOT" || exit 1


# MultiGPS requres a read distribution file (downloadable from here:)
# http://mahonylab.org/software/multigps/
READ_DIST_FILE_PATH="/storage/home/kxc1032/kelly/scripts/Read_Distribution_default.txt"

# This directory contains fasta files for each chromosome in the genome (required)
GENOME_DIR_PATH="/storage/home/kxc1032/group/genomes/${genome}"

# This file is a bed file of regions to not call peaks in
# Blacklist is taken from ENCODE (Amemiya et al. 2019) for all species
BLACKLIST_PATH="/storage/home/kxc1032/group/lab/kelly/${genome}/genome_files/${genome}_blacklist.regions"

# MultiGPS uses meme for motif finding -- need to supply the location of the software
MEME_PATH="/storage/home/kxc1032/group/software/meme_4.11.3/bin"


# This directory is where I placed my design file and where I want the outputs to be saved
EXPT_DIR="${genome}/liver/${tf}"
DESIGN_FILE="${EXPT_DIR}/mgps.design"  # you need to write your own design file
OUT_DIR="${EXPT_DIR}/mgps_out/"  # where all output files will be saved


# adjusting because MultiGPS expects the species name to match what's in the design file
# design file species name is dictated by what is in READDB
case "$genome" in
  mm10) genome_long="Mus musculus;mm10";;
  hg38) genome_long="Homo sapiens;hg38";;
  *)
    echo "Error: genome not recognized."
    exit 1
  ;;
esac


echo "Running multiGPS..."

java -Xmx45G org.seqcode.projects.multigps.MultiGPS --species "${genome_long}" --fixedpb 5 --d "$READ_DIST_FILE_PATH" --threads 8 --exclude "$BLACKLIST_PATH" --verbose --probshared 0.99 --seq "$GENOME_DIR_PATH" --memepath "$MEME_PATH" --mememinw 6 --mememaxw 16 --design "$DESIGN_FILE" --verbose --poissongausspb --out "$OUT_DIR" 2>&1

echo "Done"
