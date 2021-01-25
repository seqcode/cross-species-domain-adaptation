#!/bin/bash

### NOTE: This script expects to see the file chr2.bed in the $ROOT directory.
# This file just needs to have all the chromosome, start, and end infos from
# all windows on hg38 chromosome 2. You could make this file by taking any
# all.all file (which Tf you use the file for doesn't matter), grep/awking for
# chromosome 2, and then using awk to filter for only the first 3 columns.



### Some analyses will require either a file containing all annotated Alu
# repeats (chr2_alus.bed) or a file containing all genomic windows that
# overlap with any Alu repeat (chr2_alus_intersect.bed). This script will
# create those files.

# This script also creates a sines.bed file -- a similar line of code was
# used to create corresponding files for other repeat types (e.g. LINEs)
# for later analyses.



ROOT="/users/kcochran/projects/domain_adaptation/data/hg38"


### Step 1: download rmsk.bed from UCSC Table Browser of tracks (RepeatMasker track)
# Confirm with your eyeballs that the columns look correct (see awk below)

if [[ ! -f "$ROOT/rmsk.bed" ]]; then
	echo "RepeatMasker track needs to be downloaded from UCSC."
	exit 1
fi


### Step 2: isolate SINEs and Alus

awk -v OFS="\t" '{ if ($12 == "SINE") print $6, $7, $8, $11, $12, $13 }' "$ROOT/rmsk.bed" > "$ROOT/sines.bed"
awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$ROOT/rmsk.bed" > "$ROOT/alus.bed"
awk -v OFS="\t" '$1 == "chr2"' "$ROOT/alus.bed" > "$ROOT/chr2_alus.bed"

bedtools intersect -a "chr2.bed" -b "$ROOT/chr2_alus.bed" -u > "$ROOT/chr2_alus_intersect.bed"
