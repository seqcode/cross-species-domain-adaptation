# This script uses the output from multiGPS
# to make whole-genome 0/1 labels for TF binding.

set -e

if [[ $# -ne 3 ]]; then
  echo "Error: wrong number of arguments passed to ${0}."
  exit 1
fi

ROOT=$1
#ROOT="/users/kcochran/projects/domain_adaptation"
genome=$2
tf=$3

cell="liver"

# peaks are called at base resolution. windowSize specifies how large
# to make the full peak window, which will be centered at the summit base;
# note that if overlap is 0.5, this doesn't matter
windowSize=200

# what fraction of overlap is needed between a peak and a window?
# an overlap of 0.5 means a majority ( > 50%) of the peak must overlap
overlap=0.5


DATA_DIR="$ROOT/raw_data"
multiGPS_out_dir="$DATA_DIR/${genome}/${tf}"
genome_windows_file="$DATA_DIR/${genome}/windows.bed"  # output from make_windows_files.sh
peak_call_file="$multiGPS_out_dir/mgps_out_${tf}.bed"  # output from MultiGPS

if [[ ! -f "$peak_call_file" ]]; then
  echo "Error: output file from multiGPS cannot be found." && exit 1
fi

if [[ ! -f "$genome_windows_file" ]]; then
  echo "Error: windows.bed does not exist." && exit 1
fi


# tmp files (will be deleted at the end of the script)
peak_windows_file="$multiGPS_out_dir/peaks_${windowSize}bp.bed"
genome_windows_with_peaks_sorted_file="$multiGPS_out_dir/${peaks}_200bp.sorted.bed"
pos_labels_file="$multiGPS_out_dir/binding_labels.pos.bed"
neg_labels_file="$multiGPS_out_dir/binding_labels.neg.bed"
sorted_genome_windows_file="$DATA_DIR/${genome}/windows.dictsort.bed"

# final output file
labels_file="$multiGPS_out_dir/binding_labels.bed"



echo "Getting TF binding peak windows from mGPS output..."

# convert mGPS peak center coords to 200-bp windows, in bed format
grep -E "chr[0-9]+" "$peak_call_file" | awk -v winSize="$windowSize" 'BEGIN{OFS="\t"}{if ($2 - winSize / 2 > 0) {print $1,$2 - winSize / 2,$3 + winSize / 2 - 1} else print $1, 0, $3 + winSize / 2 - 1}' | sort -k1,1 -k2,2n > "$peak_windows_file"
[ ! -s "$peak_windows_file" ] && exit 1

# find all genome windows that overlap with one or more peaks by $overlap fraction
bedtools intersect -a "$genome_windows_file" -b "$peak_windows_file" -F "$overlap" -wa | sort | uniq > "$genome_windows_with_peaks_sorted_file"
[ ! -s "$genome_windows_with_peaks_sorted_file" ] && exit 1

rm "$peak_windows_file" 


echo "Making TF 0/1 labels..."

# assign 1 labels to all windows that did overlap with a peak
sed "s/$/	1/" "$genome_windows_with_peaks_sorted_file" > "$pos_labels_file"

if [[ ! -f "$sorted_genome_windows_file" ]]; then
  sort "$genome_windows_file" > "$sorted_genome_windows_file"
fi

# assign 0 labels to all other windows
# the genome windows file needs to be sorted (dictonary sort, not bed sort) for comm to work
comm "$sorted_genome_windows_file" "$genome_windows_with_peaks_sorted_file" -23 | sed "s/$/	0/" > "$neg_labels_file"

# merge the two and sort (normal bed-file sort)
cat "$neg_labels_file" "$pos_labels_file" | sort -k1,1 -k2,2n > "$labels_file"

# cleanup
poslines=$(wc -l < "$pos_labels_file")
lines=$(wc -l < "$labels_file")

echo "${tf} labels finished. Output file contains ${lines} windows; ${poslines} overlapped with ${tf} peaks."

rm "$genome_windows_with_peaks_sorted_file" "$pos_labels_file" "$neg_labels_file" "$sorted_genome_windows_file"

exit 0

