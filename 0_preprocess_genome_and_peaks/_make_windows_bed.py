import sys


RAW_DATA_DIR=sys.argv[1]
#ROOT = "/users/kcochran/projects/domain_adaptation/raw_data/"

# this script assumes you've put this file in this location with the correct name!
CHROMOSOME_SIZE_FILE = RAW_DATA_DIR + "/chrom.sizes"

OUT_FILE = RAW_DATA_DIR + "/windows.unfiltered.bed"

# these parameters decide the length and stride of the windows created
# across all chromosomes.
# the window size should be consistent with the expected input sequence of the model.
WINDOW_SIZE = 500
WINDOW_STRIDE = 50


def make_windows():
	with open(CHROMOSOME_SIZE_FILE, "r") as gInfoFile, open(OUT_FILE, "w") as outFile:
		for chromLine in gInfoFile:
			chrom,length = chromLine.strip().split()
			length = int(length)
			window_start = 0
			while window_start + WINDOW_SIZE < length:
				line = "\t".join([chrom, str(window_start), str(window_start + WINDOW_SIZE)])
				outFile.write(line + "\n")
				window_start += WINDOW_STRIDE


if __name__ == "__main__":
	make_windows()
