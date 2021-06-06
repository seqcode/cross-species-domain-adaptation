import sys


ROOT = "/users/kcochran/projects/domain_adaptation/raw_data/"

GENOME = sys.argv[1]  # mm10 or hg38
assert GENOME == "mm10" or GENOME == "hg38", GENOME

# this script assumes you've put this file in this location with the correct name!
CHROMOSOME_SIZE_FILE = ROOT + GENOME + "/chrom.sizes"

OUT_FILE = ROOT + GENOME + "/windows.unfiltered.bed"

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
