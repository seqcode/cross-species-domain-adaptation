from math import floor
from subprocess import check_output
from pprint import pprint
import numpy as np
from datetime import datetime


SPECIES = ["mm10", "hg38"]

# Need to provide seqdataloader with locations of genome fasta files
# seqdataloader is expecting that there will be a fasta index in the same directory
GENOMES = {"mm10" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
			"hg38" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"}


TFS = ["CTCF", "CEBPA", "Hnf4a", "RXRA"]
DATA_ROOT = "/users/kcochran/projects/domain_adaptation/data/"

# These files are created by the script runall_setup_training_data.sh
VAL_FILENAME = "chr1_random_1m.bed"
TRAIN_POS_FILENAME = "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME = "chr3toY_neg_shuf_runX_1E.bed"

# where models will be saved during/after training
MODEL_ROOT = "/users/kcochran/projects/domain_adaptation/models/"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Params:
	def __init__(self, args):
		self.batchsize = 400
		self.seqlen = 500
		self.convfilters = 240
		self.filtersize = 20
		self.strides = 15
		self.pool_size = 15
		self.lstmnodes = 32
		self.dl1nodes = 1024
		self.dl2nodes = 512
		self.valbatchsize = 10000
		self.epochs = 15

		self.parse_args(args)
		self.set_steps_per_epoch()
		self.chromsize = 0  # leftover from when accessibility was also input to model

		pprint(vars(self))

		self.set_val_labels()


	def parse_args(self, args):
		# NOTE: this method is expecting arguments input in a particular order!!
		assert len(args) >= 4, len(args)
		self.tf = args[1]
		assert self.tf in TFS, self.tf
		self.source_species = args[2]
		assert self.source_species in SPECIES, self.source_species	
		self.run = int(args[3])
        
		source_root = DATA_ROOT + self.source_species + "/" + self.tf + "/"
		self.target_species = [species for species in SPECIES if species != self.source_species][0]
		target_root = DATA_ROOT + self.target_species + "/" + self.tf + "/"

		self.bindingtrainposfile = source_root + TRAIN_POS_FILENAME
		self.bindingtrainnegfile = source_root + TRAIN_NEG_FILENAME
		self.bindingtrainnegfile = self.bindingtrainnegfile.replace("runX", "run" + str(self.run))
        
		self.sourcevalfile = source_root + VAL_FILENAME
		self.targetvalfile = target_root + VAL_FILENAME

		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
		self.modelfile = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/basic_model/" + timestamp + "_run" + str(self.run)

		self.source_genome_file = GENOMES[self.source_species]
		self.target_genome_file = GENOMES[self.target_species]


	def get_output_path(self):
		return self.modelfile.split(".")[0] + ".probs.out"


	def set_steps_per_epoch(self):
		# NOTE: here we are assuming that the training set is balanced (50% bound examples)
		command = ["wc", "-l", self.bindingtrainposfile]
		linecount = int(check_output(command).strip().split()[0])
		self.train_steps = int(floor((linecount * 2) / self.batchsize))


	def set_chromsize(self):
		# leftover from when accessibility was also input to model
		command = ["head", "-n1", self.bindingtrainposfile]
		line1 = check_output(command).strip()
		# should be 5 columns besides chromatin info in file
		self.chromsize = len(line1.split()) - 5


	def set_val_labels(self):
		# to avoid doing this repeatedly later, we load in all binary labels for val set now
		with open(self.targetvalfile) as f:
			self.target_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.target_val_steps = int(floor(self.target_val_labels.shape[0] / self.valbatchsize))
		self.target_val_labels = self.target_val_labels[:self.target_val_steps * self.valbatchsize]

		with open(self.sourcevalfile) as f:
			self.source_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.source_val_steps = int(floor(self.source_val_labels.shape[0] / self.valbatchsize))
		self.source_val_labels = self.source_val_labels[:self.source_val_steps * self.valbatchsize]

