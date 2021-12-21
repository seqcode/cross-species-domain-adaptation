from math import floor
from subprocess import check_output
from pprint import pprint
import numpy as np
from datetime import datetime
import os


SPECIES = ["mm10", "hg38"]
TFS = ["CTCF", "CEBPA", "Hnf4a", "RXRA"]

# Need to provide seqdataloader with locations of genome fasta files
# seqdataloader is expecting that there will be a fasta index in the same directory
GENOMES = {"mm10" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
			"hg38" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"}

ROOT = "/users/kcochran/projects/domain_adaptation_nosexchr"
DATA_ROOT = ROOT + "/data/"

# These files are created by the script 1_make_training_and_testing_data/1_runall_setup_model_data.sh
VAL_FILENAME = "chr1_random_1m.bed"
TRAIN_POS_FILENAME = "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME = "chr3toY_neg_shuf_runX_1E.bed"

# where models will be saved during/after training
MODEL_ROOT = ROOT + "/models/"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Params:
	''' This class is a data structure that contains the hard-coded
		parameters and filepaths needed by the rest of the model-training
		code.

		This class is specific to models that don't domain-adapt;
		See DA_params.py for the domain-adaptive model parameters. That class
		does inherit most of the parameters from this class, but there are
		some additional specifics.
	'''

	def __init__(self, args):
		self.batchsize = 400  # number of examples seen every batch during training
		self.seqlen = 500  # the input sequence length that will be expected by the model
		self.convfilters = 240  # number of filters in the convolutional layer
		self.filtersize = 20  # the size of the convolutional filters
		self.strides = 15  # the max-pooling layer's stride
		self.pool_size = 15  # the max-pooling layer's pooling size
		self.lstmnodes = 32  # "width" of the LSTM layer
		self.dl1nodes = 1024  # neurons in the first dense layer (after LSTM)
		self.dl2nodes = 512  # neurons in the second dense layer (before output)
		self.dropout = 0.5  # fraction of neurons in the first dense layer to randomly dropout
		self.valbatchsize = 10000
		self.epochs = 15

		self.parse_args(args)
		self.set_steps_per_epoch()

		pprint(vars(self))

		self.set_val_labels()


	def parse_args(self, args):
		''' This method parses the info passed in (TF, source species, and run #)
			and determines the filepaths for the data, genomes, and model save location.

			This method is expecting that the data for each tf and species
			is organized in a particular directory structure. See
			setup_directories_and_download_files.sh for this directory structure.

			This method is also expecting arguments input in a particular order.
			Should be: TF, source species (mm10 or hg38), run number
		'''
		assert len(args) >= 4, len(args)  # the first item in argv is the script name
		self.tf = args[1]
		assert self.tf in TFS, self.tf
		self.source_species = args[2]
        
		# check for human no-SINEs condition, update file pointers accordingly
		if self.source_species == "NS":
			NS = True
			TRAIN_POS_FILENAME = "chr3toY_pos_nosines_shuf.bed"
			TRAIN_NEG_FILENAME = "chr3toY_neg_nosines_shuf_runX_1E.bed"
			self.source_species = "hg38"
		else:
			NS = False
            
		assert self.source_species in SPECIES, self.source_species
		self.run = int(args[3])

		source_root = DATA_ROOT + self.source_species + "/" + self.tf + "/"
		# the target species is just the opposite species from the source species
		self.target_species = [species for species in SPECIES if species != self.source_species][0]
		target_root = DATA_ROOT + self.target_species + "/" + self.tf + "/"

		self.bindingtrainposfile = source_root + TRAIN_POS_FILENAME

		# The file of negative/unbound examples is specific to each run and epoch.
		# For now we're loading in the filename for the first epoch, but we need to
		# make the filename run-specific.
		self.bindingtrainnegfile = source_root + TRAIN_NEG_FILENAME
		self.bindingtrainnegfile = self.bindingtrainnegfile.replace("runX", "run" + str(self.run))

		self.sourcevalfile = source_root + VAL_FILENAME
		self.targetvalfile = target_root + VAL_FILENAME

		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
		# this filepath is specific to the non-domain-adaptive models
		if not NS:
			self.modeldir = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/basic_model/"
		else:
			self.modeldir = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/basic_model_nosines/"
		os.makedirs(self.modeldir, exist_ok=True)
		self.modelfile = self.modeldir + timestamp + "_run" + str(self.run)

		self.source_genome_file = GENOMES[self.source_species]
		self.target_genome_file = GENOMES[self.target_species]


	def get_output_path(self):
		return self.modelfile.split(".")[0] + ".probs.out"


	def set_steps_per_epoch(self):
		''' This method determines the number of batches for the model to load
			for a complete epoch. It is defined by the floor of the number of examples
			in the training data divided by the batchsize.
			
			Because we know that exactly half our training data is bound, the size of
			the bound examples file is half the number of examples seen in an epoch.
			So, the # of train steps is floor(bound_examples * 2 / batchsize).
			
			Note that we assume the training set is balanced (50% bound examples).
		'''
		command = ["wc", "-l", self.bindingtrainposfile]
		linecount = int(check_output(command).strip().split()[0])
		self.train_steps = int(floor((linecount * 2) / self.batchsize))


	def set_val_labels(self):
		''' This method reads in the labels for the validation data for both species.
			Doing this now avoids having to repeatedly do it every epoch as part of
			model evaluation.
			
			The labels are read in as numpy arrays containing binary values. Because
			the dataset is truncated to have length that is a multiple of the batch size
			when fed into the model, we need to similarly truncate these labels. If your
			batch size is a factor of your validation dataset size, this doesn't have any effect.
		'''
		with open(self.targetvalfile) as f:
			self.target_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.target_val_steps = int(floor(self.target_val_labels.shape[0] / self.valbatchsize))
		self.target_val_labels = self.target_val_labels[:self.target_val_steps * self.valbatchsize]

		with open(self.sourcevalfile) as f:
			self.source_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.source_val_steps = int(floor(self.source_val_labels.shape[0] / self.valbatchsize))
		self.source_val_labels = self.source_val_labels[:self.source_val_steps * self.valbatchsize]

