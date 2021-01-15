from __future__ import division
from default_params import *
from math import ceil, floor
from subprocess import check_output
from pprint import pprint
import numpy as np


SPECIES_FILENAME = "chr3toY_shuf_runX_1E.bed"


class DA_Params(Params):
	# This class is a subclass of the Params class in default_params.py.

	def __init__(self, args):
		Params.__init__(self, args)
		self.parse_args(args)

	def parse_args(self, args):
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
		self.modelfile = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/DA/" + timestamp + "_run" + str(self.run)

		self.source_genome_file = GENOMES[self.source_species]
		self.target_genome_file = GENOMES[self.target_species]

		self.source_species_file = source_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
		self.target_species_file = target_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
		self.lamb = 1
		self.loss_weight = 1


	def get_reshape_size(self):
		# DA models contain a reshape layer above the convolutional filters/pooling.
		# That layer needs, as input when initialized, what shape of input to expect.
		tmp = ceil(self.seqlen / self.strides)
		return int(tmp * self.convfilters)


	def set_steps_per_epoch(self):
		command = ["wc", "-l", self.bindingtrainposfile]
		linecount = int(check_output(command).strip().split()[0])
		print("linecount:" , linecount)
		self.train_steps = int(floor(linecount / (self.batchsize / 2))) ###
		print("Train steps:", self.train_steps)

