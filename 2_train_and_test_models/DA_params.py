from __future__ import division
from default_params import *
from math import ceil, floor
from subprocess import check_output
from pprint import pprint
import numpy as np
import os

SPECIES_FILENAME = "chr3toY_shuf_runX_1E.bed"


class DA_Params(Params):
	''' This class is a data structure that contains the hard-coded
		parameters and filepaths needed by the rest of the model-training
		code.

		This class is specific to models that domain-adapt;
		See default_params.py for the domain-adaptive model parameters. This class
		does inherit most of the parameters from that class, but there are
		some additional specifics.
	'''

	def __init__(self, args):
		Params.__init__(self, args)
		self.parse_args(args)


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
		# this filepath is specific to the domain-adaptive (DA) models
		self.modeldir = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/DA/"
		os.makedirs(self.modeldir, exist_ok=True)
		self.modelfile = self.modeldir + timestamp + "_run" + str(self.run)

		self.source_genome_file = GENOMES[self.source_species]
		self.target_genome_file = GENOMES[self.target_species]

		# These paths are unique to the domain-adaptive models.
		# In addition to binding-labeled training data, the DA models also
		# use "species-background" training data from both species, without binding labels.
		self.source_species_file = source_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
		self.target_species_file = target_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))


		### These parameters are specific to domain-adaptive models.

		# self.lamb or lambda is the parameter for the weighting of the gradient
		# in the gradient reversal layer. The gradient is multiplied by -1 for "reversal",
		# but it can also be scaled. Set this to 1 to not do any scaling.
		self.lamb = 1

		# This value controls the relative weights of the losses for the two tasks
		# that the domain-adaptive model trains on: binding and species classification.
		# Normally, the gradient backpropogated for either task is not multiplied by anything:
		# this corresponds to a loss_weight of 1. Importantly, for the gradients to be
		# totally equally weighted, you also need equal amounts of examples per batch for 
		# each task. If the loss_weight is < 1, the species discrimination task will be
		# downweighted, and if it is > 1, then it will be upweighted, relative to the binding task.
		self.loss_weight = 1


	def get_reshape_size(self):
		''' DA models contain a reshape layer above the convolutional filters/pooling.
			This layer feeds into the gradient reversal layer and then a dense layer,
			which requires dimensions to be flattened.

			When the Reshape layer is initialized, it needs to know what input shape
			to expect, so this method calculates that.
		'''
		return int(ceil(self.seqlen / self.strides) * self.convfilters)

