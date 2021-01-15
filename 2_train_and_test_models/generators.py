from keras.utils import Sequence
import numpy as np
import random

from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals

import os
import signal



class TrainGenerator(Sequence):
	def __init__(self, params):
		self.posfile = params.bindingtrainposfile
		self.negfile = params.bindingtrainnegfile
		self.converter = PyfaidxCoordsToVals(params.source_genome_file)
		self.batchsize = params.batchsize
		self.halfbatchsize = self.batchsize // 2
		self.steps_per_epoch = params.train_steps
		self.total_epochs = params.epochs
		self.current_epoch = 1

		self.get_coords()
		self.on_epoch_end()


	def __len__(self):
		return self.steps_per_epoch


	def get_coords(self):
		# Using current filenames stored in self.posfile and self.negfile,
		# load in all of the training data as coordinates only.
		# Then, when it is time to fetch individual batches, a chunk of
		# coordinates will be converted into one-hot encoded sequences
		# ready for model input.
		try:
			with open(self.posfile) as posf:
				pos_coords_tmp = [line.split()[:3] for line in posf]  # expecting bed file format
				self.pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
		except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
			print(e)
			raise e


	def __getitem__(self, batch_index):	
		try:
			# First, get chunk of coordinates
			pos_coords_batch = self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
			neg_coords_batch = self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]

			# if train_steps calculation is off, lists of coords may be empty
			assert len(pos_coords_batch) > 0, len(pos_coords_batch)
			assert len(neg_coords_batch) > 0, len(neg_coords_batch)

			# Seconds, convert the coordinates into one-hot encoded sequences
			pos_onehot = self.converter(pos_coords_batch)
			neg_onehot = self.converter(neg_coords_batch)

			# seqsdataloader returns empty array if coords are empty list or not in genome
			assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
			assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

			# Third, combine bound and unbound sites into one large array, and create label vector
			all_seqs = np.concatenate((pos_onehot, neg_onehot))
			labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))

			assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]
			return all_seqs, labels
		except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
			print(e)
			raise e


	def on_epoch_end(self):
		try:
			# switch to next set of negative examples
			prev_epoch = self.current_epoch
			next_epoch = prev_epoch + 1

			# update file where we will retrieve unbound site coordinates from
			prev_negfile = self.negfile
			next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
			self.negfile = next_negfile

			# load in new unbound site coordinates
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
		
			# then shuffle positive examples
			random.shuffle(self.pos_coords)

		except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
			print(e)
			raise e


class ValGenerator(Sequence):
	def __init__(self, params, target_species = False):
		if target_species:
			self.valfile = params.targetvalfile
			self.steps_per_epoch = params.target_val_steps
			self.converter = PyfaidxCoordsToVals(params.target_genome_file)
		else:
			self.valfile = params.sourcevalfile
			self.steps_per_epoch = params.source_val_steps
			self.converter = PyfaidxCoordsToVals(params.source_genome_file)

		self.batchsize = params.valbatchsize
		self.get_coords()


	def __len__(self):
		return self.steps_per_epoch


	def get_coords(self):
		# load in coordinates of each validation set example into memory
		try:
			with open(self.valfile) as f:
				coords_tmp = [line.split()[:3] for line in f]
				self.coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
		except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
			print(e)
			raise e


	def __getitem__(self, batch_index):
		try:
			# get chunk of coordinates
			coords_batch = self.coords[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]
			assert len(coords_batch) > 0, len(coords_batch)

			# convert chunk of coordinates to array of one-hot encoded sequences
			seq_onehot = self.converter(coords_batch)
			assert seq_onehot.shape[0] > 0, seq_onehot.shape[0] 
			return seq_onehot
		except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
			print(e)
			raise e



class DATrainGenerator(Sequence):
	def __init__(self, params):
		print(vars(params))

		self.posfile = params.bindingtrainposfile
		self.negfile = params.bindingtrainnegfile
		self.source_species_file = params.source_species_file
		self.target_species_file = params.target_species_file
		self.source_converter = PyfaidxCoordsToVals(params.source_genome_file)
		self.target_converter = PyfaidxCoordsToVals(params.target_genome_file)
		self.batchsize = params.batchsize
		self.halfbatchsize = self.batchsize // 2
		self.steps_per_epoch = params.train_steps
		self.total_epochs = params.epochs
		self.current_epoch = 1

		self.get_binding_coords()
		self.get_species_coords()
		self.on_epoch_end()


	def __len__(self):
		return self.steps_per_epoch


	def get_binding_coords(self):
		try:
			# Using current filenames stored in self.posfile and self.negfile,
			# load in all of the "binding" training data as coordinates only.
			# Then, when it is time to fetch individual batches, a chunk of
			# coordinates will be converted into one-hot encoded sequences
			# ready for model input.
			with open(self.posfile) as posf:
				pos_coords_tmp = [line.split()[:3] for line in posf]
				self.pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
		except Exception as e:
			print(e)
			raise e


	def get_species_coords(self):
		try:
			# Same as get_binding_coords(), but loading in coordinates
			# for DA-specific training data (from both species).
			with open(self.source_species_file) as sourcef:
				source_coords_tmp = [line.split()[:3] for line in sourcef]
				self.source_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
			with open(self.target_species_file) as targetf:
				target_coords_tmp = [line.split()[:3] for line in targetf]
				self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
		except Exception as e:
			print(e)
			raise e


	def __getitem__(self, batch_index):
		try:
			# First, we retrieve a chunk of coordinates for both the bound and unbound site examples,
			# and convert those coordinates to one-hot encoded sequence arrays
			pos_onehot = self.source_converter(self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize])
			neg_onehot = self.source_converter(self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize])
			assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
			assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

			# Second, we do the same thing again, but for the "species-background" data of both species
			source_onehot = self.source_converter(self.source_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize])
			target_onehot = self.target_converter(self.target_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize])
			assert source_onehot.shape[0] > 0, source_onehot.shape[0]
			assert target_onehot.shape[0] > 0, target_onehot.shape[0]
			
			# Third, concatenate all the one-hot encoded sequences together
			all_seqs = np.concatenate((pos_onehot, neg_onehot, source_onehot, target_onehot))

			# Fourth, create label vectors for both tasks
			# Note that a label of -1 will correspond to a masking of the loss function
			# (so if the label for the binding task is -1 for example i, then when the
			# loss gradient backpropagates, example i will not be included in that calculation

			# label vector for binding prediction task:
			binding_labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],),
										-1 * np.ones(source_onehot.shape[0],), -1 * np.ones(target_onehot.shape[0],)))
			# label vector for species discrimination task:
			species_labels = np.concatenate((-1 * np.ones(pos_onehot.shape[0],), -1 * np.ones(neg_onehot.shape[0],),
										np.zeros(source_onehot.shape[0],), np.ones(target_onehot.shape[0],)))

			assert all_seqs.shape[0] == self.batchsize * 2, all_seqs.shape[0]
			assert binding_labels.shape == species_labels.shape, (binding_labels.shape, species_labels.shape)

			# here we assign the name "classifier" to the binding prediction task, and
			# "discriminator" to the species prediction task
			return all_seqs, {"classifier":binding_labels, "discriminator":species_labels}
		except Exception as e:
			print(e)
			raise e


	def on_epoch_end(self):
		try:
			# switch to next set of negative and species examples
			prev_epoch = self.current_epoch
			next_epoch = prev_epoch + 1

			# update file to pull coordinates from, for unbound examples and species-background examples
			prev_negfile = self.negfile
			next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
			self.negfile = next_negfile

			prev_sourcefile = self.source_species_file
			next_sourcefile = prev_sourcefile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
			self.source_species_file = next_sourcefile

			prev_targetfile = self.target_species_file
			next_targetfile = prev_targetfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
			self.target_species_file = next_targetfile


			# load in coordinates into memory for unbound examples and species-background examples		
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]

			with open(self.source_species_file) as sourcef:
				source_coords_tmp = [line.split()[:3] for line in sourcef]
				self.source_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
			with open(self.target_species_file) as targetf:
				target_coords_tmp = [line.split()[:3] for line in targetf]
				self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
		
			# then shuffle positive examples
			random.shuffle(self.pos_coords)
		except Exception as e:
			print(e)
			raise e


