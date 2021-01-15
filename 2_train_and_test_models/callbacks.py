from keras.callbacks import Callback
import numpy as np
import subprocess

import plot_print_utils as putils
import utils
from generators import ValGenerator


class MetricsHistory(Callback):
	# This callback calculates and saves the validation auPRCs from both species, each epoch.

	def __init__(self, parameters):
		self.auprcs = []
		# some builtin variable is called params so be careful
		self.parameters = parameters

	def on_epoch_end(self, batch, logs={}):
		params = self.parameters
		
		print("Validating on target species...")
		params.target_val_probs = self.model.predict_generator(ValGenerator(params, True),
																use_multiprocessing = True,
																workers = 8)
		target_auprc = putils.print_val_metrics(params, target_data = True)

		print("Validating on source species...")
		params.source_val_probs = self.model.predict_generator(ValGenerator(params, False),
																use_multiprocessing = True,
																workers = 8)
		source_auprc = putils.print_val_metrics(params, target_data = False)
		
		current_auprcs = self.auprcs
		if len(current_auprcs) == 0 or source_auprc > max(current_auprcs):
			print("Best model so far! (source species) validation auPRC = ", source_auprc)
			self.model.save(params.modelfile + "_best.model")
		current_auprcs.append(source_auprc)
		self.auprcs = current_auprcs


class ModelSaveCallback(Callback):
	# This callback saves the model in its current state at the beginning of each epoch,
	# and at the end of training.

	def __init__(self, parameters):
		self.model_save_file = parameters.modelfile
		self.epoch_count = 0

	def on_epoch_begin(self, batch, logs={}):
		filename = self.model_save_file + "_" + str(self.epoch_count) + "E.model"
		self.model.save(filename)
		self.epoch_count += 1

	def on_train_end(self, logs={}):
		filename = self.model_save_file + "_" + str(self.epoch_count) + "E_end.model"
		self.model.save(filename)

