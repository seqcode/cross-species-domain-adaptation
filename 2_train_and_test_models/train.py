from utils import *
from callbacks import *
from generators import *
from default_params import Params
import sys, os
import tensorflow as tf



if __name__ == "__main__":
	params = Params(sys.argv)
	callback = MetricsHistory(params)
	save_callback =  ModelSaveCallback(params)
	
	model = basic_model(params)
	model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

	print(model.summary())

	hist = model.fit_generator(epochs = params.epochs,
							steps_per_epoch = params.train_steps,
							generator = TrainGenerator(params),
							use_multiprocessing = True, workers = 8,
							callbacks = [callback, save_callback])

