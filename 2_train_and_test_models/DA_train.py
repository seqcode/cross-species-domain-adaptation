from utils import *
from DA_utils import *
from callbacks import *
from generators import *
from DA_params import *
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
import flipGradientTF
import os, sys





if __name__ == "__main__":
	params = DA_Params(sys.argv)
	callback = MetricsHistory(params)
	save_callback = ModelSaveCallback(params)
	
	model = DA_model(params)

	model.compile(loss = [custom_loss, custom_loss], loss_weights = [1, params.loss_weight], optimizer = "adam", metrics = ["accuracy"])
	print(model.summary())

	hist = model.fit_generator(epochs = params.epochs,
								steps_per_epoch = params.train_steps,
								generator = DATrainGenerator(params),
								use_multiprocessing = False, workers = 8,
								callbacks = [callback, save_callback])

