''' This script trains a binary classifier neural network on
    TF binding data.
    
    The model is a hybrid CNN-LSTM architecture
    implemented in the basic_model function below.
    
    The specific hyperparameters of the model, as well as paths
    to files containing the datasets used for training and validation,
    are stored in the Params data structure (see default_params.py).
    
    See the TrainGenerator class in generators.py for how the
    data loading is implemented. The generator relies on the package
    seqdataloader to transform bed coordinates of example training
    sequences into one-hot encodings.
    
    During training, the performance (auPRC) of the model is 
    evaluated each epoch on the validation sets of both species via
    the MetricsHistory callback class. The results of this evaluation 
    are printed to stdout (if running run_training.sh, they will be
    written to the log file). The per-epoch performance on the 
    source/training species is used as part of a simple model selection
    protocol: the model with the best-so-far source species validation
    set performance is saved for downstream analysis.
    
    Separately, the ModelSaveCallback class saves the model each epoch.
    
    The model trains using the ADAM optimizer and binary cross-entropy loss.
'''

from callbacks import MetricsHistory, ModelSaveCallback
from generators import TrainGenerator
from default_params import Params
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D
from keras.models import Model
import sys




def basic_model(params):
	# Here we specify the basic model architecture.
	# See default_params.py for specific values of network parameters used.

	seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
	seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
	seq = Activation("relu")(seq)
	seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)
	seq = LSTM(params.lstmnodes)(seq)
	seq = Dense(params.dl1nodes, activation = "relu")(seq)
	seq = Dropout(params.dropout)(seq)
	seq = Dense(params.dl2nodes, activation = "sigmoid")(seq)
	result = Dense(1, activation = 'sigmoid')(seq)

	model = Model(inputs = seq_input, outputs = result)
	return model



if __name__ == "__main__":
	params = Params(sys.argv)
	metrics_callback = MetricsHistory(params)
	save_callback = ModelSaveCallback(params)

	model = basic_model(params)
	model.compile(loss = "binary_crossentropy",
					optimizer = "adam",
					metrics = ["accuracy"])
	print(model.summary())

	hist = model.fit_generator(epochs = params.epochs,
							steps_per_epoch = params.train_steps,
							generator = TrainGenerator(params),
							use_multiprocessing = True, workers = 8,
							callbacks = [metrics_callback, save_callback])

