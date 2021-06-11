''' This script trains a binary classifier neural network on
    TF binding data.
    
    The model is a hybrid CNN-LSTM architecture augmented with a 
    second "head" and a gradient reversal layer, implemented in the
    DA_model function below. The second head is tasked with trying to
    discriminate between sequences from the source and target species,
    and trains on mutually exclusive data from the normal binding task head.
    
    The specific hyperparameters of the model, as well as paths
    to files containing the datasets used for training and validation,
    are stored in the DA_Params data structure (see default_params.py and
    DA_params.py).
    
    See the DATrainGenerator class in generators.py for how the
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
    
    The model trains using the ADAM optimizer and binary cross-entropy
    loss with one augmentation -- in order to train both the TF binding and
    species-discriminating "heads" within the same batch, when the two heads
    train on mutually exclusive datasets, the loss must be masked for one of
    the two heads for each example. Implemented in the custom_loss function
    below is standard BCE loss with that mask included. See the code for the
    DATrainGenerator class for details on how the masked labels are created.
    
    By default, the losses of the TF binding and species-discriminating heads
    are equally weighted because the same amount of examples are used each batch
    for each task and because the loss_weight parameter used below is 1. But
    this parameter can be changed to upweight or downweight one task relative
    to the other.
'''

from callbacks import MetricsHistory, ModelSaveCallback
from generators import DATrainGenerator
from DA_params import DA_Params
from flipGradientTF import GradientReversal
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from keras.models import Model
from keras.losses import binary_crossentropy
import tensorflow as tf
import sys





def custom_loss(y_true, y_pred):
	# The model will be trained using this loss function, which is identical to normal BCE
	# except when the label for an example is -1, that example is masked out for that task.
	# This allows for examples to only impact loss backpropagation for one of the two tasks.
	y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
	y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
	return binary_crossentropy(y_true, y_pred)



def DA_model(params):
	# Here we specify the architecture of the domain-adaptive model.
	# See DA_params.py for specific parameters values used here.

	seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
	seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
	seq = Activation('relu')(seq)
	seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)

	classifier = LSTM(params.lstmnodes)(seq)
	classifier = Dense(params.dl1nodes)(classifier)
	classifier = Activation('relu')(classifier)
	classifier = Dense(params.dl2nodes, activation = 'sigmoid')(classifier)

	discriminator = Reshape((params.get_reshape_size(), ))(seq)
	discriminator = GradientReversal(params.lamb)(discriminator)
	discriminator = Dense(params.dl1nodes)(discriminator)
	discriminator = Activation('relu')(discriminator)
	discriminator = Dense(params.dl2nodes, activation = 'sigmoid')(discriminator)
	disc_result = Dense(1, activation = 'sigmoid', name = "discriminator")(discriminator)

	class_result = Dense(1, activation = 'sigmoid', name = "classifier")(classifier)

	model = Model(inputs = seq_input, outputs = [class_result, disc_result])
	return model




if __name__ == "__main__":
	params = DA_Params(sys.argv)
	metrics_callback = MetricsHistory(params)
	save_callback = ModelSaveCallback(params)

	model = DA_model(params)
	# this custom loss is just standard binary cross-entropy,
	# but all examples are masked for one of the two tasks, since
	# the binding and species-discriminating tasks do not share data.
	model.compile(loss = [custom_loss, custom_loss],
					loss_weights = [1, params.loss_weight],
					optimizer = "adam", metrics = ["accuracy"])
	print(model.summary())

	hist = model.fit_generator(epochs = params.epochs,
								steps_per_epoch = params.train_steps,
								generator = DATrainGenerator(params),
								use_multiprocessing = False, workers = 8,
								callbacks = [metrics_callback, save_callback])

