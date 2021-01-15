from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from keras.optimizers import Adam
import numpy as np

import iterutils as iu



def get_training_gen(params):
	# Each iteration, this generator will retrieve one batch worth of 
	# both bound and unbound examples, and then concatenate them into the full batch.

	pos_gen = iu.get_generator(params.bindingtrainposfile, int(params.batchsize / 2), True)
	neg_gen = iu.get_generator(params.bindingtrainnegfile, int(params.batchsize / 2), True)

	while True:
		pos_examples = next(pos_gen)
		neg_examples = next(neg_gen)

		seqs = np.concatenate((pos_examples[0], neg_examples[0]))
		labels = np.concatenate((pos_examples[1], neg_examples[1]))

		shuffle_order = np.arange(seqs.shape[0])
		np.random.shuffle(shuffle_order)

		seqs = seqs[shuffle_order, :,  :]
		labels = labels[shuffle_order]

		yield (seqs, labels)


def get_val_gen(params, labels = False, target_data = True):
	if target_data:  # if validating on non-training species
		return iu.get_generator(params.targetvalfile, params.valbatchsize, labels)
	else:  # if validating on training species
		return iu.get_generator(params.sourcevalfile, params.valbatchsize, labels)


def basic_model(params):
	# Here we specify the basic model architecture.
	# See default_params.py for specific values of network parameters used.

	seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
	seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
	seq = Activation("relu")(seq)
	seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)
	
	seq = LSTM(params.lstmnodes)(seq)

	seq = Dense(params.dl1nodes, activation = "relu")(seq)
	seq = Dropout(0.5)(seq)
	seq = Dense(params.dl2nodes, activation = "sigmoid")(seq)
	
	if params.chromsize > 0:  # legacy code from when accessibility input into models was attempted
		acc_input = Input(shape = (params.chromsize, ), name = "accessibility")
		acc = Dense(1, activation = 'sigmoid')(acc_input) 
		merge = concatenate([seq, acc])
		result = Dense(1, activation = 'sigmoid')(merge)
		inputs = [seq_input, acc_input]
	else:
		result = Dense(1, activation = 'sigmoid')(seq)
		inputs = seq_input

	model = Model(inputs = inputs, outputs = result)
	return model


# no longer used
def val(params, model, target_data = True, save = False):
	gen_val = get_val_gen(params, target_data = target_data)
	
	if save:
		suffix = ".test"
	else:
		suffix = ".val"
	
	if target_data:
		steps = params.target_val_steps
		suffix = suffix + ".target"
	else:
		steps = params.source_val_steps
		suffix = suffix + ".source"
	
	probs = fix_probs_shape(model.predict_generator(gen_val, steps = steps))

	if save:
		with open(params.get_output_path() + suffix, "w") as f:
			np.savetxt(f, probs)
	return probs


def fix_probs_shape(probs):
	# TODO: figure out which versions of Keras require this and which don't...
	probs_fixed = np.array(probs)
	if len(np.shape(probs)) == 3:
		probs_fixed = probs_fixed.reshape(-1)
		probs_fixed = probs_fixed[:int(len(probs_fixed) / 2)]
	else:
		# only applies to old version of Keras, shouldn't happen
		print("WARNING: using old version of Keras -- array shapes are off")
		probs_fixed = probs_fixed.reshape(sorted(probs_fixed.shape, reverse = True))

		if probs_fixed.shape[1] == 2:
			print("Discarding right-most column of probs (for DA only)")
			probs_fixed = probs_fixed[:,0] # remove probs from species discriminator

	return probs_fixed


