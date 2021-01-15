from utils import *
from DA_params import DA_Params
from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy



def species_gen(params):
	# This generator returns the species-background data for 1 batch.

	# Each species' worth of data will be 1/3 of the total batchsize
	batchfrac = params.batchsize / 3
	# We mask the binding labels for all species data
	# (so the binding classifier model half does not train on this info)
	dummy_binding_labels = np.array([-1 for _ in xrange(batchfrac * 2)])
	dummy_binding_labels = np.reshape(dummy_binding_labels, (batchfrac * 2, 1))

	# Load 1 generator each for the source and target species
	source_gen = iu.get_generator(params.sourcetrainfile, batchfrac, True) 
	target_gen = iu.get_generator(params.targettrainfile, batchfrac, True) 
	
	while True:
		source_batch = next(source_gen)
		target_batch = next(target_gen)

		if params.chromsize > 0:  # legacy code from when accessability was part of model input
			seqs = np.concatenate((source_batch[0][0], target_batch[0][0]))
			accessibility = np.concatenate((source_batch[0][1], target_batch[0][1]))
			model_input = [seqs, accessibility]
		else:
			# concatenate the data from the two species
			seqs = np.concatenate((source_batch[0], target_batch[0]))
			model_input = seqs

		# create the labels vector for the species discriminator
		species_labels = np.array([0 for _ in xrange(batchfrac)] + [1 for _ in xrange(batchfrac)])
		species_labels = np.reshape(species_labels, (batchfrac * 2, 1))

		yield model_input, dummy_binding_labels, species_labels



def binding_gen(params):
	# This generator returns the data used to train the binding classifier for 1 batch.

	# Binding data will make up 1/3 of the total batch (rest is species-background data)
	batchfrac = params.batchsize / 3
	# The species discriminator does not train on this data, so we mask the labels
	dummy_species_labels = np.reshape(np.array([-1 for _ in xrange(batchfrac)]), (batchfrac, 1))
	
	# 1 generator each for bound and unbound examples (each fetches 1/2 a batch)
	pos_gen = iu.get_generator(params.bindingtrainposfile, int(batchfrac / 2), True) 
	neg_gen = iu.get_generator(params.bindingtrainnegfile, int(batchfrac / 2), True)
	
	while True:
		pos_examples = next(pos_gen)
		neg_examples = next(neg_gen)
		
		# concatenate sequences and labels vectors
		seqs = np.concatenate((pos_examples[0], neg_examples[0]))
		labels = np.concatenate((pos_examples[1], neg_examples[1]))

		# shuffle (not necessary)
		shuffle_order = np.arange(seqs.shape[0])
		np.random.shuffle(shuffle_order)

		seqs = seqs[shuffle_order, :,  :]
		labels = labels[shuffle_order]

		yield seqs, labels, dummy_species_labels



def get_training_gen(params):
	# This generator returns all the data needed each batch to train the model,
	# for both the binding classifier and species discriminator tasks.

	# Species discriminator data generator:
	speciesdisc_gen = species_gen(params)
	# Binding classifier data generator:
	bindclass_gen = binding_gen(params)

	while True:
		# fetch data from the two generators for this batch
		speciesdisc_input, speciesdisc_bind_labels, speciesdisc_species_labels = next(speciesdisc_gen)
		bindclass_input, bindclass_bind_labels, bindclass_species_labels = next(bindclass_gen)

		if params.chromsize > 0:  # legacy code from when accessibility was a possible model input
			seqs = np.concatenate((bindclass_input[0], speciesdisc_input[0]))
			acc = np.concatenate((bindclass_input[1], speciesdisc_input[1]))
			input = [seqs, acc]
		else:
			# concatenate the data from both generators together
			input = np.concatenate((bindclass_input, speciesdisc_input))

		# concatenate label vectors together
		bind_labels = np.concatenate((bindclass_bind_labels, speciesdisc_bind_labels))
		species_labels = np.concatenate((bindclass_species_labels, speciesdisc_species_labels))

		yield input, {"classifier" : bind_labels, "discriminator" : species_labels}


def get_val_gen(params, labels = False, target_data = True):
	# This generator returns a batch of sequences to validate/test the model on each yield.
	# If "labels" is True, what's returned will be a tuple of arrays: (sequences, binding labels)
	# By default, labels is False and only sequences are returned
	# If "target_data" is True, data will be fetched from the target (non-training) species
	# Otherwise, data will be fetched from the source (training) species

	# using dummy species labels here -- we only really care about evaluating binding task performance
	dummy_species_labels = np.reshape(np.array([-1 for _ in xrange(params.valbatchsize)]), (params.valbatchsize, 1))

	if target_data:
		gen = iu.get_generator(params.targetvalfile, params.valbatchsize, labels)
	else:
		gen = iu.get_generator(params.sourcevalfile, params.valbatchsize, labels)
	
	while True:
		batch = next(gen)
		if labels:
			yield batch[0], {"classifier" : batch[1], "discriminator" : dummy_species_labels}
		else:
			yield batch


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
    discriminator = flipGradientTF.GradientReversal(params.lamb)(discriminator)
    discriminator = Dense(params.dl1nodes)(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = Dense(params.dl2nodes, activation = 'sigmoid')(discriminator)
    disc_result = Dense(1, activation = 'sigmoid', name = "discriminator")(discriminator)

    if params.chromsize > 0:  # legacy code from when accessibility was possible model input
        acc_input = Input(shape = (params.chromsize, ), name = 'accessibility')
        acc = Dense(1, activation = 'sigmoid')(acc_input)
        merge_classifier = concatenate([classifier, acc])
        class_result = Dense(1, activation = 'sigmoid', name = "classifier")(merge_classifier)
        inputs = [seq_input, acc_input]
    else:
        class_result = Dense(1, activation = 'sigmoid', name = "classifier")(classifier)
        inputs = seq_input

    model = Model(inputs = inputs, outputs = [class_result, disc_result])
    return model



