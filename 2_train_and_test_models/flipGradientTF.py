''' This file is modified from Miche Tonu's Gradient Reversal Layer code at:
    https://github.com/michetonu/gradient_reversal_keras_tf
    
    The gradient reversal layer implementation requires implementing
    what to do on the forward pass and the backward pass through the layer.
    On the forward pass, the GRL just outputs its input; on the backward
    pass, it outputs the gradient passing in times negative lambda, where
    lambda is a hyperparameter (in this workflow, set to 1).
'''


import tensorflow as tf
from keras.engine import Layer
import keras.backend as K

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        # return the gradient times negative lambda
        return [tf.negative(grad) * hp_lambda]

    # during the forward pass, the output should be the same
    # as the input (or the identity of the tensor)
    g = tf.compat.v1.Session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        
        # the gradient will be multiplied by this lambda
        # at the same time that it is reversed
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        # there are no weights to train in this layer
        self.trainable_weights = []
        super(GradientReversal, self).build(input_shape)

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
