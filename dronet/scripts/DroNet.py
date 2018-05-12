import os

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, \
                           Conv2DLayer, MaxPool2DLayer, ElemwiseSumLayer, \
                           Pool2DLayer, FlattenLayer, \
                           batch_norm, DropoutLayer, get_all_param_values, \
                           get_output, get_all_params, set_all_param_values
from lasagne.updates import adam
from lasagne.objectives import binary_crossentropy, binary_accuracy, squared_error
from lasagne.nonlinearities import rectify, sigmoid, softmax, tanh

class DroNet:
    def __init__(self,
                 load_weights = True,
                 model_name   = 'dronet_weights.npz'):

        self.model_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name)

        def network(image):
            input_image = InputLayer(input_var = image,
                                     shape     = (None, 1, 120, 160))

            conv1       = Conv2DLayer(input_image,
                                      num_filters  = 32,
                                      filter_size  = (5,5),
                                      stride       = (2,2),
                                      nonlinearity = rectify,
                                      pad          = 'same')

            pool1       = MaxPool2DLayer(conv1,
                                         pool_size = (3,3),
                                         stride = (2,2),
                                         pad = 1)

            conv2       = batch_norm(Conv2DLayer(pool1,
                                                 num_filters  = 32,
                                                 filter_size  = (3,3),
                                                 stride       = (2,2),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            conv2       = batch_norm(Conv2DLayer(conv2,
                                                 num_filters  = 32,
                                                 filter_size  = (3,3),
                                                 stride       = (1,1),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            downsample1 = Conv2DLayer(pool1,
                                      num_filters  = 32,
                                      filter_size  = (1,1),
                                      stride       = (2,2),
                                      nonlinearity = rectify,
                                      pad          = 'same')

            input3      = ElemwiseSumLayer([downsample1,
                                            conv2])

            conv3       = batch_norm(Conv2DLayer(input3,
                                                 num_filters  = 64,
                                                 filter_size  = (3,3),
                                                 stride       = (2,2),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            conv3       = batch_norm(Conv2DLayer(conv3,
                                                 num_filters  = 64,
                                                 filter_size  = (3,3),
                                                 stride       = (1,1),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            downsample2 = Conv2DLayer(input3,
                                      num_filters  = 64,
                                      filter_size  = (1,1),
                                      stride       = (2,2),
                                      nonlinearity = rectify,
                                      pad          = 'same')

            input4      = ElemwiseSumLayer([downsample2,
                                            conv3])

            conv4       = batch_norm(Conv2DLayer(input4,
                                                 num_filters  = 128,
                                                 filter_size  = (3,3),
                                                 stride       = (2,2),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            conv4       = batch_norm(Conv2DLayer(conv4,
                                                 num_filters  = 128,
                                                 filter_size  = (3,3),
                                                 stride       = (1,1),
                                                 nonlinearity = rectify,
                                                 pad          = 'same'))

            downsample3 = Conv2DLayer(input4,
                                      num_filters  = 128,
                                      filter_size  = (1,1),
                                      stride       = (2,2),
                                      nonlinearity = rectify,
                                      pad          = 'same')

            input5      = ElemwiseSumLayer([downsample3,
                                            conv4])


            flatten     = DropoutLayer(FlattenLayer(input5), 0.5)

            prob_out    = DenseLayer(flatten,
                                     num_units    = 1,
                                     nonlinearity = sigmoid)

            turn_angle  = DenseLayer(flatten,
                                     num_units    = 1,
                                     nonlinearity = tanh)

            return prob_out, turn_angle


        # declare the variables used in the network
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.Z = T.fmatrix()


        # Lasagne object for the network
        self.CollisionProbability, self.TurnAngle = network(self.X)


        # collision probability for training
        # and testing. Output is a theano object
        self.collision_prob      = get_output(self.CollisionProbability)
        self.collision_prob_test = get_output(self.CollisionProbability, deterministic=True)


        # turn angle for training anf testing.
        # Output is a theano object.
        self.turn_angle      = get_output(self.TurnAngle)
        self.turn_angle_test = get_output(self.TurnAngle, deterministic=True)


        # Loss for the network.
        self.collision_loss = binary_crossentropy(self.collision_prob, self.Y).mean()
        self.turn_loss      = squared_error(self.turn_angle, self.Z).mean()


        # Loss to call for testing and validation.
        self.test_collision_loss = binary_crossentropy(self.collision_prob_test, self.Y).mean()
        self.test_turn_loss      = squared_error(self.turn_angle_test, self.Z).mean()

        # network parameters for training.
        self.collision_params = get_all_params(self.CollisionProbability, trainable=True)
        self.turn_params = get_all_params(self.TurnAngle, trainable=True)


        # network updates
        self.collision_updates = adam(self.collision_loss,
                                      self.collision_params,
                                      learning_rate = 0.001)

        self.turn_updates = adam(self.turn_loss,
                                 self.turn_params,
                                 learning_rate = 0.00005)


        # get test loss
        self.test_collision = theano.function(inputs               = [self.X, self.Y],
                                              outputs              = self.test_collision_loss,
                                              allow_input_downcast = True)

        self.test_turn = theano.function(inputs               = [self.X, self.Z],
                                         outputs              = self.test_turn_loss,
                                         allow_input_downcast = True)



        # training functions
        self.train_collision = theano.function(inputs               = [self.X, self.Y],
                                               outputs              = self.collision_loss,
                                               updates              = self.collision_updates,
                                               allow_input_downcast = True)

        self.train_turn = theano.function(inputs               = [self.X, self.Z],
                                          outputs              = self.turn_loss,
                                          updates              = self.turn_updates,
                                          allow_input_downcast = True)



        # run the network to calculate collision probability
        # and turn angle given an input.
        self.dronet = theano.function(inputs = [self.X],
                                      outputs = [self.turn_angle_test,
                                                 self.collision_prob_test],
                                      allow_input_downcast = True)

        def load():
            with np.load(self.model_name) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            set_all_param_values([self.CollisionProbability,
                                  self.TurnAngle], param_values)

        if load_weights:
            load()

    def save_weights(self):
        np.savez(self.model_name,
                 *get_all_param_values([self.CollisionProbability,
                                        self.TurnAngle]))

    def __call__(self,
                 image):
        image = image.reshape(1,1,120,160)

        collision_prob, turn_angle = self.dronet(image)

        return collision_prob[0,0], turn_angle[0,0]
