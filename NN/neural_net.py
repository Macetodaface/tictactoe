import numpy as np
import random
from copy import deepcopy

class NeuralNet:
  def __init__(self, structure, learning_rate):
    """
    Initialize the Neural Network.

    - structure is a dictionary with the following keys defined:
        num_inputs
        num_outputs
        num_hidden
    - learning rate is a float that should be used as the learning
        rate coefficient in training

    When building your net, make sure to initialize your weights
    to random values in the range [-0.05, 0.05]. Specifically, you
    should use some transformation of 'np.random.rand(n,m).'
    """
    self.num_hidden = structure["num_hidden"]
    num_inputs = structure["num_inputs"]
    self.num_outputs = structure["num_outputs"]
    range = (-.05, .05)
    self.weightsList1 = np.random.rand(num_inputs+1, self.num_hidden)
    self.weightsList1 = self.weightsList1 * (range[1] - range[0]) + range[0]
    self.weightsList2 = np.random.rand(self.num_hidden, self.num_outputs)
    self.weightsList2 = self.weightsList2 * (range[1] - range[0]) + range[0]
    self.learning_rate = learning_rate

  def get_weights(self):
    """
    Returns (w1, w2) where w1 is a matrix representing the current
    weights from the input to the hidden layer and w2 is a similar
    matrix for the hidden to output layers. Specifically, w1[i,j]
    should be the weight from input node i to hidden unit j.
    """
    return self.weightsList1, self.weightsList2

  def forward_propagate(self, x):
    """
    Push the input 'x' through the network and returns the activations
    on the output nodes.

    - x is a numpy array representing an input to the NN

    Return a numpy array representing the activations of the output nodes.

    Hint: you may want to update state here, since you should call this
    method followed by back_propagate in your train method.
    """
    self.inputs = np.append(x, 1)
    self.hidden = [0] * self.num_hidden
    self.outputs = [0] * self.num_outputs
    self.get_next_layer(self.weightsList1, self.hidden, self.inputs)
    self.get_next_layer(self.weightsList2, self.outputs, self.hidden)
    return np.asarray(self.outputs)

  def get_next_layer(self, weightsList, next, prev):
    for i in xrange(len(prev)):
      for j in xrange(len(next)):
        next[j] += weightsList[i][j]*prev[i]

    for i in xrange(len(next)):
      next[i] = self.sigmoid(next[i])
    return next


  def sigmoid(self, n):
    return 1/(1+np.exp(-n))


  def back_propagate(self, target):
    """
    Updates the weights of the NN for the last forward_propagate call.

    - target is the label of the last forward_propogate input
    """
    output_deltas = []
    newWeightsList2 = deepcopy(self.weightsList2)
    for i in xrange(len(self.outputs)):
      output = self.outputs[i]
      output_delta = output*(1-output)*(target[i]-output)
      output_deltas.append(output_delta)
      for j in xrange(len(self.hidden)):
        weight = self.weightsList2[j][i]
        a = self.hidden[j]
        newWeight = weight + self.learning_rate * output_delta * a
        newWeightsList2[j][i] = newWeight
    for i in xrange(len(self.hidden)):
      hidden = self.hidden[i]
      sum_weighted_deltas = 0
      for j in xrange(len(self.outputs)):
        weight = self.weightsList2[i][j]
        sum_weighted_deltas += weight * output_deltas[j]
      hidden_delta = hidden * (1 - hidden) * sum_weighted_deltas
      for j in xrange(len(self.inputs)):
        weight = self.weightsList1[j][i]
        a = self.inputs[j]
        newWeight = weight + self.learning_rate * hidden_delta * a
        self.weightsList1[j][i] = newWeight

    self.weightsList2 = newWeightsList2
    return

  def train(self, X, Y, iterations=1000):
    """
    Trains the NN on observations X with labels Y.

    - X is a numpy matrix (array of arrays) corresponding to a series of
        observations. Each row is a new observation.
    - Y is a numpy matrix (array of arrays) corresponding to the labels
        of the observations.
    - iterations is how many passes over X should be completed.
    """
    for iter in xrange(iterations):
      for i in xrange(len(X)):
        self.forward_propagate(X[i])
        self.back_propagate(Y[i])
    return

  def test(self, X, Y):
    """
    Tests the NN on observations X with labels Y.

    - X is a numpy matrix (array of arrays) corresponding to a series of
        observations. Each row is a new observation.
    - Y is a numpy matrix (array of arrays) corresponding to the labels
        of the observations.

    Returns the mean squared error.
    """
    total_error = 0
    for i in xrange(len(X)):
      self.forward_propagate(X[i])
      diff = Y[i] - self.outputs
      error = np.dot(diff, diff)
      total_error += error
    return total_error/len(X)