import unittest
import random
import numpy as np

import copy

from neural_net import NeuralNet

class TestNeuralNet(unittest.TestCase):

  def setUp(self):
    random.seed(0)
    np.random.seed(0)

  def test_weight_shapes(self):    
    learning_rate = 0.8
    structure = {'num_inputs': 2, 'num_outputs': 1, 'num_hidden': 5}
    candidate = NeuralNet(structure, learning_rate)

    cand_weights = candidate.get_weights()

    self.assertEqual(cand_weights[0].shape, (3, 5))
    self.assertEqual(cand_weights[1].shape, (5, 1))

  def test_forward_propagate(self):
    learning_rate = 0.8
    structure = {'num_inputs': 2, 'num_outputs': 1, 'num_hidden': 1}
    candidate = NeuralNet(structure, learning_rate)

    x = np.array([1, 0])
    cand_out = candidate.forward_propagate(x)

    expected_result = .500615025728

    print(cand_out)
    self.assertAlmostEqual(cand_out, expected_result, 4)

  def tet_backward_propagate(self):
    learning_rate = 0.8
    structure = {'num_inputs': 2, 'num_outputs': 1, 'num_hidden': 1}
    candidate = NeuralNet(structure, learning_rate)

    cand_weights = candidate.get_weights()

    X = np.array([np.array([1, 0])])
    Y = np.array([np.array([0])])
    candidate.train(X, Y)

    cand_weights = candidate.get_weights()
    print(cand_weights)


    # You can do the math to see what the new weights should be
    # and assert them here.

    self.assertTrue(True)

  def test_xor(self):
    learning_rate = .2
    structure = {'num_inputs': 2, 'num_hidden': 2, 'num_outputs': 1}
    candidate = NeuralNet(structure, learning_rate)

    labeled_data = [
        (np.array([0,0]), np.array([0])),
        (np.array([0,1]), np.array([1])),
        (np.array([1,0]), np.array([1])),
        (np.array([1,1]), np.array([0]))
    ]

    iterations = 15000

    trainX, trainY = zip(*labeled_data)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    candidate.train(trainX, trainY, iterations)

    cand_error = candidate.test(trainX, trainY)
    print "XOR Error: ", cand_error

if __name__ == '__main__':
  testSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralNet)
  testRunner = unittest.TextTestRunner(descriptions=True, verbosity=2)

  testResult = testRunner.run(testSuite)