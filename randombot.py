from random import randint
from copy import deepcopy
from NN.neural_net import NeuralNet
import os.path

class RandomBot:

    states = []
    structure = {"num_inputs": 81, 'num_hidden': 1, 'num_outputs': 1}
    learning_rate = .2
    NN = NeuralNet(structure, learning_rate)
    
    def get_move(self, pos, left):
        lmoves = pos.legal_moves()
        max_score = 0
        for move in lmoves:
            new_pos = deepcopy(pos)
            x = move / 9 #Backwards ???
            y = move % 9
            new_pos.make_move(x, y, self.myid)
            new_score = NeuralNet.forward_propogate(new_pos)
            if new_score > max_score:
                max_score = new_score
                best_move = move
                best_pos = new_pos

        self.states.append((deepcopy(pos), self.myid))
        self.states.append((deepcopy(best_pos), self.myid % 2 + 1))

        return (x, y)

    def save_data(self):
        # Write data to disk
        with open("inputs", "w") as f:
            f.write(repr(self.states))

    def __init__(self):
        if not os.path.isfile("weights"):
            return
        # Read current weights
        with open("weights") as f:
            data = eval(f.read())
        self.NN.weightsList1 = data[0]
        self.NN.weightsList2 = data[1]
        self.train()

    def train(self):
        if not os.path.isfile("inputs"):
            return
        return
        # Read training data
        with open("inputs") as f:
            inputs = eval(f.read())
        with open("winnerId") as f:
            winner = eval(f.read())
        self.NN.train(inputs, [winner]*len(inputs))
        weights = (self.NN.weightsList1, self.NN.weightsList2)
        with open("weights", "w") as f:
            f.write(repr(weights))