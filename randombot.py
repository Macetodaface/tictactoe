from random import randint
from copy import deepcopy
from NN.neural_net import NeuralNet
import os.path
import sys
import numpy as np

class RandomBot:

    boards = []
    opp_boards = []
    macroboards = []
    opp_macroboards = []
    structure = {"num_inputs": 81, 'num_hidden': 1, 'num_outputs': 1}
    learning_rate = .2
    NN = NeuralNet(structure, learning_rate)

    def get_move(self, pos, left):
        lmoves = pos.legal_moves()
        max_score = 0
        for move in lmoves:

            new_pos = deepcopy(pos)

            x = lmoves[0][0]
            y = lmoves[0][1]

            new_pos.make_move(x, y, self.myid)

            new_score = self.NN.forward_propagate(new_pos.macroboard)
            if new_score > max_score:
                max_score = new_score
                best_move = (x, y)
                best_pos = new_pos

        self.boards.append(deepcopy(pos.board))
        self.macroboards.append(deepcopy(pos.macroboard))
        self.opp_boards.append(deepcopy(best_pos.board))
        self.opp_macroboards.append(deepcopy(best_pos.macroboard))
        self.save_data()
        return best_move

    def save_data(self):
        # Write data to disk
        with open("boards", "w") as f:
            f.write(repr(self.boards))
        with open("opp_boards", "w") as f:
            f.write(repr(self.opp_boards))
        #print("Saving states ...")

    def __init__(self):
        # Get preexisting weights
        if os.path.isfile("weights"):
            print("Got weights from disk")
            with open("weights") as f:
                data = eval(f.read())
            self.NN.weightsList1 = np.array(data[0])
            self.NN.weightsList2 = np.array(data[1])
        self.train()

    def train(self):
        # Check for previous games
        if not os.path.isfile("boards"):
            print("No previous game found")
            return

        print("Training NN ...")

        # Train on p1's boards
        with open("boards") as f:
            inputs = eval(f.read())
        with open("winner.txt") as f:
            winner = f.read()
        output = 1 if winner is "player1" else 0

        iters = 100
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations = iters)
        output = (output + 1) % 2

        # Train on p2's boards
        with open("opp_boards") as f:
            inputs = eval(f.read())
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations = iters)
        weights = (self.NN.weightsList1.tolist(),
                   self.NN.weightsList2.tolist())

        #Record new weights
        with open("weights", "w") as f:
            f.write(repr(weights))
            print("Recorded new weights")