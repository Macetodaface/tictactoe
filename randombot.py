from random import randint
from copy import deepcopy
from NN.neural_net import NeuralNet
import os.path
import sys
import numpy as np

class RandomBot:

    boards = []
    opp_boards = []
    structure = {"num_inputs": 90, 'num_hidden': 10, 'num_outputs': 1}
    learning_rate = 1
    NN = NeuralNet(structure, learning_rate)

    def get_move(self, pos, left):
        lmoves = pos.legal_moves()
        max_score = 0
        for (x, y) in lmoves:

            new_pos = deepcopy(pos)

            new_pos.make_move(x, y, self.myid)

            new_score = self.NN.forward_propagate(
                self.translate_macroboard(deepcopy(new_pos.board)+deepcopy(new_pos.macroboard),self.myid))
            if new_score > max_score:
                max_score = new_score
                best_move = (x, y)
                best_pos = new_pos

        self.boards.append(self.translate_macroboard(deepcopy(pos.board)+deepcopy(pos.macroboard),self.myid))
        self.opp_boards.append(
            self.translate_macroboard(deepcopy(best_pos.board)+ deepcopy(best_pos.macroboard),2-self.myid)
        )
        if self.log_data:
            self.save_data()
        return best_move

    def save_data(self):
        # Write data to disk
        with open("boards", "w") as f:
            f.write(repr(self.boards))
        with open("opp_boards", "w") as f:
            f.write(repr(self.opp_boards))

    def __init__(self, log_data=True):
        self.log_data = log_data
        # Get preexisting weight
        if os.path.isfile("weights"):
            print("Got weights from disk")
            with open("weights") as f:
                data = eval(f.read())
            self.NN.weightsList1 = np.array(data[0])
            self.NN.weightsList2 = np.array(data[1])
        if log_data:
            print("Data will be saved")
            self.train()

    def translate_macroboard(self,macroboard,myid):
        new_p1_value= 2*(1-myid)
        new_p2_value= 2*myid
        # for value in macroboard:
        macroboard = [new_p1_value if value==1 else new_p2_value if value==2 else 1 for value in macroboard]
        # macroboard[macroboard==1]=new_p1_value
        # macroboard[macroboard==2]=new_p2_value
        # macroboard[macroboard==0]=1
        # macroboard[macroboard==-1]=42
        return macroboard

    def translate_board(self,board,myid):
        new_p1_value= 2*(1-myid)
        new_p2_value= 2*myid
        board[board==1]=new_p1_value
        board[board==2]=new_p2_value
        board[board==0]=1
        return board

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
        print("winner was")
        print(winner)
        if winner == "nobody":
            if self.log_data:
                print("tie game")
            output = .5
        elif winner == "player1":
            output = 0
        else:
            output = 1
        iters = 10
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations=iters)

        # Train on p2's boards
        if output == 1 or output == 0:
            output = (output + 1) % 2

        with open("opp_boards") as f:
            inputs = eval(f.read())
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations=iters)
        weights = (self.NN.weightsList1.tolist(),
                   self.NN.weightsList2.tolist())

        # Record new weights
        with open("weights", "w") as f:
            f.write(repr(weights))
            print("Recorded new weights")