from random import randint
from copy import deepcopy
from NN.neural_net import NeuralNet
import os.path
import sys
import numpy as np

class LoserBot:

    boards = []
    opp_boards = []
    structure = {"num_inputs": 9, 'num_hidden': 1, 'num_outputs': 1}
    learning_rate = 0.05
    NN = NeuralNet(structure, learning_rate)


    def evaluate_state(self,macroboard,board,myid):
        # try self.estimate_score(macroboard,myid) instead
        return self.forward_score(macroboard,myid)

    def estimate_score(self,board,myid):
        # greedy 'simple': return squares win - opponent squares won
        return sum(self.translate_macroboard(deepcopy(board),myid))

    def estimate_score_2(self, macroboard, board, myid):
        sum = self.estimate_score(macroboard, myid)*16
        for mb_i in range(9):
            sum -= self.estimate_score_2_help(
                self.translate_macroboard(board, myid), mb_i)
        return sum

    def estimate_score_2_help(self, board, mb_i):
        if len(board) < 4:
            print(board)
        sum = 0
        start_index = (mb_i/3)*27 + (mb_i % 3)*3
        # check rows/columns
        for i in range(3):
            row_val = board[i*9+start_index]
            col_val = board[i+start_index]
            for j in [1,2,0]:
                new_row_val = board[i*9+j+start_index]
                if new_row_val == row_val:
                    sum += 0 if row_val == 1 else -1 if row_val == 2 else 1
                row_val = new_row_val
                new_col_val = board[j*9+i+start_index]
                if new_col_val == col_val:
                    sum += 0 if col_val == 1 else -1 if col_val == 2 else 1
                col_val = new_col_val

        # Check diagonals
        d1_val = board[start_index]
        d2_val = board[2+start_index]
        for i in [1, 2, 0]:
            new_d1_val = board[i*10+start_index]
            if new_d1_val == d1_val:
                sum += 0 if d1_val == 1 else -1 if d1_val == 2 else 1
            d1_val = new_d1_val
            new_d2_val = board[i*8+2+start_index]
            if new_d2_val == d2_val:
                sum += 0 if d2_val == 1 else -1 if d2_val == 2 else 1
            d2_val = new_d2_val
        return sum

    def forward_score(self,board,myid):
        return self.NN.forward_propagate(self.translate_macroboard(deepcopy(board),myid))

    def get_max(self,n,myid,state,orig_move):
        lmoves = state.legal_moves()
        if len(lmoves) > 9:
            n-=1
        if n<=0 or len(lmoves)==0:
            return(self.evaluate_state(state.macroboard, state.board,
                                      myid), orig_move)
        else:
            new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            for new_state in new_states:
                new_state[0].make_move(new_state[1],new_state[2],myid)
            results = [self.get_min(n-1,myid,new_state[0],orig_move) for new_state in new_states]
            return max(results)

    def get_min(self,n,myid,state,orig_move):
        lmoves = state.legal_moves()
        if len(lmoves) > 9:
            n-=1
        if n<=0 or len(lmoves)==0:
            return (self.evaluate_state(state.macroboard, state.board,
                                          myid), orig_move)
        else:
            new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            for new_state in new_states:
                new_state[0].make_move(new_state[1],new_state[2],myid)
            results = [self.get_max(n-1,myid%2+1,new_state[0],orig_move) for new_state in new_states]
            return min(results)

    def get_move(self, pos, left):
        lmoves = pos.legal_moves()
        max_score = 0

        new_things = [(deepcopy(pos),move) for move in lmoves]
        for tup in new_things:
            tup[0].make_move(tup[1][0],tup[1][1],self.myid)
        tuples = [self.get_min(2,3-self.myid,tup[0],tup[1]) for tup in new_things]
        best_move = max(tuples)[1]
        best_pos = deepcopy(pos)
        best_pos.make_move(best_move[0],best_move[1],self.myid)
        self.boards.append(self.translate_macroboard(deepcopy(pos.macroboard),self.myid))
        self.opp_boards.append(
            self.translate_macroboard(deepcopy(best_pos.macroboard),self.myid%2+1)
        )
        if self.log_data:
            self.save_data()
        return best_move

    def save_data(self):
        # Write data to disk
        with open("boards", "w") as f:
            f.write(repr([self.boards[-1]]))
        with open("opp_boards", "w") as f:
            f.write(repr([self.opp_boards[-1]]))

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
        new_p1_value= 2 if myid==1 else 0
        new_p2_value= 2 if myid==2 else 0
        macroboard = [new_p1_value if value==1 else new_p2_value if value==2 else 1 for value in macroboard]
        return macroboard

    def translate_board(self,board,myid):
        new_p1_value= 2*(1-myid)
        new_p2_value= 2*myid
        board[board==1]=new_p1_value
        board[board==2]=new_p2_value
        board[board==0]=1
        return board


    def get_boards(self,thing):
        return np.array(eval(thing))
    def train(self):
        # Check for previous games
        if not os.path.isfile("boards"):
            print("No previous game found")
            return

        print("Training NN ...")

        # Train on p1's boards
        with open("boards") as f:
            inputs = self.get_boards(f.read())
        with open("winner.txt") as f:
            winner = f.read()
        print("winner was")
        print(winner)
        if winner == "nobody":
            print("tie game")
            output = .5
        #return
        if winner == "player1":
            output = 1
        else:
            output = 0
        iters = 5
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations=iters)

        # Train on p2's boards
        if output == 1 or output == 0:
            output = (output + 1) % 2
	
        with open("opp_boards") as f:
            inputs = self.get_boards(f.read())
        self.NN.train(np.array(inputs),
                      np.array([[output]]*len(inputs)),
                      iterations=iters)
        weights = (self.NN.weightsList1.tolist(),
                   self.NN.weightsList2.tolist())

        # Record new weights
        with open("weights", "w") as f:
            f.write(repr(weights))
            print("Recorded new weights")
