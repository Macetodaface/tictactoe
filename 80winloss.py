from random import randint
from copy import deepcopy
from NN.neural_net import NeuralNet
import os.path
import sys
import numpy as np

class RandomBot:

    boards = []
    opp_boards = []
    structure = {"num_inputs": 9, 'num_hidden': 2, 'num_outputs': 1}
    learning_rate = 0.05
    NN = NeuralNet(structure, learning_rate)


    def estimate_score(self,board,myid):
        # print(board)
        return sum(self.translate_macroboard(deepcopy(board),myid))
        # return sum(board)

    def forward_score(self,board,myid):
        # print(board)
        # print(self.NN.forward_propagate(self.translate_macroboard(deepcopy(board),myid)))
        return self.NN.forward_propagate(self.translate_macroboard(deepcopy(board),myid))

    def get_max(self,n,myid,state,orig_move):
        #print(state.macroboard)
        # print(len(state.legal_moves()))
        lmoves = state.legal_moves()
        if len(lmoves) > 9:
            n-=1
        if n<=0 or len(lmoves)==0:
            return(self.forward_score(state.macroboard,myid),orig_move)
        else:
            # new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            new_states=[]
            new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            for new_state in new_states:
                new_state[0].make_move(new_state[1],new_state[2],myid)
            results = [self.get_min(n-1,myid%2+1,new_state[0],orig_move) for new_state in new_states]
            return max(results)

    def get_min(self,n,myid,state,orig_move):
        # print(len(state.legal_moves()))
        lmoves = state.legal_moves()
        if len(lmoves) > 9:
            n-=1
        if n<=0 or len(lmoves)==0:
            return(self.forward_score(state.macroboard,myid),orig_move)
        else:
            # new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            new_states=[(deepcopy(state),move[0],move[1]) for move in lmoves]
            for new_state in new_states:
                new_state[0].make_move(new_state[1],new_state[2],myid)
            results = [self.get_max(n-1,myid%2+1,new_state[0],orig_move) for new_state in new_states]
            return min(results)





    def get_move(self, pos, left):
        lmoves = pos.legal_moves()
        max_score = 0


        # best_move = self.get_max(1,self.myid,)
        new_things = [(deepcopy(pos),move) for move in lmoves]
        for tup in new_things:
            tup[0].make_move(tup[1][0],tup[1][1],self.myid)
        # print(new_things)
        tuples = [self.get_min(1,3-self.myid,tup[0],tup[1]) for tup in new_things]
        best_move = min(tuples)[1]
        best_moves = []
        best_score = 999999999
        for tup in tuples:
            score = tup[0]
            if score == best_score:
                best_moves.append(tup[1])
            if score < best_score:
                best_moves = [tup[1]]
            best_score = score
        rand = randint(0, len(best_moves)-1)
        best_move = best_moves[len(best_moves)/2]
        #best_move = max(best_moves)

        best_pos = deepcopy(pos)
        # print("b4")
        # print(best_pos.macroboard)
        best_pos.make_move(best_move[0],best_move[1],self.myid)
        # print(best_pos.macroboard)
        # for (x, y) in lmoves:

        #     new_pos = deepcopy(pos)

        #     new_pos.make_move(x, y, self.myid)

        #     # new_score = self.NN.forward_propagate(
        #         # self.translate_macroboard(deepcopy(new_pos.board)+deepcopy(new_pos.macroboard),self.myid))
        #     new_score = self.NN.forward_propagate(
        #         self.translate_macroboard(deepcopy(new_pos.macroboard),self.myid))

        #     if new_score > max_score:
        #         max_score = new_score
        #         best_move = (x, y)
        #         best_pos = new_pos

        # self.boards.append(self.translate_macroboard(deepcopy(pos.board)+deepcopy(pos.macroboard),self.myid))
        # self.opp_boards.append(
        #     self.translate_macroboard(deepcopy(best_pos.board)+ deepcopy(best_pos.macroboard),2-self.myid)
        # )
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
        new_p1_value= 2 if myid==1 else 0
        new_p2_value= 2 if myid==2 else 0
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
            print("tie game")
            output = .5
        return
        if winner == "player1":
            output = 0
        elif winner == "player2":
            output = 1
        iters = 5
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