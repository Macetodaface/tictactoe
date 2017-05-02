from random import randint
from copy import deepcopy
from neural_net import NeuralNet

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
