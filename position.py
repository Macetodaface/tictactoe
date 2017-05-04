
class Position:
    
    def __init__(self):
        self.board = []
        self.macroboard = []
    
    def parse_field(self, fstr):
        flist = fstr.replace(';', ',').split(',')
        self.board = [ int(f) for f in flist ]
    
    def parse_macroboard(self, mbstr):
        mblist = mbstr.replace(';', ',').split(',')
        self.macroboard = [ int(f) for f in mblist ]
    
    def is_legal(self, x, y):
        mbx, mby = x/3, y/3
        return self.macroboard[3*mby+mbx] == -1 and self.board[9*y+x] == 0

    def legal_moves(self):
        return [ (x, y) for x in range(9) for y in range(9) if self.is_legal(x, y) ]

    def make_move(self, x, y, pid):
        mb_new = 3*(y%3)+(x%3) #new macroboard
        mbx, mby = x/3, y/3 # old macroboard
        mb_old = mby*3 + mbx

        self.board[9*y+x] = pid


        winner = self.get_winner(mb_old)

        if self.macroboard[mb_new] <= 0: #if new is valid
            #invalidate others
            self.macroboard = [0 if n == -1 else n for n in self.macroboard]
            self.macroboard[mb_new] = -1
        else:
            self.macroboard = [-1 if n == 0 else n for n in self.macroboard]
        
        if winner != 0:
            # print("nonzero!!!!!")
            self.macroboard[mb_old] = winner

            
        #print("after", self.macroboard)

    def get_winner(self, mb_i):
        start_index = (mb_i/3)*27 + (mb_i % 3)*3
        board = self.board
        # check rows/columns
        for i in range(3):
            row_value = board[i*9+start_index]
            col_value = board[i+start_index]
            for j in range(3):
                if board[i*9+j+start_index] != row_value:
                    row_value = -1
                if board[j*9+i+start_index] != col_value:
                    col_value = -1
            if row_value > 0:
                return row_value
            if col_value > 0:
                return col_value

        # Check diagonals
        d1_val = board[start_index]
        d2_val = board[2+start_index]
        for i in [1, 2]:
            if board[i*10+start_index] != d1_val:
                d1_val = -1
            if board[i*8+2+start_index] != d2_val:
                d2_val = -1
        if d2_val > 0:
            return d2_val
        if d1_val > 0:
            return d1_val
        # print("zero")
        return 0

    def get_board(self):
        return ''.join(self.board, ',')

    def get_macroboard(self):
        return ''.join(self.macroboard, ',')

