
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
        mb_i = 3*(y%3)+(x%3)
        self.macroboard[mb_i] = self.get_winner(mb_i*9)

        mbx, mby = x/3, y/3
        if self.macroboard[3*mby+mbx] == 0:
            self.macroboard = [0 if n == -1 else -1 for n in self.macroboard]
        self.macroboard[3*mby+mbx] = -1
        self.board[9*y+x] = pid

    def get_winner(self, startIndex):
        board = self.board
        # check rows/columns
        for i in range(3):
            row_value = board[i*3 + startIndex]
            col_value = board[i + startIndex]
            for j in range(3):
                if board[i*3+j+startIndex] != row_value:
                    row_value = -1
                if board[j*3+i+startIndex] != col_value:
                    col_value = -1
            if row_value > 0:
                return row_value
            if col_value > 0:
                return col_value

        # Check diagonals
        d1_val = board[0]
        d2_val = board[2]
        for i in [1, 2]:
            if board[i*3+j+startIndex] != d1_val:
                d1_val = -1
            if board[i*3+2-j + startIndex] != d2_val:
                d2_val = -1
        if d2_val > 0:
            return d2_val
        if d1_val > 0:
            return d1_val

        return 0

    def get_board(self):
        return ''.join(self.board, ',')

    def get_macroboard(self):
        return ''.join(self.macroboard, ',')
    