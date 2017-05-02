from main import parse_command
from randombot import RandomBot
from position import Position

def run():
    bot1 = RandomBot()
    bot2 = RandomBot()
    bots = [bot1, bot2]
    pos = Position()
    send_settings(bot1, pos)
    send_settings(bot2, pos)
    time_limit = 10000
    while not is_game_over():
        bot1.parse_command("action move " + str(time_limit))
        if (is_game_over(pos)):
            break
        bot2.parse_command("action move " + str(time_limit))
    return get_winner(pos)

def is_game_over(pos):
    return get_winner(pos) != -1

def get_winner(pos):
    board = pos.macroboard
    
    #check rows/columns
    for i in range(3):
        row_value = board[i*3]
        col_value = board[j*3]
        for j in range(3):
            if board[i*3 + j] != row_value:
                row_value = -1
            if board[j*3 + i] != col_value:
                col_value = -1
        if row_value > 0:
            return row_value
        if col_value > 0:
            return col_value
        
    #Check diagonals
    d1_val = board[0]
    d2_val = board[2]
    for i in [1, 2]:
        if board[i*3 + j] != d1_val:
            d1_val = -1
        if board[i*3 + 2-j] != d2_val:
            d2_val = -1
    if d2_val > 0:
        return d2_val
    if d1_val > 0:
        return d1_val

    return -1

def send_settings(bot, pos):
    instructions = ["settings timebank 10000",
    "settings time_per_move 500",
    "settings player_names player1,player2",
    "settings your_bot player1",
    "settings your_botid 1"]
    for instruction in instructions:
        parse_command(instruction, bot, pos)
    return
