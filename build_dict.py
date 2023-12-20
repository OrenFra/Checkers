import numpy as np
import random
import copy
import json





class checkers_game:

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.human = 1
        self.human_king = 2
        self.computer = 3
        self.computer_king = 4
        self.board_list = []
        self.fill_computer_side()
        self.fill_human_side()


    def update_dict(self):
        file_path = "checkers_test_dict1.json"
        with open(file_path, 'r') as json_file:
            dict = json.load(json_file)

        k = 1
        for m in range(600000):
            if k % 10000 == 0:
                print(k)
            if k % 50000 == 0:
                with open(file_path, 'w') as json_file:
                    json.dump(dict, json_file)
            k+=1
            check1 = checkers_game()
            tuple = check1.play()
            if tuple[1] == False:
                for i in tuple[0]:
                    if i not in dict:
                        dict[i] = (0,1)
                    else:
                        average = float((dict[i][0]*dict[i][1])/(dict[i][1]+1))
                        dict[i] = (average, dict[i][1]+1)
            else:
                num = 1
                for i in reversed(tuple[0]):
                    if i not in dict:
                        dict[i] = (num, 1)
                    else:
                        average = float(((dict[i][0]*dict[i][1])+num)/(dict[i][1]+1))
                        dict[i] = (average, dict[i][1]+1)
                    num = num*0.9

        with open(file_path, 'w') as json_file:
            json.dump(dict, json_file)


    def fill_computer_side(self):
        for i in range(3):
            for j in range(8):
                if i%2 == 0:
                    if j%2 == 1:
                        self.board[i][j] = self.computer
                else:
                    if j%2 == 0:
                        self.board[i][j] = self.computer

    def fill_human_side(self):
        for i in range(5,8):
            for j in range(8):
                if i%2 == 1:
                    if j%2 == 0:
                        self.board[i][j] = self.human
                else:
                    if j%2 == 1:
                        self.board[i][j] = self.human

    def play(self):
        self.c1 = Computer(self.board)
        self.h1 = Human(self.board)
        human_win = False
        comp_win = False
        str = np.array2string(self.board)
        str = self.change_str(str)
        self.board_list.append(str)
        while human_win == False and comp_win == False:
            probability = random.random()
            if probability > 0.3:
                m1 = MiniMax()
                p1 = m1.minimax(copy.deepcopy(self.board), 2, 0, True)[1]
            else:
                p1 = self.h1.play()
            self.change_board(p1)
            self.check_human_kings()
            str = np.array2string(self.board)
            str = self.change_str(str)
            self.board_list.append(str)
            if self.check_human_win() == True:
                human_win = True
                break
            if self.check_computer_win() == True:
                comp_win = True
                break
            probability = random.random()
            if probability > 0.3:
                m2 = MiniMax()
                p1 = m2.minimax(copy.deepcopy(self.board), 2, 0, False)[1]
            else:
                p1 = self.c1.play()
            self.change_board(p1)
            self.check_computer_kings()
            str = np.array2string(self.board)
            str = self.change_str(str)
            self.board_list.append(str)
            if self.check_computer_win() == True:
                comp_win = True
                break
            if self.check_human_win() == True:
                human_win = True
                break

        if comp_win == True:
            return ((self.board_list, True))
        else:
            return ((self.board_list, False))


    def change_str(self, str):
        str = str.replace("[", "")
        str = str.replace("]", "")
        str = str.replace(" ", "")
        str = str.replace("\n", "")
        return str


    def change_board(self, p1):
        num = self.board[p1.begin_pos[0]][p1.begin_pos[1]]
        self.board[p1.begin_pos[0]][p1.begin_pos[1]] = 0
        self.board[p1.end_pos[0]][p1.end_pos[1]] = num
        for i in p1.delete_pos:
            self.board[i[0]][i[1]] = 0


    def check_computer_kings(self):
        for j in range(8):
            if self.board[7][j] == self.computer:
                self.board[7][j] = self.computer_king

    def check_human_kings(self):
        for j in range(8):
            if self.board[0][j] == self.human:
                self.board[0][j] = self.human_king

    def check_human_win(self):
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == self.human_king:
                    return True
        if self.c1.posible_moves() == []:
            return True
        return False

    def check_computer_win(self):
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == self.computer_king:
                    return True
        if self.h1.posible_moves() == []:
            return True
        return False


class Computer:
    def __init__(self, board):
        self.board = board
        self.human = 1
        self.human_king = 2
        self.computer = 3
        self.computer_king = 4


    def play(self):
        self.posible_moves()
        return random.choice(self.list)


    def posible_moves(self):
        self.list = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == self.computer:
                    self.move(i,j)
        return self.list


    def move(self, i, j):
        if i+1<8 and j-1>=0 :
            if self.board[i+1][j-1] == 0:
                p1 = Possition((i,j), (i+1, j-1))
                self.list.append(p1)
            if self.board[i+1][j-1] == self.human or self.board[i+1][j-1] == self.human_king:
                self.capture_move(i,j,'left')

        if i+1<8 and j+1<8:
            if self.board[i+1][j+1] == 0:
                p1 = Possition((i, j), (i + 1, j + 1))
                self.list.append(p1)
            if self.board[i+1][j+1] == self.human or self.board[i+1][j+1] == self.human_king:
                self.capture_move(i,j,'right')

    def capture_move(self, i, j, direction):
        if direction == 'right':
            if i+2<8 and j+2<8:
                if self.board[i+2][j+2] == 0:
                    p1 = Possition((i, j), (i + 2, j +2))
                    p1.delete_pos.append((i+1, j+1))
                    self.list.append(p1)
                    self.double_capture(i+2,j+2, copy.deepcopy(p1))
        else:
            if i+2<8 and j-2>=0:
                if self.board[i+2][j-2] == 0:
                    p1 = Possition((i, j), (i + 2, j - 2))
                    p1.delete_pos.append((i + 1, j - 1))
                    self.list.append(p1)
                    self.double_capture(i+2,j-2,copy.deepcopy(p1))

    def double_capture(self, i2, j2, p):
        if i2+1<8 and j2-1>=0:
            if self.board[i2+1][j2-1] == self.human or self.board[i2+1][j2-1] == self.human_king:
                if i2+2<8 and j2-2>=0:
                    if self.board[i2+2][j2-2] == 0:
                        p1 = copy.deepcopy(p)
                        p1.end_pos = (i2+2, j2-2)
                        p1.delete_pos.append((i2+1, j2-1))
                        self.list.append(p1)
                        self.double_capture(i2+2, j2-2, copy.deepcopy(p1))
        if i2+1<8 and j2+1<8:
            if self.board[i2+1][j2+1] == self.human or self.board[i2+1][j2+1] == self.human_king:
                if i2+2<8 and j2+2<8:
                    if self.board[i2+2][j2+2] == 0:
                        p1 = copy.deepcopy(p)
                        p1.end_pos = (i2 + 2, j2 + 2)
                        p1.delete_pos.append((i2 + 1, j2 + 1))
                        self.list.append(p1)
                        self.double_capture(i2+2, j2+2, copy.deepcopy(p1))


class Human:
    def __init__(self, board):
        self.board = board
        self.human = 1
        self.human_king = 2
        self.computer = 3
        self.computer_king = 4

    def play(self):
        self.posible_moves()
        return random.choice(self.list)


    def posible_moves(self):
        self.list = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == self.human:
                    self.move(i,j)
        return self.list

    def move(self, i, j):
        if i-1>=0 and j-1>=0 :
            if self.board[i-1][j-1] == 0:
                p1 = Possition((i,j), (i-1, j-1))
                self.list.append(p1)
            if self.board[i-1][j-1] == self.computer or self.board[i-1][j-1] == self.computer_king:
                self.capture_move(i,j,'left')

        if i-1>=0 and j+1<8:
            if self.board[i-1][j+1] == 0:
                p1 = Possition((i, j), (i - 1, j + 1))
                self.list.append(p1)
            if self.board[i-1][j+1] == self.computer or self.board[i-1][j+1] == self.computer_king:
                self.capture_move(i,j,'right')

    def capture_move(self, i, j, direction):
        if direction == 'right':
            if i-2>=0 and j+2<8:
                if self.board[i-2][j+2] == 0:
                    p1 = Possition((i, j), (i - 2, j + 2))
                    p1.delete_pos.append((i - 1, j + 1))
                    self.list.append(p1)
                    self.double_capture(i - 2, j + 2, copy.deepcopy(p1))
        else:
            if i-2>=0 and j-2>=0:
                if self.board[i-2][j-2] == 0:
                    p1 = Possition((i, j), (i - 2, j - 2))
                    p1.delete_pos.append((i - 1, j - 1))
                    self.list.append(p1)
                    self.double_capture(i - 2, j - 2, copy.deepcopy(p1))

    def double_capture(self, i2, j2, p):
        if i2-1>=0 and j2-1>=0:
            if self.board[i2-1][j2-1] == self.computer or self.board[i2-1][j2-1] == self.computer_king:
                if i2-2>=0 and j2-2>=0:
                    if self.board[i2-2][j2-2] == 0:
                        p1 = copy.deepcopy(p)
                        p1.end_pos = (i2 - 2, j2 - 2)
                        p1.delete_pos.append((i2 - 1, j2 - 1))
                        self.list.append(p1)
                        self.double_capture(i2 - 2, j2 - 2, copy.deepcopy(p1))
        if i2-1>=0 and j2+1<8:
            if self.board[i2-1][j2+1] == self.computer or self.board[i2-1][j2+1] == self.computer_king:
                if i2-2>=0 and j2+2<8:
                    if self.board[i2-2][j2+2] == 0:
                        p1 = copy.deepcopy(p)
                        p1.end_pos = (i2 - 2, j2 + 2)
                        p1.delete_pos.append((i2 - 1, j2 + 1))
                        self.list.append(p1)
                        self.double_capture(i2 - 2, j2 + 2, copy.deepcopy(p1))


class Possition:
    def __init__(self, begin, end):
        self.begin_pos = begin
        self.end_pos = end
        self.delete_pos = []


class MiniMax:#max player-True-human,false-computer side
    def __init__(self):
        return


    def get_num_pieces(self, board, num):
        counter = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == num:
                    counter+=1
        return counter


    def evaluate(self, board):
        player_pieces = self.get_num_pieces(board, 1)
        player_kings = self.get_num_pieces(board, 2)
        computer_pieces = self.get_num_pieces(board, 3)
        computer_kings = self.get_num_pieces(board, 4)
        return player_pieces - computer_pieces + (player_kings * 4 - computer_kings * 4)


    def minimax(self,board,depth, move, max_player):

        if depth == 0 or self.check_human_win(board) == True or self.check_computer_win(board) == True:
            return self.evaluate(board), move

        if max_player:
            maxEval = float('-inf')
            best_move = None
            h1 = Human(board)
            all_moves = h1.posible_moves()
            for pos in all_moves:
                board1 = self.change_board(copy.deepcopy(board),pos)
                evaluation = self.minimax(board1, depth - 1,pos, False)[0]
                maxEval = max(maxEval, evaluation)
                if maxEval == evaluation:
                    best_move = pos

            return maxEval, best_move
        else:
            minEval = float('inf')
            best_move = None
            c1 = Computer(board)
            all_moves = c1.posible_moves()
            for pos in all_moves:
                board1 = self.change_board(copy.deepcopy(board),pos)
                evaluation = self.minimax(board1, depth - 1,pos, True)[0]
                minEval = min(minEval, evaluation)
                if minEval == evaluation:
                    best_move = pos

            return minEval, best_move



    def change_board(self,board, p1):
        num = board[p1.begin_pos[0]][p1.begin_pos[1]]
        board[p1.begin_pos[0]][p1.begin_pos[1]] = 0
        board[p1.end_pos[0]][p1.end_pos[1]] = num
        for i in p1.delete_pos:
            board[i[0]][i[1]] = 0
        for i in range(len(board[0])):
            if board[0][i] == 1:
                board[0][i] == 2
            if board[7][i] == 3:
                board[7][i] == 4

        return board

    def check_human_win(self,board):
        if self.get_num_pieces(board,2) >0:
            return True
        return False


    def check_computer_win(self,board):
        if self.get_num_pieces(board,4) >0:
            return True
        return False



'''dict = {}
file_path = "checkers_test_dict1.json"
with open(file_path, 'w') as json_file:
    json.dump(dict, json_file)'''


if __name__ == "__main__":
    check = checkers_game()
    check.update_dict()








