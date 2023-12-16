import numpy as np
import random
import copy
import json
import time
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivymd.utils import asynckivy
from kivy.core.window import Window
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
from kivy.graphics import Rectangle
import pickle



class checkers_game_with_graphics(App):

    def build(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.human = 1
        self.human_king = 2
        self.computer = 3
        self.computer_king = 4
        self.board_list = []
        self.fill_computer_side()
        self.fill_human_side()
        self.layout = GridLayout(cols=1)
        self.buttons = []
        self.showed_bottuns = []
        self.game_level = ''
        self.c1 = Computer(self.board)
        self.h1 = Human(self.board)
        str = np.array2string(self.board)
        str = self.change_str(str)
        self.board_list.append(str)
        print(self.board)
        label1 = Label(text='Welcome to the Checkers Game!')
        label2 = Label(text='Instructions:\n\n'
                            'In each turn you will need to move one of your tools\n'
                            ' diagonally to the right or left.\n'
                            ' The goal is to get one of your tools\n '
                            'to the other side of the board (to get a queen).\n'
                            'You will be playing with the red tools \n'
                            'and your opponent with the black tools.\n'
                            ' You will play the first turn of the game.\n'
                            'When you are ready, press continue\n'
                            ' and pick the level of your opponent.\n'
                            'Good Luck!!!')
        label1.font_size = 40
        label1.size_hint_y = 0.2
        label2.font_size = 25
        self.layout.add_widget(label1)
        self.layout.add_widget(label2)
        button1 = Button(text = 'continue',background_normal='blue_opening_background.jpg', font_size = 30)
        button1.bind(on_press = self.pick_level)
        button1.size_hint_y = 0.15
        self.layout.add_widget(button1)
        return self.layout


    def pick_level(self, instance):
        easy = Button(text='Easy',background_normal='test_background.jpg', size_hint=(1, None), height=200,font_size=60,color=(0, 0, 0, 1))
        medium = Button(text='Medium',background_normal='test_background1.jpg', size_hint=(1, None), height=200,font_size=60,color=(0, 0, 0, 1))
        hard = Button(text='Hard',background_normal='test_background2.jpg', size_hint=(1, None), height=200,font_size=60,color=(0, 0, 0, 1))
        hard.bind(on_press=self.play_hard)
        medium.bind(on_press=self.play_medium)
        easy.bind(on_press=self.play_easy)

        self.layout.add_widget(easy)
        self.layout.add_widget(medium)
        self.layout.add_widget(hard)

    def play_easy(self,instance):
        self.layout.clear_widgets()
        self.layout.cols = 8
        self.game_level = 'easy'
        self.create_board()

    def play_medium(self,instance):
        self.layout.clear_widgets()
        self.layout.cols = 8
        self.game_level = 'medium'
        self.create_board()

    def play_hard(self,instance):
        self.layout.clear_widgets()
        self.layout.cols = 8
        self.game_level = 'hard'
        self.create_board()

    def create_board(self):
        for i in range(8):
            for j in range(8):
                if i%2 == 0:
                    if j%2 == 0:
                        button = Button(background_normal='bright_background 1.jpg', font_size=1)
                    else:
                        if self.board[i][j] == self.human:
                            button = Button(background_normal='dark_human_background 1.jpg', font_size=1)
                        elif self.board[i][j] == self.computer:
                            button = Button(background_normal='dark_computer_background 1.jpg', font_size=1)
                        else:
                            button = Button(background_normal='dark_background 1.jpg', font_size=1)
                else:
                    if j % 2 == 0:
                        if self.board[i][j] == self.human:
                            button = Button(background_normal='dark_human_background 1.jpg', font_size=1)
                        elif self.board[i][j] == self.computer:
                            button = Button(background_normal='dark_computer_background 1.jpg', font_size=1)
                        else:
                            button = Button(background_normal='dark_background 1.jpg', font_size=1)
                    else:
                        button = Button(background_normal='bright_background 1.jpg', font_size=1)

                if self.board[i][j] == self.human:
                    button.bind(on_press=self.show_moves)
                    button.pos_hint = {'center_x': (i), 'center_y': (j)}

                self.layout.add_widget(button)
                self.buttons.append(button)
        self.buttons = [self.buttons[i:i + 8] for i in range(0, len(self.buttons), 8)]

    def show_moves(self, instance):
        for i in self.showed_bottuns:
            i.background_normal = 'dark_background 1.jpg'
            i.unbind(on_press=self.human_play)

        self.showed_bottuns = []
        options_list = self.h1.posible_moves()
        self.row = instance.pos_hint['center_x']
        self.cul = instance.pos_hint['center_y']
        for i in options_list:
            if i.begin_pos == (self.row,self.cul):
                self.show_move(i)

    def show_move(self, p1):
        tuple = p1.end_pos
        self.buttons[tuple[0]][tuple[1]].background_normal = ("green_background.jpg")
        self.showed_bottuns.append(self.buttons[tuple[0]][tuple[1]])
        self.buttons[tuple[0]][tuple[1]].bind(on_press=self.human_play)
        self.buttons[tuple[0]][tuple[1]].pos_hint = {'center_x': (tuple[0]), 'center_y': (tuple[1])}



    def human_play(self,instance):
        async def human_play():
            options_list = self.h1.posible_moves()
            self.end_row = instance.pos_hint['center_x']
            self.end_cul = instance.pos_hint['center_y']
            for i in options_list:
                if i.begin_pos == (self.row,self.cul) and i.end_pos == (self.end_row,self.end_cul):
                    p1 = i
                    break
            self.change_board(p1)
            self.check_human_kings()
            print(self.board)
            str = np.array2string(self.board)
            str = self.change_str(str)
            self.board_list.append(str)
            for i in self.showed_bottuns:
                i.background_normal = 'dark_background 1.jpg'
                i.unbind(on_press=self.human_play)
            self.showed_bottuns = []
            self.change_human_graphic_board(p1)
            self.buttons[p1.begin_pos[0]][p1.begin_pos[1]].unbind(on_press=self.show_moves)
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].bind(on_press=self.show_moves)
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].pos_hint = {'center_x': (p1.end_pos[0]), 'center_y': (p1.end_pos[1])}

            if self.check_human_win() == True:
                await asynckivy.sleep(3)
                self.win_message()
            else:
                await asynckivy.sleep(1)
                self.computer_play()

        asynckivy.start(human_play())

    def computer_play(self):
        async def computer_play():
            if self.game_level == 'easy':
                p1 = self.c1.play()
            if self.game_level == 'medium':
                '''m1 = network_MiniMax()
                p1 = m1.minimax(copy.deepcopy(self.board), 2, 0, True)
                p1 = p1[1]'''
                p1 = self.c1.smart_play()
            if self.game_level == 'hard':
                m1 = MiniMax()
                p1 = m1.minimax(copy.deepcopy(self.board), 2, 0, False)
                p1 = p1[1]

            self.change_board(p1)
            self.check_computer_kings()
            print(self.board)
            str = np.array2string(self.board)
            str = self.change_str(str)
            self.board_list.append(str)
            self.change_computer_graphic_board(p1)

            if self.check_computer_win() == True:
                await asynckivy.sleep(5)
                self.lose_message()

        asynckivy.start(computer_play())


    def change_human_graphic_board(self, p1):
        self.buttons[p1.begin_pos[0]][p1.begin_pos[1]].background_normal = 'dark_background 1.jpg'
        if self.board[p1.end_pos[0]][p1.end_pos[1]] == self.human_king:
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].background_normal = 'dark_human_king_background 1.jpg'
        else:
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].background_normal = 'dark_human_background 1.jpg'

        for i in p1.delete_pos:
            self.buttons[i[0]][i[1]].background_normal = 'dark_background 1.jpg'

    def change_computer_graphic_board(self, p1):
        self.buttons[p1.begin_pos[0]][p1.begin_pos[1]].background_normal = 'dark_background 1.jpg'
        if self.board[p1.end_pos[0]][p1.end_pos[1]] == self.computer_king:
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].background_normal = 'dark_computer_king_background 1.jpg'
        else:
            self.buttons[p1.end_pos[0]][p1.end_pos[1]].background_normal = 'dark_computer_background 1.jpg'

        for i in p1.delete_pos:
            self.buttons[i[0]][i[1]].background_normal = 'dark_background 1.jpg'
            self.buttons[i[0]][i[1]].unbind(on_press=self.show_moves)


    def win_message(self):
        async def win_message():
            self.layout.clear_widgets()
            win_label = Label(text="NICE JOB!\n"
                                   "YOU WON!", font_size=70)
            self.layout.add_widget(win_label)
            await asynckivy.sleep(5)
            Window.close()

        asynckivy.start(win_message())

    def lose_message(self):
        async def lose_message():
            self.layout.clear_widgets()
            lost_label = Label(text="NICE TRY! \n"
                                    "THIS TIME YOU LOST!", font_size=70)
            self.layout.add_widget(lost_label)
            await asynckivy.sleep(5)
            Window.close()

        asynckivy.start(lose_message())

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
        self.model = load_model("test_model1.h5")
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


    def create_possible_moves_list(self):
        self.posible_moves()
        self.dict_moves = []

        for i in self.list:
            board = copy.deepcopy(self.board)
            board = self.change_board(board, i)
            str = np.array2string(board)
            str = self.change_str(str)
            if str in self.dict:
                self.dict_moves.append((i, self.dict[str][0]))

    def smart_play(self):
        self.posible_moves()
        self.moves_with_rank = []
        for i in self.list:
            board = copy.deepcopy(self.board)
            board = self.change_board(board, i)
            board = board.reshape(-1, 64)
            #board = self.scale_board(board)
            prediction = self.model.predict(board)
            print(prediction)
            self.moves_with_rank.append((i, prediction))

        best_rank = -1
        for j in self.moves_with_rank:
            if j[1] > best_rank:
                best_rank = j[1]
                best_move = j[0]
        return best_move


    def scale_board(self, board):
        board = board.astype(float)
        for i in range(len(board)):
            board[i] = board[i]/4
        return board



    def change_str(self, str):
        str = str.replace("[", "")
        str = str.replace("]", "")
        str = str.replace(" ", "")
        str = str.replace("\n", "")
        return str


    def change_board(self, board, p1):
        num = board[p1.begin_pos[0]][p1.begin_pos[1]]
        board[p1.begin_pos[0]][p1.begin_pos[1]] = 0
        board[p1.end_pos[0]][p1.end_pos[1]] = num
        for i in p1.delete_pos:
            board[i[0]][i[1]] = 0

        for i in range(len(board)):
            if board[0][i] == 1:
                board[0][i] = 2
            if board[7][i] == 3:
                board[7][i] = 4

        return board

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


class network_MiniMax:#max player-True-computer,false-human
    def __init__(self):
        self.model = load_model("test_model1.h5")

    def get_num_pieces(self, board, num):
        counter = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == num:
                    counter+=1
        return counter

    def scale_board(self, board):
        board = board.astype(float)
        for i in range(len(board)):
            board[i] = board[i]/4
        return board


    def evaluate(self, board):
        board1 = copy.deepcopy(board)
        board1 = board1.reshape(-1, 64)
        #board1 = self.scale_board(board1)
        prediction = self.model.predict(board1)
        print(prediction)
        return prediction


    def minimax(self,board,depth, move, max_player):
        if depth == 0 or self.check_human_win(board) == True or self.check_computer_win(board) == True:
            return self.evaluate(board), move

        if max_player:
            maxEval = float('-inf')
            best_move = None
            c1 = Computer(board)
            all_moves = c1.posible_moves()
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
            h1 = Human(board)
            all_moves = h1.posible_moves()
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
        for i in range(len(board)):
            if board[0][i] == 1:
                board[0][i] = 2
            if board[7][i] == 3:
                board[7][i] = 4

        return board

    def check_human_win(self,board):
        if self.get_num_pieces(board, 2) >0:
            return True
        return False


    def check_computer_win(self,board):
        if self.get_num_pieces(board, 4) >0:
            return True
        return False


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
                board[0][i] = 2
            if board[7][i] == 3:
                board[7][i] = 4

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
file_path = "checkers_updated_game_new_dict.json"
with open(file_path, 'w') as json_file:
    json.dump(dict, json_file)
'''

if __name__ == "__main__":
    checkers_game_with_graphics().run()







