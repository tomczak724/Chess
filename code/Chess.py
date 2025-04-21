
import os
import pdb
import sys
import copy
import time
import json
import glob
import numpy
import pandas
import shutil
import imageio
import matplotlib
import multiprocessing
from matplotlib import pyplot

import playground_cython

CPU_COUNT = multiprocessing.cpu_count()

RANKS = [1, 2, 3, 4, 5, 6, 7, 8]
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

DX = 40
IMAGE_PIECES = imageio.imread('../data/chess_pieces.png')

IMAGE_BLANK = IMAGE_PIECES[99:101, 99:101, :]

PIECES_WHITE = {1:IMAGE_PIECES[162-DX:162+DX, 683-DX:683+DX, :], 
                2:IMAGE_PIECES[164-DX:164+DX, 557-DX:557+DX, :], 
                3:IMAGE_PIECES[160-DX:160+DX, 431-DX:431+DX, :], 
                4:IMAGE_PIECES[162-DX:162+DX, 305-DX:305+DX, :], 
                5:IMAGE_PIECES[161-DX:161+DX, 179-DX:179+DX, :], 
                6:IMAGE_PIECES[161-DX:161+DX, 54-DX:54+DX, :]}

PIECES_BLACK = {-1:IMAGE_PIECES[55-DX:55+DX, 682-DX:682+DX, :], 
                -2:IMAGE_PIECES[55-DX:55+DX, 557-DX:557+DX, :], 
                -3:IMAGE_PIECES[53-DX:53+DX, 430-DX:430+DX, :], 
                -4:IMAGE_PIECES[55-DX:55+DX, 305-DX:305+DX, :], 
                -5:IMAGE_PIECES[54-DX:54+DX, 178-DX:178+DX, :], 
                -6:IMAGE_PIECES[55-DX:55+DX, 53-DX:53+DX, :]}

MATERIAL_VALUES = {'P':1, 'p':1, 'N':3, 'n':3, 'B':3, 'b':3, 'R':5, 'r':5, 'Q':9, 'q':9, 'K':9999, 'k':9999, 1:1, -1:1, 2:3, -2:3, 3:3, -3:3, 4:5, -4:5, 5:9, -5:9, 6:9999, -6:9999}

PIECE_CHAR_2_ID = {' ':0, 'P':1, 'p':-1, 'N':2, 'n':-2, 'B':3, 'b':-3, 'R':4, 'r':-4, 'Q':5, 'q':-5, 'K':6, 'k':-6}
PIECE_ID_2_CHAR = {0:' ', 1:'P', -1:'p', 2:'N', -2:'n', 3:'B', -3:'b', 4:'R', -4:'r', 5:'Q', -5:'q', 6:'K', -6:'k'}

CENTER_WEIGHT_MAP = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                                 [1, 2, 2, 2, 2, 2, 2, 1], 
                                 [1, 2, 3, 3, 3, 3, 2, 1], 
                                 [1, 2, 3, 4, 4, 3, 2, 1], 
                                 [1, 2, 3, 4, 4, 3, 2, 1], 
                                 [1, 2, 3, 3, 3, 3, 2, 1], 
                                 [1, 2, 2, 2, 2, 2, 2, 1], 
                                 [1, 1, 1, 1, 1, 1, 1, 1]])
CENTER_WEIGHT_MAP = CENTER_WEIGHT_MAP / CENTER_WEIGHT_MAP.sum()

OTHER_PLAYER = {'white':'black', 'black':'white'}
PLAYER_2_INT = {'white':1, 'w':1, 'black':-1, 'b':-1}


class chessBoard(object):

    def __init__(self, pgn_file=None):


        self.pgn_file = pgn_file
        self.list_moves = []


        ###  setting up chess board and pieces
        self.chess_boards = [numpy.array([[ 4,  2,  3,  5,  6,  3,  2,  4],
                                          [ 1,  1,  1,  1,  1,  1,  1,  1],
                                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                                          [-1, -1, -1, -1, -1, -1, -1, -1],
                                          [-4, -2, -3, -5, -6, -3, -2, -4]])]

        self.list_eval_metrics = [self.calc_evaluation_metrics(self.chess_boards[-1])]

        ###  setting up lists of legal moves
        self.halfmove_count = 0
        self.ep_target = 0
        self.fens = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1']
        self.promotion_prompt = False

        self.player_on_move = 1 # 1=white, -1=black
        self.castle_rights = 'KQkq'
        self.list_possible_moves = []


        ###  initializing figure
        self.fig, self.ax = pyplot.subplots(figsize=(7, 7))
        self.fig.subplots_adjust(left=0.05, top=0.98, right=0.98, bottom=0.05)
        self.ax.axis([0.5, 8.5, 0.5, 8.5])
        self.ax.set_aspect('equal')
        self.ax.xaxis.set_tick_params(size=0)
        self.ax.yaxis.set_tick_params(size=0)
        self.ax.xaxis.set_ticks(numpy.arange(1, 9, 1))
        self.ax.yaxis.set_ticks(numpy.arange(1, 9, 1))
        self.ax.xaxis.set_ticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], size=15)
        self.ax.yaxis.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8'], size=15)

        ###  plotting squares of the board
        for i in range(0, 8, 2):
            self.ax.axvline(i+0.5, color='k', lw=1)
            self.ax.axhline(i+0.5, color='k', lw=1)
            self.ax.axhline(i+1.5, color='k', lw=1)
            self.ax.axvline(i+1.5, color='k', lw=1)
            self.ax.axhline(i+2.5, color='k', lw=1)
            self.ax.axvline(i+2.5, color='k', lw=1)
            for j in range(0, 8, 2):
                self.ax.fill_between([i+0.5, i+1.5], j+0.5, j+1.5, color='k', alpha=0.25)
                self.ax.fill_between([i+1.5, i+2.5], j+1.5, j+2.5, color='k', alpha=0.25)

        ###  setting up board of axes subplots
        self.image_board = {}
        for i_rank, y0 in enumerate(numpy.arange(0.05, 0.98, 0.93/8)):
            rank = {}
            for i_file, x0 in enumerate(numpy.arange(0.05, 0.98, 0.93/8)):

                ###  labeling image squares by [file][rank] (e.g. 45 represents e5)
                ax = self.fig.add_axes([x0, y0, 0.93/8, 0.93/8], label='%i%i'%(i_file+1, i_rank+1))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.axes.set_facecolor('none')

                imdat = ax.imshow(IMAGE_BLANK, origin='upper')

                rank.update({i_file:imdat})

            self.image_board.update({i_rank:rank})

        ###  plotting images of pieces
        for r in range(8):
            for f in range(8):
                if self.chess_boards[-1][r][f] in PIECES_WHITE.keys():
                    self.image_board[r][f].set_data(PIECES_WHITE[self.chess_boards[-1][r][f]])
                if self.chess_boards[-1][r][f] in PIECES_BLACK.keys():
                    self.image_board[r][f].set_data(PIECES_BLACK[self.chess_boards[-1][r][f]])


        ###  instantiating interactive variables
        self.selected_square = None


        ###  playing game against self
        if False:

            timestamps = [time.time()]
            while True:

                list_legal_moves = self.get_legal_moves(self.player_on_move, self.chess_boards[-1], get_eval=True)

                print(time.ctime())
                if self.player_on_move == 'white':
                    print('\nTop three moves for white:')
                    list_legal_moves_sorted = list_legal_moves.sort_values(by=['evaluation'],ascending=False)
                    print(list_legal_moves_sorted.head(3))
                else:
                    print('\nTop three moves for black:')
                    list_legal_moves_sorted = list_legal_moves.sort_values(by=['evaluation'],ascending=True)
                    print(list_legal_moves_sorted.head(3))

                ###  applying top move
                self.move_piece(list_legal_moves_sorted.iloc[0]['notation'], self.player_on_move, board=None, redraw=True)
                self.player_on_move = OTHER_PLAYER[self.player_on_move]

                print('\nPGN')
                self.print_png_body()
                print('')
                timestamps.append(time.time())
                print('time to calc move  = %.1f minutes' % ((timestamps[-1]-timestamps[-2])/60.))
                print('total time elapsed = %.1f minutes' % ((timestamps[-1]-timestamps[0])/60.))
                print('')


        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)


    def _onClick(self, event):

        if (event.inaxes is not None):

            ###  unhighlight all squares
            for r in range(8):
                for f in range(8):
                    self.image_board[r][f].axes.set_facecolor('none')


            ###  if clicked square is the currently-selected square then truncate
            clicked_square = event.inaxes.get_label()
            print('CLICKED SQUARE: ', clicked_square)
            f = int(clicked_square[0]) - 1
            r = int(clicked_square[1]) - 1
            if clicked_square == self.selected_square:
                self.selected_square = None
                self.list_possible_moves = []
                self._redraw_board()


            ###  handling pawn promotion prompt
            elif self.promotion_prompt is True:

                if clicked_square in self.promotion_options.keys():
                    search_str_white = '%s8=%s' % (f, self.promotion_options[clicked_square])
                    search_str_black = '%s1=%s' % (f, self.promotion_options[clicked_square])
                    idx_search = self.df_possible_moves['notation'].apply(lambda row: (search_str_white in row) or (search_str_black in row))
                    move_text = self.df_possible_moves.loc[idx_search, 'notation'].iloc[0]
                    self.move_piece(move_text, self.player_on_move, board=None, redraw=True)

                    self.selected_square = None
                    self.df_possible_moves = pandas.DataFrame()
                    self.promotion_options = {}
                    self.promotion_prompt = False
                    self.toggle_transparency('off')
                    self.player_on_move = OTHER_PLAYER[self.player_on_move]

            elif (len(self.list_possible_moves) > 0) and (clicked_square in ','.join([str(move[2]) for move in self.list_possible_moves])):

                ###  if pawn promotion then prompt for which piece
                ###  handling promotions for white
                r0 = int(self.selected_square) % 10 - 1
                f0 = (int(self.selected_square) - r0) // 10 - 1
                if (self.chess_boards[-1][r0][f0] == 1) and (r == 8-1):
                    self.promotion_prompt = True
                    self.promotion_options = {}
                    self.toggle_transparency('on')
                    self.image_board[7][f0].set_data(IMAGE_BLANK)
                    for ri, pi in zip([8, 7, 6, 5], [5, 2, 4, 3]):
                        self.image_board[ri][f].axes.set_facecolor('w')
                        self.promotion_options.update({'%s%i'%(f, ri): pi})
                        self.image_board[ri][f].set_data(PIECES_WHITE[pi])
                        self.image_board[ri][f].set_alpha(1)
                    pyplot.draw()

                ###  handling promotions for black
                elif (self.chess_boards[-1][int(r0)][f0] == -1) and (r == 1-1):
                    self.promotion_prompt = True
                    self.promotion_options = {}
                    self.toggle_transparency('on')
                    self.image_board[2][f0].set_data(IMAGE_BLANK)
                    for ri, pi in zip([1, 2, 3, 4], [-5, -2, -4, -3]):
                        self.image_board[ri][f].axes.set_facecolor('w')
                        self.promotion_options.update({'%s%i'%(f, ri): pi})
                        self.image_board[ri][f].set_data(PIECES_BLACK[pi])
                        self.image_board[ri][f].set_alpha(1)
                    pyplot.draw()

                ###  else moving piece to clicked square
                else:

                    ###  saving evaluation metrics
                    self.list_eval_metrics.append(self.calc_evaluation_metrics(self.chess_boards[-1]))

                    for move in self.list_possible_moves:
                        if move[2] == int(clicked_square):
                            break

                    self.move_piece(move, self.chess_boards[-1], redraw=True)
                    self._redraw_board()

                    self.selected_square = None
                    self.list_possible_moves = []
                    self.player_on_move *= -1


                    ###  printing top move to play next
                    if self.player_on_move == 'black' and False:
                        t0 = time.time()
                        list_legal_moves = self.get_legal_moves(self.player_on_move, self.chess_boards[-1], get_eval=True)
                        tf = time.time()
                        #list_legal_moves.to_csv('list_legal_moves_%03i.csv' % self.df_moves.index[-1], index=False)
                        print('')
                        print('time elapsed = %.1f minutes' % ((tf-t0)/60.))
                        if self.player_on_move == 'white':
                            list_legal_moves_sorted = list_legal_moves.sort_values(by=['evaluation'],ascending=False)
                            print(list_legal_moves_sorted.head(3))
                        else:
                            list_legal_moves_sorted = list_legal_moves.sort_values(by=['evaluation'],ascending=True)
                            print(list_legal_moves_sorted.head(3))

                        ###  applying top move
                        self.move_piece(list_legal_moves_sorted.iloc[0]['notation'], self.player_on_move, board=None, redraw=True)
                        self.player_on_move = OTHER_PLAYER[self.player_on_move]



            ###  if selecting a piece, get legal moves, highlight squares
            else:

                list_legal_moves = self.get_legal_moves(self.player_on_move, self.chess_boards[-1], get_eval=False)
                self.selected_square = event.inaxes.get_label()
                self.list_possible_moves = [row for row in list_legal_moves if row[1]==int(clicked_square)]

                ###  highlighting squares for player on move
                occupant = self.chess_boards[-1][r][f]
                if (self.player_on_move == 1 and occupant in PIECES_WHITE.keys()) or ((self.player_on_move == -1 and occupant in PIECES_BLACK.keys())):

                    for move in self.list_possible_moves:
                        r_end = move[2] % 10 - 1
                        f_end = (move[2] - r_end) // 10 - 1
                        self.image_board[r_end][f_end].axes.set_facecolor('#ff9999')

                self._redraw_board()


    def toggle_transparency(self, on_off):

        if on_off == 'on':
            self.ax.set_facecolor('gray')
            for r in range(1, 8+1):
                for f in range(1, 8+1):
                    self.image_board[r][f].set_alpha(0.5)

        elif on_off == 'off':
            self.ax.set_facecolor('none')
            for r in range(1, 8+1):
                for f in range(1, 8+1):
                    self.image_board[r][f].set_alpha(1)


    def load_fen(self, fen):

        #r3k2r/3n1p1p/p3p1p1/1b1pP3/2nP1P2/2qB1Q1P/2P1N1PK/R1B2R2 b kq - 4 22
        ###
        ###  Mate in two:
        ###  r3r1k1/p1p2ppp/1p4q1/8/1PP5/P7/4QPPP/R3R1K1 w - - 0 1
        ###
        ###  Mate in one:
        ###  5r1k/6pp/8/8/8/8/6PP/5R1K w - - 0 1
        ###
        ###  Position where top 3 engine moves are eval~0 and rest are eval<-2.5
        ###  r3k2r/pbpnqp2/1p1ppn1p/8/2PPP1p1/3B1NB1/P1P1QPPP/R4RK1 w kq - 0 13

        fen_partitions = fen.strip().split(' ')
        fen_board = fen_partitions[0]
        fen_player_on_move = fen_partitions[1]
        fen_castle_rights = fen_partitions[2]
        fen_ep_target = fen_partitions[3]
        fen_halfmove_count = fen_partitions[4]
        fen_move_count = fen_partitions[5]

        board = {}
        for i_rank, fen_rank in enumerate(fen_board.split('/')):

            i_file = 0
            rank = {}
            for char in fen_rank:

                if char in ['1', '2', '3', '4', '5' ,'6', '7', '8']:

                    for i in range(int(char)):
                        rank[i_file+1] = 0
                        i_file += 1

                else:
                    rank[i_file+1] = PIECE_CHAR_2_ID[char]
                    i_file += 1

            board[8-i_rank] = rank


        self.chess_boards = [board]
        self.player_on_move = {'w':1, 'b':-1}[fen_player_on_move]
        self.castle_rights = fen_castle_rights
        self.halfmove_count = int(fen_halfmove_count)

        #i = len(self.df_moves)
        #self.df_moves.loc[i, 'ep_target'] = fen_ep_target

        self._redraw_board()


    def get_fen(self, board, board_prev, player_on_move):
        '''
        Parameters
        ----------
            board : 2d-array of ints
                Full layout of chessboard
                0 = vacant square
                1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
                -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

            board_prev : 2d-array of ints
                Full layout of chessboard from previous position.
                Used for determining en passant targets.

        Example FEN
        -----------
            r3k2r/3n1p1p/p3p1p1/1b1pP3/2nP1P2/2qB1Q1P/2P1N1PK/R1B2R2 b kq - 4 22

        '''

        ###  adding peice locations to FEN string
        fen = self.generate_fen_board_position(board)

        ###  adding next player to FEN string
        fen += ' %s ' % {1:'w', -1:'b'}[player_on_move]

        ###  adding castling rights to FEN string
        fen += ' %s ' % self.castle_rights

        ###  adding available en passant target
        ###  identifying which piece moved between which squares
        ###  NOTE: Castling will produce 4 differences
        ###        En Passant will produce 3 differences
        ###        All other will produce 2 differences
        diffs = []
        for r in RANKS:
            for f in FILES:
                if board[r][f] != board_prev[r][f]:
                    diffs.append('%s%i' % (f, r))


    def generate_fen_board_position(self, board):
        '''
        Description
        -----------
            Returns the board position portion of an FEN string

        Parameters
        ----------
            board : dict
                Layout of board to generate FEN of
        '''

        fen_board_position = ''
        n_vacant_files = 0
        for r in range(7, -1, -1):

            n_vacant = 0
            for f in range(8):

                ###  if square is empty, increment vacant count
                if board[r][f] == 0:
                    n_vacant += 1

                else:
                    ###  if n_vacant is not 0, add to FEN string
                    if n_vacant > 0:
                        fen_board_position += '%i' % n_vacant
                        n_vacant = 0

                    ###  add piece to FEN string
                    fen_board_position += PIECE_ID_2_CHAR[board[r][f]]

            if n_vacant > 0:
                fen_board_position += '%i' % n_vacant

            if r > 1:
                fen_board_position += '/'

        return fen_board_position


    def move_piece(self, move, board, redraw=False):
        '''
        Parameters
        ----------
            move : array of ints
                Array of moves to be applied where:
                column 1 : ID of piece is moving
                column 2 : Start square of piece (e.g. 34 for "c4")
                column 3 : End square of piece (e.g. 67 for "f7")
                column 4 : Capture flag (0=False, 1=True, 2=en passant)
                column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
                column 6 : Promotion flag (ID of piece being promoted to)

                Note: 100 represents king-side castling and 1000
                represents queen-side castling

            board : 2d-array of ints
                Current layout of chessboard
                0 = vacant square
                1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
                -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

            redraw : bool
                Whether or not to redraw the peices on the image board

        Returns
        -------
            new_board : 2d-array of ints
                Layout of chessboard after piece is moved
                0 = vacant square
                1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
                -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k
        '''

        ###  generating new board position
        new_board = playground_cython.move_piece(numpy.array(move), board)

        ###  parsing start / end squares and piece being moved
        r_start = move[1] % 10 - 1
        f_start = (move[1] - r_start) // 10 - 1
        r_end = move[2] % 10 - 1
        f_end = (move[2] - r_start) // 10 - 1


        ###  redraw plot window and appending new board to history
        if redraw == True:

            ###  updating castling rights
            if move[0] == 6:
                self.castle_rights = self.castle_rights.replace('K', '')
                self.castle_rights = self.castle_rights.replace('Q', '')
            elif move[0] == -6:
                self.castle_rights = self.castle_rights.replace('k', '')
                self.castle_rights = self.castle_rights.replace('q', '')
            elif (move[0] == 4) and (f_start == 0):
                self.castle_rights = self.castle_rights.replace('Q', '')
            elif (move[0] == 4) and (f_start == 7):
                self.castle_rights = self.castle_rights.replace('K', '')
            elif (move[0] == -4) and (f_start == 0):
                self.castle_rights = self.castle_rights.replace('q', '')
            elif (move[0] == -4) and (f_start == 7):
                self.castle_rights = self.castle_rights.replace('k', '')

            if self.castle_rights == '':
                self.castle_rights = '-'


            ###  adding move info to dataframe
            fbp = self.generate_fen_board_position(new_board)

            ###  identifying en passant targets
            ep_target = '-'
            if (abs(move[0]) == 1) and (abs(r_end-r_start) == 2):
                ep_target = '%s%i' % (FILES[f_start], (r_end+r_start)/2)


            #self.df_moves.loc[i_move, 'ep_target'] = ep_target

            #self.df_moves.loc[i_move, 'castle_rights'] = self.castle_rights

            #if (moved_piece == 'pawn') or ('x' in move_text):
            #    self.halfmove_count = 0
            #self.df_moves.loc[i_move, 'halfmove_count'] = self.halfmove_count

            #self.fens.append('%s %s %s %s %i %i' % (fbp, next_player, self.castle_rights, ep_target, self.halfmove_count, (i_move+3)//2))

            #self.halfmove_count += 1

            self.chess_boards.append(numpy.array(new_board))
            self._redraw_board()

        else:
            return new_board


    def _redraw_board(self, board=None):

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        ###  plotting current position
        for r in range(8):
            for f in range(8):
                if board[r][f] in PIECES_WHITE.keys():
                    self.image_board[r][f].set_data(PIECES_WHITE[board[r][f]])
                elif board[r][f] in PIECES_BLACK.keys():
                    self.image_board[r][f].set_data(PIECES_BLACK[board[r][f]])
                else:
                    self.image_board[r][f].set_data(IMAGE_BLANK)
        pyplot.draw()


    def print_chess_board(self, board=None):
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        print('\n' + '-'*(4*8+1))
        for r in range(8, 0, -1):
            s = '| '
            for f in FILES:
                s += '%s | ' % PIECE_ID_2_CHAR[board[r][f]]
            print(s)
            print('-'*(4*8+1))


    def evaluate_depth1_single_proc(self, i_proc, board, list_legal_moves_0):


        for idx0, row0 in list_legal_moves_0.iterrows():

            ###  applying candidate move, getting opponent's followup moves
            next_board_0 = self.move_piece(row0['notation'], row0['player'], board=board, redraw=False)
            evaluation0 = self.get_eval(next_board_0, OTHER_PLAYER[row0['player']])

            list_legal_moves_0.loc[idx0, 'evaluation'] = evaluation0
            list_legal_moves_0.loc[idx0, 'eval_seq'] = '%.2f' % evaluation0
            list_legal_moves_0.loc[idx0, 'move_seq'] = row0['notation']

        ###  saving results
        if len(list_legal_moves_0) > 0:
            list_legal_moves_0.to_csv('list_legal_moves_PROC%02i.csv'%i_proc, index=None)


    def evaluate_depth2_single_proc(self, i_proc, board, list_legal_moves_0):


        for idx0, row0 in list_legal_moves_0.iterrows():

            ###  applying candidate move, getting opponent's followup moves
            next_board_0 = self.move_piece(row0['notation'], row0['player'], board=board, redraw=False)
            evaluation0 = self.get_eval(next_board_0, OTHER_PLAYER[row0['player']])
            list_legal_moves_1 = self.get_legal_moves(OTHER_PLAYER[row0['player']], board=next_board_0, get_eval=False)

            for idx1, row1 in list_legal_moves_1.iterrows():
                ###  applying candidate move, getting opponent's followup moves
                next_board_1 = self.move_piece(row1['notation'], row1['player'], board=next_board_0, redraw=False)
                evaluation1 = self.get_eval(next_board_1, OTHER_PLAYER[row1['player']])
                list_legal_moves_2 = self.get_legal_moves(OTHER_PLAYER[row1['player']], board=next_board_1, get_eval=False)

                for idx2, row2 in list_legal_moves_2.iterrows():
                    ###  applying candidate move, getting opponent's followup moves
                    next_board_2 = self.move_piece(row2['notation'], row2['player'], board=next_board_1, redraw=False)
                    evaluation2 = self.get_eval(next_board_2, OTHER_PLAYER[row2['player']])

                    list_legal_moves_2.loc[idx2, 'evaluation'] = evaluation2
                    list_legal_moves_2.loc[idx2, 'eval_seq'] = '%.2f_%.2f_%.2f' % (evaluation0, evaluation1, evaluation2)
                    list_legal_moves_2.loc[idx2, 'move_seq'] = '%s_%s_%s' % (row0['notation'], row1['notation'], row2['notation'])



                ###  propigating best move of opponent backwards
                if (row1['player'] == 'white') and (len(list_legal_moves_2)>0):
                    best_move = list_legal_moves_2.sort_values(by=['evaluation'], ascending=True).iloc[0]
                    list_legal_moves_1.loc[idx1, 'move_seq'] = best_move['move_seq']
                    list_legal_moves_1.loc[idx1, 'eval_seq'] = best_move['eval_seq']
                    list_legal_moves_1.loc[idx1, 'evaluation'] = best_move['evaluation']
                elif (row1['player'] == 'black') and (len(list_legal_moves_2)>0):
                    best_move = list_legal_moves_2.sort_values(by=['evaluation'], ascending=False).iloc[0]
                    list_legal_moves_1.loc[idx1, 'move_seq'] = best_move['move_seq']
                    list_legal_moves_1.loc[idx1, 'eval_seq'] = best_move['eval_seq']
                    list_legal_moves_1.loc[idx1, 'evaluation'] = best_move['evaluation']
                else:
                    list_legal_moves_1.loc[idx1, 'evaluation'] = evaluation1



            ###  propigating best move of opponent backwards
            if (row0['player'] == 'white') and (len(list_legal_moves_1)>0):
                best_move = list_legal_moves_1.sort_values(by=['evaluation'], ascending=True).iloc[0]
                list_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                list_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                list_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
            elif (row0['player'] == 'black') and (len(list_legal_moves_1)>0):
                best_move = list_legal_moves_1.sort_values(by=['evaluation'], ascending=False).iloc[0]
                list_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                list_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                list_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
            else:
                list_legal_moves_0.loc[idx0, 'evaluation'] = evaluation0

        ###  saving results
        if len(list_legal_moves_0) > 0:
            list_legal_moves_0.to_csv('list_legal_moves_PROC%02i.csv'%i_proc, index=None)


    def evaluate_multiproc(self, player, move_text, board):
        '''
        Evaluate the provided move from player on the provided board
        '''

        ###  applying the provided move, generating next board position
        ###  getting opponent's following legal moves
        next_board_0 = self.move_piece(move_text, player, board=board, redraw=False)
        list_legal_moves_0 = self.get_legal_moves(OTHER_PLAYER[player], board=next_board_0, get_eval=False)

        processes = []
        for i_proc in range(CPU_COUNT):

            ###  don't spawn more processes than there are candidate moves
            if i_proc == len(list_legal_moves_0):
                break

            ###  initializing process on segment of candidatate moves
            ii = range(i_proc, len(list_legal_moves_0), CPU_COUNT)
            list_legal_moves_i = list_legal_moves_0.loc[ii]

            ctx = multiprocessing.get_context('spawn')
            args = (i_proc, next_board_0, list_legal_moves_i)
            proc = multiprocessing.Process(target=self.evaluate_depth2_single_proc, args=args)
            processes.append(proc)
            proc.start()

        ###  joining processes
        for proc in processes:
            proc.join()

        ###  loading files and deleting afterward
        fnames = glob.glob('list_legal_moves_PROC*csv')
        list_legal_moves_w_eval = pandas.concat([pandas.read_csv(f) for f in fnames])
        for f in fnames:
            os.remove(f)
            #shutil.move(f, f.replace('list_legal_moves_', 'list_legal_moves_%s_'%move_text))


        if player == 'white':
            best_move = list_legal_moves_w_eval.sort_values(by=['evaluation'], ascending=True).iloc[0]
            return best_move
        else:
            best_move = list_legal_moves_w_eval.sort_values(by=['evaluation'], ascending=False).iloc[0]
            return best_move


    def get_eval(self, board, player_on_move):

        ###  determining if position is checkmate
        list_legal_moves = self.get_legal_moves(player_on_move, board=board, get_eval=False)
        if (len(list_legal_moves) == 0) and (player_on_move == 1):
            evaluation = -9999

        elif (len(list_legal_moves) == 0) and (player_on_move == -1):
            evaluation = 9999

        else:
            eval_metrics = self.calc_evaluation_metrics(board)
            evaluation = 0
            evaluation += eval_metrics['delta_material']
            #evaluation += eval_metrics['delta_vision'] / 10
            evaluation += eval_metrics['delta_central_control'] / 1.
            #evaluation += eval_metrics['delta_0x_defends'] / 1.
            #evaluation += eval_metrics['delta_overattacks'] / 1.


        return evaluation


    def calc_evaluation_metrics(self, board):
        '''
        delta_material --------- Total white material minus total black material
        delta_vision ----------- Difference in number of squares targeted by players' pieces
        delta_[N]x_attacks ----- Difference in number of squares occupied by opponent pieces targeted `N` times
        delta_[N]x_defends ----- Difference in number of squares occupied by player's own pieces targeted `N` times
        delta_overattacks ------ Difference in number of squares where player's attackers outnumber opponent's defeenders
        delta_central_control -- Difference in cumulative attacks weighted by CENTER_WEIGHT_MAP
        '''


        ###  counting material value
        delta_material = 0
        delta_material += MATERIAL_VALUES[1] * ((board==1).sum() - (board==-1).sum())
        delta_material += MATERIAL_VALUES[2] * ((board==2).sum() - (board==-2).sum())
        delta_material += MATERIAL_VALUES[3] * ((board==3).sum() - (board==-3).sum())
        delta_material += MATERIAL_VALUES[4] * ((board==4).sum() - (board==-4).sum())
        delta_material += MATERIAL_VALUES[5] * ((board==5).sum() - (board==-5).sum())
        delta_material += MATERIAL_VALUES[6] * ((board==6).sum() - (board==-6).sum())


        ###  getting attacks
        n_targets_by_white = self.get_attacked_squares(1, board)
        n_targets_by_black = self.get_attacked_squares(-1, board)
        total_vision_by_white, total_vision_by_black = 0, 0
        n_0x_defends_by_white, n_0x_defends_by_black = 0, 0
        n_overattacks_by_white, n_overattacks_by_black = 0, 0
        central_control_by_white, central_control_by_black = 0, 0
        for r in range(8):
            for f in range(8):
                total_vision_by_white += (n_targets_by_white[r][f] > 0)
                total_vision_by_black += (n_targets_by_black[r][f] > 0)

                central_control_by_white += n_targets_by_white[r][f] * CENTER_WEIGHT_MAP[r][f]
                central_control_by_black += n_targets_by_black[r][f] * CENTER_WEIGHT_MAP[r][f]

                if board[r][f] in PIECES_WHITE.keys():
                    if n_targets_by_white[r][f] == 0: n_0x_defends_by_white += 1
                    if n_targets_by_black[r][f] > n_targets_by_white[r][f]:
                        n_overattacks_by_black += 1

                if board[r][f] in PIECES_BLACK.keys():
                    if n_targets_by_black[r][f] == 0: n_0x_defends_by_black += 1
                    if n_targets_by_white[r][f] > n_targets_by_black[r][f]:
                        n_overattacks_by_white += 1

        eval_metrics = {}
        eval_metrics['delta_material'] = delta_material
        eval_metrics['delta_vision'] = (total_vision_by_white - total_vision_by_black)
        eval_metrics['delta_0x_defends'] = (n_0x_defends_by_black - n_0x_defends_by_white)
        eval_metrics['delta_overattacks'] = (n_overattacks_by_white - n_overattacks_by_black)
        eval_metrics['delta_central_control'] = (central_control_by_white - central_control_by_black)

        return eval_metrics


    def get_pawn_moves(self, f, r, player, int_board):
        ep_target = 0
        moves = playground_cython.get_pawn_moves(f, r, player, int_board, ep_target)
        return moves


    def get_knight_moves(self, f, r, player, int_board):
        moves = playground_cython.get_knight_moves(f, r, player, int_board)
        return moves


    def get_bishop_moves(self, f, r, player, int_board):
        moves = playground_cython.get_bishop_moves(f, r, player, int_board)
        return moves


    def get_rook_moves(self, f, r, player, int_board):
        moves = playground_cython.get_rook_moves(f, r, player, int_board)
        return moves


    def get_queen_moves(self, f, r, player, int_board):
        moves = playground_cython.get_queen_moves(f, r, player, int_board)
        return moves


    def get_king_moves(self, f, r, player, int_board):
        moves = playground_cython.get_king_moves(f, r, player, int_board)
        return moves


    def get_legal_moves(self, player_on_move, int_board, ep_target=0, get_eval=False):
        moves = playground_cython.get_legal_moves(player_on_move, int_board, ep_target, get_eval=0)
        return moves


    def get_attacked_squares(self, player_on_move, int_board):
        return playground_cython.get_attacked_squares(player_on_move, int_board)


    def print_n_attacks(self, board=None):

        n_attacks_by_white = self.get_attacked_squares(1, board=board)
        n_attacks_by_black = self.get_attacked_squares(-1, board=board)

        print('\nNumber of Attacks by White')
        for r, files in n_attacks_by_white.items():
            s = ''
            for f, n in files.items():
                s += '%i ' % n
            print(s)

        print('\nNumber of Attacks by Black')
        for r, files in n_attacks_by_black.items():
            s = ''
            for f, n in files.items():
                s += '%i ' % n
            print(s)
 



if __name__ == '__main__':


    if len(sys.argv) == 1:
        asdf = chessBoard()

    elif len(sys.argv) == 2:
        asdf = chessBoard(sys.argv[1])