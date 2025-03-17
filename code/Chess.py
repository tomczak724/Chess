
import os
import pdb
import sys
import copy
import time
import json
import glob
import numpy
import pandas
import imageio
import matplotlib
import multiprocessing
from matplotlib import pyplot

CPU_COUNT = multiprocessing.cpu_count()

RANKS = [1, 2, 3, 4, 5, 6, 7, 8]
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

DX = 40
IMAGE_PIECES = imageio.imread('../data/chess_pieces.png')

IMAGE_BLANK = IMAGE_PIECES[99:101, 99:101, :]

PIECES_WHITE = {'P':IMAGE_PIECES[162-DX:162+DX, 683-DX:683+DX, :], 
                'N':IMAGE_PIECES[164-DX:164+DX, 557-DX:557+DX, :], 
                'B':IMAGE_PIECES[160-DX:160+DX, 431-DX:431+DX, :], 
                'R':IMAGE_PIECES[162-DX:162+DX, 305-DX:305+DX, :], 
                'Q':IMAGE_PIECES[161-DX:161+DX, 179-DX:179+DX, :], 
                'K':IMAGE_PIECES[161-DX:161+DX, 54-DX:54+DX, :]}

PIECES_BLACK = {'p':IMAGE_PIECES[55-DX:55+DX, 682-DX:682+DX, :], 
                'n':IMAGE_PIECES[55-DX:55+DX, 557-DX:557+DX, :], 
                'b':IMAGE_PIECES[53-DX:53+DX, 430-DX:430+DX, :], 
                'r':IMAGE_PIECES[55-DX:55+DX, 305-DX:305+DX, :], 
                'q':IMAGE_PIECES[54-DX:54+DX, 178-DX:178+DX, :], 
                'k':IMAGE_PIECES[55-DX:55+DX, 53-DX:53+DX, :]}

MATERIAL_VALUES = {'P':1, 'p':1, 'N':3, 'n':3, 'B':3, 'b':3, 'R':5, 'r':5, 'Q':9, 'q':9, 'K':9999, 'k':9999}

CENTER_WEIGHT_MAP = {8:{'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1, 'g':1, 'h':1}, 
                     7:{'a':1, 'b':2, 'c':2, 'd':2, 'e':2, 'f':2, 'g':2, 'h':1}, 
                     6:{'a':1, 'b':2, 'c':3, 'd':3, 'e':3, 'f':3, 'g':2, 'h':1}, 
                     5:{'a':1, 'b':2, 'c':3, 'd':4, 'e':4, 'f':3, 'g':2, 'h':1}, 
                     4:{'a':1, 'b':2, 'c':3, 'd':4, 'e':4, 'f':3, 'g':2, 'h':1}, 
                     3:{'a':1, 'b':2, 'c':3, 'd':3, 'e':3, 'f':3, 'g':2, 'h':1}, 
                     2:{'a':1, 'b':2, 'c':2, 'd':2, 'e':2, 'f':2, 'g':2, 'h':1}, 
                     1:{'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1, 'g':1, 'h':1}}
S = 0
for r, files in CENTER_WEIGHT_MAP.items():
    for f, val in files.items():
        S += val
for r in RANKS:
    for f in FILES:
        CENTER_WEIGHT_MAP[r][f] /= S

OTHER_PLAYER = {'white':'black', 'black':'white'}


class chessBoard(object):

    def __init__(self, pgn_file=None):


        self.pgn_file = pgn_file
        if self.pgn_file is not None:
            print('\nLoading PGN from: %s' % self.pgn_file)
            self.df_pgn = self.load_pgn()
            self.player_on_move = 'white'
            self.current_move = 1

        ###  setting up chess board and pieces
        self.chess_boards = [{8:{'a':'r', 'b':'n', 'c':'b', 'd':'q', 'e':'k', 'f':'b', 'g':'n', 'h':'r'}, 
                              7:{'a':'p', 'b':'p', 'c':'p', 'd':'p', 'e':'p', 'f':'p', 'g':'p', 'h':'p'}, 
                              6:{'a':' ', 'b':' ', 'c':' ', 'd':' ', 'e':' ', 'f':' ', 'g':' ', 'h':' '}, 
                              5:{'a':' ', 'b':' ', 'c':' ', 'd':' ', 'e':' ', 'f':' ', 'g':' ', 'h':' '}, 
                              4:{'a':' ', 'b':' ', 'c':' ', 'd':' ', 'e':' ', 'f':' ', 'g':' ', 'h':' '}, 
                              3:{'a':' ', 'b':' ', 'c':' ', 'd':' ', 'e':' ', 'f':' ', 'g':' ', 'h':' '}, 
                              2:{'a':'P', 'b':'P', 'c':'P', 'd':'P', 'e':'P', 'f':'P', 'g':'P', 'h':'P'}, 
                              1:{'a':'R', 'b':'N', 'c':'B', 'd':'Q', 'e':'K', 'f':'B', 'g':'N', 'h':'R'}}]

        self.list_eval_metrics = [self.calc_evaluation_metrics(self.chess_boards[-1])]

        ###  setting up lists of legal moves
        self.df_moves = pandas.DataFrame(columns=['player', 'notation', 'piece', 'start_square', 'end_square', 'capture_flag', 'check_flag', 'fen_board_position', 'ep_target', 'castle_rights', 'halfmove_count'])
        self.halfmove_count = 0
        self.fens = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1']
        self.promotion_prompt = False

        self.player_on_move = 'white'
        self.previous_move = {'player':'', 'move':''}
        self.castle_rights = 'KQkq'
        self.df_possible_moves = pandas.DataFrame()


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
                ax = self.fig.add_axes([x0, y0, 0.93/8, 0.93/8], label='%s%i'%(FILES[i_file], RANKS[i_rank]))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.axes.set_facecolor('none')

                imdat = ax.imshow(IMAGE_BLANK, origin='upper')

                rank.update({FILES[i_file]:imdat})

            self.image_board.update({i_rank+1:rank})

        ###  plotting images of pieces
        for r in RANKS:
            for f in FILES:
                if self.chess_boards[-1][r][f] in PIECES_WHITE.keys():
                    self.image_board[r][f].set_data(PIECES_WHITE[self.chess_boards[-1][r][f]])
                if self.chess_boards[-1][r][f] in PIECES_BLACK.keys():
                    self.image_board[r][f].set_data(PIECES_BLACK[self.chess_boards[-1][r][f]])

        ###  iterating over PGN if provided
        if (pgn_file is not None) and True:

            ###  applying all moves in PGN, one-by-one
            for idx, row in self.df_pgn.iterrows():

                if pandas.isnull(row['white']):
                    break
                else:
                    self.next(1)
                    self.list_eval_metrics.append(self.calc_evaluation_metrics(self.chess_boards[-1]))
                    if ('#' in row['white']) or ('++' in row['white']):
                        break

                if pandas.isnull(row['black']):
                    break
                else:
                    self.next(1)
                    self.list_eval_metrics.append(self.calc_evaluation_metrics(self.chess_boards[-1]))
                    if ('#' in row['black']) or ('++' in row['black']):
                        break

            fnames_existing_evals = glob.glob('../data/evaluations/game_*csv')
            fout = '../data/evaluations/game_%04i.csv' % len(fnames_existing_evals)
            pandas.DataFrame(self.list_eval_metrics).to_csv(fout, index=None)

        ###  instantiating interactive variables
        self.selected_square = None





        ###  playing game against self
        if False:

            timestamps = [time.time()]
            while True:

                df_legal_moves = self.get_legal_moves(self.player_on_move, get_eval=True)

                print(time.ctime())
                if self.player_on_move == 'white':
                    print('\nTop three moves for white:')
                    df_legal_moves_sorted = df_legal_moves.sort_values(by=['evaluation'],ascending=False)
                    print(df_legal_moves_sorted.head(3))
                else:
                    print('\nTop three moves for black:')
                    df_legal_moves_sorted = df_legal_moves.sort_values(by=['evaluation'],ascending=True)
                    print(df_legal_moves_sorted.head(3))

                ###  applying top move
                self.move_piece(df_legal_moves_sorted.iloc[0]['notation'], self.player_on_move, board=None, redraw=True)
                self.player_on_move = OTHER_PLAYER[self.player_on_move]

                print('\nPGN')
                self.print_png_body()
                print('')
                timestamps.append(time.time())
                print('time to calc move  = %.1f minutes' % ((timestamps[-1]-timestamps[-2])/60.))
                print('total time elapsed = %.1f minutes' % ((timestamps[-1]-timestamps[0])/60.))


        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)


    def _onClick(self, event):

        if (event.inaxes is not None):

            ###  unhighlight all squares
            for r in RANKS:
                for f in FILES:
                    self.image_board[r][f].axes.set_facecolor('none')


            ###  if clicked square is the currently-selected square then truncate
            f = event.inaxes.get_label()[0]
            r = int(event.inaxes.get_label()[1])
            clicked_square = '%s%i' % (f, r)
            if clicked_square == self.selected_square:
                self.selected_square = None
                self.df_possible_moves = pandas.DataFrame()
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


            ###  moving piece if clicked square is in list of possible moves
            elif (len(self.df_possible_moves) > 0) and (clicked_square in self.df_possible_moves['end_square'].tolist()):

                ###  if pawn promotion then prompt for which piece
                ###  handling promotions for white
                f0, r0 = self.selected_square
                if (self.chess_boards[-1][int(r0)][f0] == 'P') and (r == 8):
                    self.promotion_prompt = True
                    self.promotion_options = {}
                    self.toggle_transparency('on')
                    self.image_board[7][f0].set_data(IMAGE_BLANK)
                    for ri, pi in zip([8, 7, 6, 5], ['Q', 'N', 'R', 'B']):
                        self.image_board[ri][f].axes.set_facecolor('w')
                        self.promotion_options.update({'%s%i'%(f, ri): pi})
                        self.image_board[ri][f].set_data(PIECES_WHITE[pi])
                        self.image_board[ri][f].set_alpha(1)
                    pyplot.draw()

                ###  handling promotions for black
                elif (self.chess_boards[-1][int(r0)][f0] == 'p') and (r == 1):
                    self.promotion_prompt = True
                    self.promotion_options = {}
                    self.toggle_transparency('on')
                    self.image_board[2][f0].set_data(IMAGE_BLANK)
                    for ri, pi in zip([1, 2, 3, 4], ['q', 'n', 'r', 'b']):
                        self.image_board[ri][f].axes.set_facecolor('w')
                        self.promotion_options.update({'%s%i'%(f, ri): pi})
                        self.image_board[ri][f].set_data(PIECES_BLACK[pi])
                        self.image_board[ri][f].set_alpha(1)
                    pyplot.draw()

                ###  moving piece to clicked square
                else:

                    ###  saving evaluation metrics
                    self.list_eval_metrics.append(self.calc_evaluation_metrics(self.chess_boards[-1]))


                    df_candidate_moves = self.df_possible_moves.query('end_square=="%s"'%clicked_square)
                    move_text = df_candidate_moves.iloc[0]['notation']
                    self.move_piece(move_text, self.player_on_move, board=None, redraw=True)
                    self._redraw_board()

                    self.selected_square = None
                    self.df_possible_moves = pandas.DataFrame()
                    self.player_on_move = OTHER_PLAYER[self.player_on_move]


                    ###  printing top move to play next
                    if self.player_on_move == 'black' and True:
                        t0 = time.time()
                        df_legal_moves = self.get_legal_moves(self.player_on_move, get_eval=True)
                        tf = time.time()
                        #df_legal_moves.to_csv('DF_LEGAL_MOVES_%03i.csv' % self.df_moves.index[-1], index=False)
                        print('')
                        print('time elapsed = %.1f minutes' % ((tf-t0)/60.))
                        if self.player_on_move == 'white':
                            df_legal_moves_sorted = df_legal_moves.sort_values(by=['evaluation'],ascending=False)
                            print(df_legal_moves_sorted.head(3))
                        else:
                            df_legal_moves_sorted = df_legal_moves.sort_values(by=['evaluation'],ascending=True)
                            print(df_legal_moves_sorted.head(3))

                        ###  applying top move
                        self.move_piece(df_legal_moves_sorted.iloc[0]['notation'], self.player_on_move, board=None, redraw=True)
                        self.player_on_move = OTHER_PLAYER[self.player_on_move]



            ###  if selecting a piece, get legal moves
            else:

                df_legal_moves = self.get_legal_moves(self.player_on_move, get_eval=False)
                self.selected_square = event.inaxes.get_label()
                self.df_possible_moves = df_legal_moves.query('start_square=="%s"'%self.selected_square)

                ###  highlighting squares for player on move
                occupant = self.chess_boards[-1][r][f]
                if (self.player_on_move == 'white' and occupant in PIECES_WHITE.keys()) or ((self.player_on_move == 'black' and occupant in PIECES_BLACK.keys())):
                    for idx, row in self.df_possible_moves.iterrows():
                        f, r = row['end_square'][0], int(row['end_square'][1])
                        self.image_board[r][f].axes.set_facecolor('#ff9999')

                self._redraw_board()


    def get_legal_moves(self, player_on_move, board=None, get_eval=False):
        '''
        Description
        -----------
            Returns the list of all legal moves for the given player-on-move in the provided position.
            If no board is given then the current board position is used.
        '''

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        list_moves = []
        if player_on_move == 'white':
            for r in RANKS:
                for f in FILES:

                    if board[r][f] == 'P':
                        list_moves += self.get_pawn_moves(f, r, 'white', board)
                    if board[r][f] == 'N':
                        list_moves += self.get_knight_moves(f, r, 'white', board)
                    if board[r][f] == 'B':
                        list_moves += self.get_bishop_moves(f, r, 'white', board)
                    if board[r][f] == 'R':
                        list_moves += self.get_rook_moves(f, r, 'white', board)
                    if board[r][f] == 'Q':
                        list_moves += self.get_queen_moves(f, r, 'white', board)
                    if board[r][f] == 'K':
                        list_moves += self.get_king_moves(f, r, 'white', board)

        elif player_on_move == 'black':
            for r in RANKS:
                for f in FILES:
                    if board[r][f] == 'p':
                        list_moves += self.get_pawn_moves(f, r, 'black', board)
                    if board[r][f] == 'n':
                        list_moves += self.get_knight_moves(f, r, 'black', board)
                    if board[r][f] == 'b':
                        list_moves += self.get_bishop_moves(f, r, 'black', board)
                    if board[r][f] == 'r':
                        list_moves += self.get_rook_moves(f, r, 'black', board)
                    if board[r][f] == 'q':
                        list_moves += self.get_queen_moves(f, r, 'black', board)
                    if board[r][f] == 'k':
                        list_moves += self.get_king_moves(f, r, 'black', board)

        df_candidate_moves = pandas.DataFrame(list_moves).query('player=="%s"'%player_on_move)
        df_legal_moves_0 = self.filter_illegal_moves(df_candidate_moves, board)



        ###  computing evaluation of moves
        ###  version 2.0 : evaluating moves in pairs
        if True and (get_eval is True):

            for idx0, row0 in df_legal_moves_0.iterrows():
                ###  applying candidate move, getting opponent's followup moves
                next_board_0 = self.move_piece(row0['notation'], row0['player'], board=board, redraw=False)
                evaluation0 = self.get_eval(next_board_0, OTHER_PLAYER[row0['player']])
                df_legal_moves_1 = self.get_legal_moves(OTHER_PLAYER[player_on_move], board=next_board_0, get_eval=False)

                for idx1, row1 in df_legal_moves_1.iterrows():
                    ###  applying candidate move, getting opponent's followup moves
                    next_board_1 = self.move_piece(row1['notation'], row1['player'], board=next_board_0, redraw=False)
                    evaluation1 = self.get_eval(next_board_1, OTHER_PLAYER[row1['player']])

                    df_legal_moves_1.loc[idx1, 'evaluation'] = evaluation1
                    df_legal_moves_1.loc[idx1, 'eval_seq'] = '%.2f_%.2f' % (evaluation0, evaluation1)
                    df_legal_moves_1.loc[idx1, 'move_seq'] = '%s_%s' % (row0['notation'], row1['notation'])

                ###  propigating best move of opponent backwards
                if (row0['player'] == 'white') and (len(df_legal_moves_1)>0):
                    best_move = df_legal_moves_1.sort_values(by=['evaluation'], ascending=True).iloc[0]
                    df_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                    df_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                    df_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
                elif (row0['player'] == 'black') and (len(df_legal_moves_1)>0):
                    best_move = df_legal_moves_1.sort_values(by=['evaluation'], ascending=False).iloc[0]
                    df_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                    df_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                    df_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
                else:
                    df_legal_moves_0.loc[idx0, 'evaluation'] = evaluation0


        if False and (get_eval is True):
            for idx, row in df_legal_moves_0.iterrows():
                best_move = self.evaluate_multiproc(row['player'], row['notation'], board)
                df_legal_moves_0.loc[idx, 'move_seq'] = best_move['move_seq']
                df_legal_moves_0.loc[idx, 'eval_seq'] = best_move['eval_seq']
                df_legal_moves_0.loc[idx, 'evaluation'] = best_move['evaluation']


        return df_legal_moves_0


    def toggle_transparency(self, on_off):

        if on_off == 'on':
            self.ax.set_facecolor('gray')
            for r in RANKS:
                for f in FILES:
                    self.image_board[r][f].set_alpha(0.5)

        elif on_off == 'off':
            self.ax.set_facecolor('none')
            for r in RANKS:
                for f in FILES:
                    self.image_board[r][f].set_alpha(1)


    def load_pgn(self):

        ###  reading and parsing PGN text
        with open(self.pgn_file, 'r') as fopen:

            ###  reading all lines of text file
            lines = fopen.readlines()

            ###  parsing metadata
            metadata = {}
            for line in lines:
                if line[0] == '[':
                    line_cleaned = line.strip('[]\n')
                    key = line_cleaned.split(' ')[0]
                    value = eval(line_cleaned.replace('%s '%key, ''))
                    metadata[key] = value

            ###  sluggifying moves text
            moves_text = ''.join(lines[len(metadata):])
            moves_text = moves_text.replace('\n', ' ').replace('%', '')
            moves_text = moves_text.strip(' ')
            while '  ' in moves_text:
                moves_text = moves_text.replace('  ', ' ')

            ###  parsing moves data from moves text
            list_moves = []
            move_data = None
            move_counter = 1
            player = 'black'
            for block in moves_text.split(' '):

                if block == '%i.' % move_counter:
                    if move_data is not None:
                        list_moves.append(move_data)

                    move_data = {'turn': move_counter}
                    move_counter += 1

                elif '{[' in block:
                    k = block.replace('{[', '')

                elif ']}' in block:
                    v = block.replace(']}', '')
                    move_data['%s_%s' % (k, player)] = v
                    k, v = None, None

                elif block not in ['1-0', '0-1', '1/2-1/2', '0.5-0.5']:
                    player = OTHER_PLAYER[player]
                    move_data[player] = block

            list_moves.append(move_data)

            return pandas.DataFrame(list_moves)


    def load_fen(self, fen):

        #r3k2r/3n1p1p/p3p1p1/1b1pP3/2nP1P2/2qB1Q1P/2P1N1PK/R1B2R2 b kq - 4 22
        ###
        ###  Mate in two:
        ###  r3r1k1/p1p2ppp/1p4q1/8/1PP5/P7/4QPPP/R3R1K1 w - - 0 1
        ###
        ###  Mate in one:
        ###  5r1k/6pp/8/8/8/8/6PP/5R1K w - - 0 1

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
                        rank[FILES[i_file]] = ' '
                        i_file += 1

                else:
                    rank[FILES[i_file]] = char
                    i_file += 1

            board[8-i_rank] = rank


        self.chess_boards = [board]
        self.player_on_move = {'w':'white', 'b':'black'}[fen_player_on_move]
        self.castle_rights = fen_castle_rights
        self.halfmove_count = int(fen_halfmove_count)

        i = len(self.df_moves)
        self.df_moves.loc[i, 'ep_target'] = fen_ep_target

        self._redraw_board()


    def get_fen(self, position=0):

        #r3k2r/3n1p1p/p3p1p1/1b1pP3/2nP1P2/2qB1Q1P/2P1N1PK/R1B2R2 b kq - 4 22

        board = self.chess_boards[position]

        ###  adding peice locations to FEN string
        fen = ''
        n_vacant_files = 0
        for r in RANKS[::-1]:

            n_vacant = 0
            for f in FILES:

                ###  if square is empty, increment vacant count
                if board[r][f] == ' ':
                    n_vacant += 1

                else:
                    ###  if n_vacant is not 0, add to FEN string
                    if n_vacant > 0:
                        fen += '%i' % n_vacant
                        n_vacant = 0

                    ###  add piece to FEN string
                    fen += board[r][f]

            if n_vacant > 0:
                fen += '%i' % n_vacant

            if r > 1:
                fen += '/'

        ###  adding next player to FEN string
        fen += ' %s ' % self.player_on_move[0]

        ###  adding castling rights to FEN string
        fen += ' %s ' % self.castle_rights

        ###  adding available en passant target
        if position > 0:
            board_prev = self.chess_boards[position-1]

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
        for r in RANKS[::-1]:

            n_vacant = 0
            for f in FILES:

                ###  if square is empty, increment vacant count
                if board[r][f] == ' ':
                    n_vacant += 1

                else:
                    ###  if n_vacant is not 0, add to FEN string
                    if n_vacant > 0:
                        fen_board_position += '%i' % n_vacant
                        n_vacant = 0

                    ###  add piece to FEN string
                    fen_board_position += board[r][f]

            if n_vacant > 0:
                fen_board_position += '%i' % n_vacant

            if r > 1:
                fen_board_position += '/'

        return fen_board_position


    def next(self, n=1, verbose=False):

        for i in range(n):

            if verbose is True:
                print(self.df_pgn.query('turn==%s'%self.current_move)[self.player_on_move].iloc[0])
            self.move_piece(self.df_pgn.query('turn==%s'%self.current_move)[self.player_on_move].iloc[0], 
                            self.player_on_move, 
                            redraw=True)

            if self.player_on_move == 'white':
                self.previous_move['player'] = 'white'
                self.previous_move['move'] = self.df_pgn.query('turn==%s'%self.current_move)[self.player_on_move].iloc[0]
                self.player_on_move = 'black'
            else:
                self.player_on_move = 'white'
                self.previous_move['move'] = self.df_pgn.query('turn==%s'%self.current_move)[self.player_on_move].iloc[0]
                self.previous_move['player'] = 'black'
                self.current_move += 1


    def undo(self, n=1):

        for i in range(n):

            ###  break if returned to the beginning
            if len(self.chess_boards) == 1:
                break

            old_board = self.chess_boards.pop(len(self.chess_boards)-1)

        self._redraw_board()


    def move_piece(self, move_text, player, board=None, redraw=False):
        '''
        Description
        -----------
            Applies the given move by the given player to the given board layout.
            If no board layout is provided, uses the current board position.

        Parameters
        ----------
            move_text : str
                String of single move (e.g. Ra5)

            player : str
                String of which player made the move ('white', 'black')

            board : dict
                Layout of board

            redraw : bool
                If True redraws the board layout in the plot window.
                If False returns the updated chess board.
        Returns
            new_board : dict
                The next chess board object with the given move applied

        '''

        ###  if no board layout is provided use current position
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])


        ###  checking for pawn moves
        if move_text[0] in FILES:
            moved_piece = 'pawn'
            new_board, start_square, end_square, en_passants = self._move_pawn(move_text, player, board)

        ###  checking for rook moves
        elif move_text[0] == 'R':
            moved_piece = 'rook'
            new_board, start_square, end_square = self._move_rook(move_text, player, board)

        ###  checking for knight moves
        elif move_text[0] == 'N':
            moved_piece = 'knight'
            new_board, start_square, end_square = self._move_knight(move_text, player, board)

        ###  checking for bishop moves
        elif move_text[0] == 'B':
            moved_piece = 'bishop'
            new_board, start_square, end_square = self._move_bishop(move_text, player, board)

        ###  checking for queen moves
        elif move_text[0] == 'Q':
            moved_piece = 'queen'
            new_board, start_square, end_square = self._move_queen(move_text, player, board)

        ###  checking for King moves
        elif move_text[0] == 'K':
            moved_piece = 'king'
            new_board, start_square, end_square = self._move_king(move_text, player, board)

        ###  checking for castles
        elif 'O-O' in move_text:
            moved_piece = 'king'
            new_board, start_square, end_square = self._move_castle(move_text, player, board)


        ###  redraw plot window and appending new board to history
        if redraw == True:

            ###  updating castling rights
            if (moved_piece == 'king') and (player == 'white'):
                self.castle_rights = self.castle_rights.replace('K', '')
                self.castle_rights = self.castle_rights.replace('Q', '')
            elif (moved_piece == 'king') and (player == 'black'):
                self.castle_rights = self.castle_rights.replace('k', '')
                self.castle_rights = self.castle_rights.replace('q', '')
            elif (moved_piece == 'rook') and (start_square == 'a1'):
                self.castle_rights = self.castle_rights.replace('Q', '')
            elif (moved_piece == 'rook') and (start_square == 'h1'):
                self.castle_rights = self.castle_rights.replace('K', '')
            elif (moved_piece == 'rook') and (start_square == 'a8'):
                self.castle_rights = self.castle_rights.replace('q', '')
            elif (moved_piece == 'rook') and (start_square == 'h8'):
                self.castle_rights = self.castle_rights.replace('k', '')
            if self.castle_rights == '':
                self.castle_rights = '-'


            ###  adding move info to dataframe
            i_move = len(self.df_moves)
            self.df_moves.loc[i_move, 'player'] = player
            self.df_moves.loc[i_move, 'notation'] = move_text
            self.df_moves.loc[i_move, 'piece'] = moved_piece
            self.df_moves.loc[i_move, 'start_square'] = start_square
            self.df_moves.loc[i_move, 'end_square'] = end_square
            self.df_moves.loc[i_move, 'capture_flag'] = ('x' in move_text)
            self.df_moves.loc[i_move, 'check_flag'] = ('+' in move_text) | ('#' in move_text)
            fbp = self.generate_fen_board_position(new_board)
            self.df_moves.loc[i_move, 'fen_board_position'] = fbp

            ep_target = '-'
            if (moved_piece == 'pawn') and (abs(int(start_square[1])-int(end_square[1])) == 2):
                ep_target = '%s%i' % (start_square[0], (int(start_square[1])+int(end_square[1]))/2)
            self.df_moves.loc[i_move, 'ep_target'] = ep_target

            self.df_moves.loc[i_move, 'castle_rights'] = self.castle_rights

            if (moved_piece == 'pawn') or ('x' in move_text):
                self.halfmove_count = 0
            self.df_moves.loc[i_move, 'halfmove_count'] = self.halfmove_count
            #self.df_moves.loc[i_move, 'eval'] = self.evaluate(self.player_on_move, board=new_board)

            if player == 'white':
                next_player = 'b'
            else:
                next_player = 'w'

            self.fens.append('%s %s %s %s %i %i' % (fbp, next_player, self.castle_rights, ep_target, self.halfmove_count, (i_move+3)//2))

            self.halfmove_count += 1

            self.chess_boards.append(new_board)
            self._redraw_board()

        else:
            return new_board


    def _move_pawn(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  identifying end square
        if 'x' in m:
            end_square = m[2:4]
        else:
            end_square = m[0:2]


        ###  list to hold possible en passant captures if made available
        en_passants = []

        ###  simple pawn advance
        if len(m.strip('+').strip('#')) == 2:
            f, r = m[0], int(m[1])

            if p == 'white':
                board_copy[r][f] = 'P'

                ###  checking if first move is two squares
                if board_copy[r-1][f] == ' ':
                    board_copy[r-2][f] = ' '
                    start_square = '%s%i' % (f, r-2)

                    ###  checking if en passant is available
                    if (r == 4) and (f != 'h') and (board_copy[4][FILES[FILES.index(f)+1]] == 'p'):
                        en_passants.append('%sx%s3' % (FILES[FILES.index(f)+1], f))

                    if (r == 4) and (f != 'a') and (board_copy[4][FILES[FILES.index(f)-1]] == 'p'):
                        en_passants.append('%sx%s3' % (FILES[FILES.index(f)-1], f))

                ###  if not then it was a one-square advance
                else:
                    board_copy[r-1][f] = ' '
                    start_square = '%s%i' % (f, r-1)

            if p == 'black':
                board_copy[r][f] = 'p'

                ###  checking if first move is two squares
                if board_copy[r+1][f] == ' ':
                    board_copy[r+2][f] = ' '
                    start_square = '%s%i' % (f, r+2)

                    ###  checking if en passant is available
                    if (r == 5) and (f != 'h') and (board_copy[5][FILES[FILES.index(f)+1]] == 'P'):
                        en_passants.append('%sx%s6' % (FILES[FILES.index(f)+1], f))

                    if (r == 5) and (f != 'a') and (board_copy[5][FILES[FILES.index(f)-1]] == 'P'):
                        en_passants.append('%sx%s6' % (FILES[FILES.index(f)-1], f))

                ###  if not then it was a one-square advance
                else:
                    board_copy[r+1][f] = ' '
                    start_square = '%s%i' % (f, r+1)

        ###  pawn captures (not including promotion)
        if (len(m.strip('+').strip('#')) == 4) and (m[1] == 'x'):
            f1, f2, r = m[0], m[2], int(m[3])

            if p == 'white':
                start_square = '%s%i' % (f1, r-1)

                ###  checking for en passant capture
                if (r == 6) and (board_copy[r][f2] == ' ') and (board_copy[r-1][f2] == 'p'):
                    board_copy[r-1][f2] = ' '

                board_copy[r][f2] = 'P'
                board_copy[r-1][f1] = ' '

            if p == 'black':
                start_square = '%s%i' % (f1, r+1)

                ###  checking for en passant capture
                if (r == 3) and (board_copy[r][f2] == ' ') and (board_copy[r+1][f2] == 'P'):
                    board_copy[r+1][f2] = ' '

                board_copy[r][f2] = 'p'
                board_copy[r+1][f1] = ' '

        ###  pawn promotions
        if '=' in m:

            ###  captures with promotion
            if m[1] == 'x':
                f1, f2, r, new_piece = m[0], m[2], int(m[3]), m.split('=')[1][0]

                if p == 'white':
                    start_square = '%s%i' % (f1, r-1)
                    board_copy[r][f2] = new_piece
                    board_copy[r-1][f1] = ' '

                if p == 'black':
                    start_square = '%s%i' % (f1, r+1)
                    board_copy[r][f2] = new_piece.lower()
                    board_copy[r+1][f1] = ' '

            ###  simple promotion
            else:
                f, r, new_piece = m[0], int(m[1]), m.split('=')[1][0]

                if p == 'white':
                    start_square = '%s%i' % (f, r-1)
                    board_copy[r][f] = new_piece
                    board_copy[r-1][f] = ' '

                if p == 'black':
                    start_square = '%s%i' % (f, r+1)
                    board_copy[r][f] = new_piece.lower()
                    board_copy[r+1][f] = ' '

        return (board_copy, start_square, end_square, en_passants)


    def _move_rook(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'R'
        else:
            piece = 'r'


        ###  resolving fully-disambiguated rook moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board_copy[r1][f1] = ' '
            board_copy[r2][f2] = piece

            ###  identifying start and end squares
            start_square = m[1:3]
            end_square = m[3:5]

        ###  resolving partially-disambiguated rook moves
        elif len(m) == 4:
            rf1, f2, r2 = m[1], m[2], int(m[3])

            ###  identifying starting square
            if rf1 in FILES:
                board_copy[r2][rf1] = ' '
                start_square = '%s%i' % (rf1, r2)
            else:
                board_copy[int(rf1)][f2] = ' '
                start_square = '%s%i' % (f2, int(rf1))

            end_square = '%s%i' % (f2, r2)
            board_copy[r2][f2] = piece

        ###  resolving regular rook moves
        else:
            f2, r2 = m[1], int(m[2])
            end_square = m[1:3]

            ###  search for current location of rook, vacating square
            ###  cycling through lower RANKS
            for dr in range(1, 8, 1):
                if (r2-dr in RANKS) and (board_copy[r2-dr][f2] == piece):
                    board_copy[r2-dr][f2] = ' '
                    start_square = '%s%i' % (f2, r2-dr)
                    break
                elif (r2-dr not in RANKS) or (board_copy[r2-dr][f2] != ' '):
                    break

            ###  cycling through higher RANKS
            for dr in range(1, 8, 1):
                if (r2+dr in RANKS) and (board_copy[r2+dr][f2] == piece):
                    board_copy[r2+dr][f2] = ' '
                    start_square = '%s%i' % (f2, r2+dr)
                    break
                elif (r2+dr not in RANKS) or (board_copy[r2+dr][f2] != ' '):
                    break

            ###  cycling through lower FILES
            for df in range(1, 8, 1):
                if (FILES.index(f2)-df in range(8)) and (board_copy[r2][FILES[FILES.index(f2)-df]] == piece):
                    board_copy[r2][FILES[FILES.index(f2)-df]] = ' '
                    start_square = '%s%i' % (FILES[FILES.index(f2)-df], r2)
                    break
                elif (FILES.index(f2)-df not in range(8)) or (board_copy[r2][FILES[FILES.index(f2)-df]] != ' '):
                    break

            ###  cycling through higher FILES
            for df in range(1, 8, 1):
                if (FILES.index(f2)+df in range(8)) and (board_copy[r2][FILES[FILES.index(f2)+df]] == piece):
                    board_copy[r2][FILES[FILES.index(f2)+df]] = ' '
                    start_square = '%s%i' % (FILES[FILES.index(f2)+df], r2)
                    break
                elif (FILES.index(f2)+df not in range(8)) or (board_copy[r2][FILES[FILES.index(f2)+df]] != ' '):
                    break

            board_copy[r2][f2] = piece

        return (board_copy, start_square, end_square)


    def _move_knight(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'N'
        else:
            piece = 'n'


        ###  resolving fully-disambiguated knight moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board_copy[r1][f1] = ' '
            board_copy[r2][f2] = piece

            ###  identifying start and end squares
            start_square = m[1:3]
            end_square = m[3:5]


        ###  resolving partailly-disambiguated knight moves
        elif len(m) == 4:
            rf1, f2, r2 = m[1], m[2], int(m[3])
            end_square = m[2:4]

            ###  search for current location of knight, vacating square
            if rf1 in FILES:
                ###  cycling through RANKS
                for r1 in RANKS:
                    if board_copy[r1][rf1] == piece:
                        board_copy[r1][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board_copy[int(rf1)][f1] == piece:
                        board_copy[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board_copy[r2][f2] = piece


        ###  resolving regular knight moves
        else:
            f2, r2 = m[1], int(m[2])
            end_square = m[1:3]

            ###  identifying candidate starting squares
            candidate_starts = []
            if (3 <= r2) and (1 <= FILES.index(f2)):
                candidate_starts.append([FILES[FILES.index(f2)-1], r2-2])
            if (2 <= r2) and (2 <= FILES.index(f2)):
                candidate_starts.append([FILES[FILES.index(f2)-2], r2-1])
            if (r2 <= 7) and (2 <= FILES.index(f2)):
                candidate_starts.append([FILES[FILES.index(f2)-2], r2+1])
            if (r2 <= 6) and (1 <= FILES.index(f2)):
                candidate_starts.append([FILES[FILES.index(f2)-1], r2+2])
            if (r2 <= 6) and (FILES.index(f2) <= 6):
                candidate_starts.append([FILES[FILES.index(f2)+1], r2+2])
            if (r2 <= 7) and (FILES.index(f2) <= 5):
                candidate_starts.append([FILES[FILES.index(f2)+2], r2+1])
            if (2 <= r2) and (FILES.index(f2) <= 5):
                candidate_starts.append([FILES[FILES.index(f2)+2], r2-1])
            if (3 <= r2) and (FILES.index(f2) <= 6):
                candidate_starts.append([FILES[FILES.index(f2)+1], r2-2])

            ###  checking candidate starting squares, vacating knight
            for f1, r1 in candidate_starts:
                if board_copy[r1][f1] == piece:
                    board_copy[r1][f1] = ' '
                    start_square = '%s%i' % (f1, r1)
                    break

            board_copy[r2][f2] = piece

        return (board_copy, start_square, end_square)


    def _move_bishop(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'B'
        else:
            piece = 'b'


        ###  resolving fully-disambiguated bishop moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board_copy[r1][f1] = ' '
            board_copy[r2][f2] = piece

            ###  identifying start and end squares
            start_square = m[1:3]
            end_square = m[3:5]

        ###  resolving partially-disambiguated bishop moves
        elif len(m) == 4:
            rf1, f2, r2 = m[1], m[2], int(m[3])
            end_square = m[2:4]

            ###  searching through candidates, vacating square
            if rf1 in FILES:
                ###  cycling through RANKS
                for r1 in RANKS:
                    if board_copy[r1][rf1] == piece:
                        board_copy[r2][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board_copy[int(rf1)][f1] == piece:
                        board_copy[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board_copy[r2][f2] = piece


        ###  resolving regular bishop moves
        else:
            f2, r2 = m[1], int(m[2])
            end_square = m[1:3]

            ###  identifying starting square
            candidate_starts = []
            for i, r1 in enumerate(range(r2+1, 9, 1)):
                if (FILES.index(f2)-i-1 >= 0): 
                    candidate_starts.append([FILES[FILES.index(f2)-i-1], r1]) 
                if (FILES.index(f2)+i+1 < len(FILES)): 
                    candidate_starts.append([FILES[FILES.index(f2)+i+1], r1])

            for i, r1 in enumerate(range(r2-1, 0, -1)):
                if (FILES.index(f2)-i-1 >= 0): 
                    candidate_starts.append([FILES[FILES.index(f2)-i-1], r1]) 
                if (FILES.index(f2)+i+1 < len(FILES)): 
                    candidate_starts.append([FILES[FILES.index(f2)+i+1], r1])

            ###  searching through candidates, vacating square
            for f1, r1 in candidate_starts:
                if board_copy[r1][f1] == piece:
                    board_copy[r1][f1] = ' '
                    start_square = '%s%i' % (f1, r1)

            board_copy[r2][f2] = piece

        return (board_copy, start_square, end_square)


    def _move_queen(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'Q'
        else:
            piece = 'q'


        ###  resolving fully-disambiguated queen moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board_copy[r1][f1] = ' '
            board_copy[r2][f2] = piece

            ###  identifying start and end squares
            start_square = m[1:3]
            end_square = m[3:5]

        ###  resolving partially-disambiguated queen moves
        elif len(m) == 4:
            rf1, f2, r2 = m[1], m[2], int(m[3])
            end_square = m[2:4]

            ###  searching through candidates, vacating square
            if rf1 in FILES:
                ###  cycling through RANKS
                for r1 in RANKS:
                    if board_copy[r1][rf1] == piece:
                        board_copy[r2][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board_copy[int(rf1)][f1] == piece:
                        board_copy[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board_copy[r2][f2] = piece

        ###  resolving regular queen moves
        else:
            f2, r2 = m[1], int(m[2])
            end_square = m[1:3]


            ###  identifying starting square via ray casting
            found_queen = False

            ###  up-right diagonal
            for d in range(1, 8, 1):

                ###  checking if square is valid
                if (r2+d in RANKS) and (FILES.index(f2)+d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2+d][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2+d, FILES[FILES.index(f2)+d]
                        break
                    elif board_copy[r2+d][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  right files
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (FILES.index(f2)+d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2, FILES[FILES.index(f2)+d]
                        break
                    elif board_copy[r2][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  down-right diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS) and (FILES.index(f2)+d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2-d][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2-d, FILES[FILES.index(f2)+d]
                        break
                    elif board_copy[r2-d][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  down ranks
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2-d][f2] == piece:
                        found_queen = True
                        r1, f1 = r2-d, f2
                        break
                    elif board_copy[r2-d][f2] != ' ':
                        break


            ###  down-left diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS) and (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2-d][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2-d, FILES[FILES.index(f2)-d]
                        break
                    elif board_copy[r2-d][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  left files
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2, FILES[FILES.index(f2)-d]
                        break
                    elif board_copy[r2][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  up-left diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2+d in RANKS) and (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2+d][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2+d, FILES[FILES.index(f2)-d]
                        break
                    elif board_copy[r2+d][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  up ranks
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2+d in RANKS):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board_copy[r2+d][f2] == piece:
                        found_queen = True
                        r1, f1 = r2+d, f2
                        break
                    elif board_copy[r2+d][f2] != ' ':
                        break


            start_square = '%s%i' % (f1, r1)
            board_copy[r1][f1] = ' '
            board_copy[r2][f2] = piece

        return (board_copy, start_square, end_square)


    def _move_king(self, m, p, board):

        board_copy = copy.deepcopy(board)

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'K'
        else:
            piece = 'k'


        ###  handling king-side castling
        if (m == 'O-O') and (p == 'white'):
            r1, f1 = 1, 'e'
            r2, f2 = 1, 'g'
        elif (m == 'O-O') and (p == 'black'):
            r1, f1 = 8, 'e'
            r2, f2 = 8, 'g'

        ###  handling queen-side castling
        elif (m == 'O-O-O') and (p == 'white'):
            r1, f1 = 1, 'e'
            r2, f2 = 1, 'c'
        elif (m == 'O-O-O') and (p == 'black'):
            r1, f1 = 8, 'e'
            r2, f2 = 8, 'c'

        ###  handling fully-disambiguated move
        elif len(m) == 5:
            f1, r1 = m[1], int(m[2])
            f2, r2 = m[3], int(m[4])

        ###  else identify starting square
        else:
            f2, r2 = m[1], int(m[2])
            found_king = False
            for r1 in set(RANKS).intersection(set([r2-1, r2, r2+1])):
                for i_file in set(range(8)).intersection(set([FILES.index(f2)-1, FILES.index(f2), FILES.index(f2)+1])):
                    f1 = FILES[i_file]
                    if board_copy[r1][f1] == piece:
                        found_king = True
                        break
                if found_king is True:
                    break

        ###  relabel start and end squares
        start_square = '%s%i' % (f1, r1)
        end_square = '%s%i' % (f2, r2)
        board_copy[r1][f1] = ' '
        board_copy[r2][f2] = piece

        return (board_copy, start_square, end_square)


    def _move_castle(self, m, p, board):

        board_copy = copy.deepcopy(board)

        if (p == 'white'):
            if m.strip('+').strip('#') == 'O-O':
                board_copy[1]['e'] = ' '
                board_copy[1]['h'] = ' '
                board_copy[1]['g'] = 'K'
                board_copy[1]['f'] = 'R'
                start_square, end_square = 'e1', 'g1'

            if m.strip('+').strip('#') == 'O-O-O':
                board_copy[1]['e'] = ' '
                board_copy[1]['a'] = ' '
                board_copy[1]['c'] = 'K'
                board_copy[1]['d'] = 'R'
                start_square, end_square = 'e1', 'c1'

        elif (p == 'black'):
            if m.strip('+').strip('#') == 'O-O':
                board_copy[8]['e'] = ' '
                board_copy[8]['h'] = ' '
                board_copy[8]['g'] = 'k'
                board_copy[8]['f'] = 'r'
                start_square, end_square = 'e8', 'g8'

            if m.strip('+').strip('#') == 'O-O-O':
                board_copy[8]['e'] = ' '
                board_copy[8]['a'] = ' '
                board_copy[8]['c'] = 'k'
                board_copy[8]['d'] = 'r'
                start_square, end_square = 'e8', 'c8'

        return (board_copy, start_square, end_square)


    def _redraw_board(self, board=None):

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        ###  plotting current position
        for r in RANKS:
            for f in FILES:
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
                s += '%s | ' % board[r][f]
            print(s)
            print('-'*(4*8+1))


    def evaluate_depth1_single_proc(self, i_proc, board, df_legal_moves_0):



        for idx0, row0 in df_legal_moves_0.iterrows():
            ###  applying candidate move, getting opponent's followup moves
            next_board_0 = self.move_piece(row0['notation'], row0['player'], board=board, redraw=False)
            evaluation0 = self.get_eval(next_board_0, OTHER_PLAYER[row0['player']])
            df_legal_moves_1 = self.get_legal_moves(OTHER_PLAYER[row0['player']], board=next_board_0, get_eval=False)

            for idx1, row1 in df_legal_moves_1.iterrows():
                ###  applying candidate move, getting opponent's followup moves
                next_board_1 = self.move_piece(row1['notation'], row1['player'], board=next_board_0, redraw=False)
                evaluation1 = self.get_eval(next_board_1, OTHER_PLAYER[row1['player']])

                df_legal_moves_1.loc[idx1, 'evaluation'] = evaluation1
                df_legal_moves_1.loc[idx1, 'eval_seq'] = '%.2f_%.2f' % (evaluation0, evaluation1)
                df_legal_moves_1.loc[idx1, 'move_seq'] = '%s_%s' % (row0['notation'], row1['notation'])

            ###  propigating best move of opponent backwards
            if (row0['player'] == 'white') and (len(df_legal_moves_1)>0):
                best_move = df_legal_moves_1.sort_values(by=['evaluation'], ascending=True).iloc[0]
                df_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                df_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                df_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
            elif (row0['player'] == 'black') and (len(df_legal_moves_1)>0):
                best_move = df_legal_moves_1.sort_values(by=['evaluation'], ascending=False).iloc[0]
                df_legal_moves_0.loc[idx0, 'move_seq'] = best_move['move_seq']
                df_legal_moves_0.loc[idx0, 'eval_seq'] = best_move['eval_seq']
                df_legal_moves_0.loc[idx0, 'evaluation'] = best_move['evaluation']
            else:
                df_legal_moves_0.loc[idx0, 'evaluation'] = evaluation0

        ###  saving results
        if len(df_legal_moves_0) > 0:
            df_legal_moves_0.to_csv('DF_LEGAL_MOVES_PROC%02i.csv'%i_proc, index=None)


    def evaluate_multiproc(self, player, move_text, board):
        '''
        Evaluate the provided move from player on the provided board
        '''

        ###  applying the provided move, generating next board position
        ###  getting opponent's following legal moves
        next_board_0 = self.move_piece(move_text, player, board=board, redraw=False)
        df_legal_moves_0 = self.get_legal_moves(OTHER_PLAYER[player], board=next_board_0, get_eval=False)

        processes = []
        for i_proc in range(CPU_COUNT):

            ###  don't spawn more processes than there are candidate moves
            if i_proc == len(df_legal_moves_0):
                break

            ###  initializing process on segment of candidatate moves
            ii = range(i_proc, len(df_legal_moves_0), CPU_COUNT)
            df_legal_moves_i = df_legal_moves_0.loc[ii]

            ctx = multiprocessing.get_context('spawn')
            args = (i_proc, next_board_0, df_legal_moves_i)
            proc = multiprocessing.Process(target=self.evaluate_depth1_single_proc, args=args)
            processes.append(proc)
            proc.start()

        ###  joining processes
        for proc in processes:
            proc.join()

        ###  loading files and deleting afterward
        fnames = glob.glob('DF_LEGAL_MOVES_PROC*csv')
        df_legal_moves_w_eval = pandas.concat([pandas.read_csv(f) for f in fnames])
        for f in fnames:
            os.remove(f)


        if player == 'white':
            best_move = df_legal_moves_w_eval.sort_values(by=['evaluation'], ascending=False).iloc[0]
            return best_move
        else:
            best_move = df_legal_moves_w_eval.sort_values(by=['evaluation'], ascending=True).iloc[0]
            return best_move


    def get_eval(self, board, player_on_move):

        ###  determining if position is checkmate
        df_legal_moves = self.get_legal_moves(player_on_move, board=board, get_eval=False)
        if (len(df_legal_moves) == 0) and (player_on_move == 'white'):
            evaluation = -9999

        elif (len(df_legal_moves) == 0) and (player_on_move == 'black'):
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
        pieces = ''
        for r in RANKS:
            pieces += ''.join(list(board[r].values()))
        delta_material = 0
        delta_material += MATERIAL_VALUES['p'] * (pieces.count('P') - pieces.count('p'))
        delta_material += MATERIAL_VALUES['n'] * (pieces.count('N') - pieces.count('n'))
        delta_material += MATERIAL_VALUES['b'] * (pieces.count('B') - pieces.count('b'))
        delta_material += MATERIAL_VALUES['r'] * (pieces.count('R') - pieces.count('r'))
        delta_material += MATERIAL_VALUES['q'] * (pieces.count('Q') - pieces.count('q'))
        delta_material += MATERIAL_VALUES['k'] * (pieces.count('K') - pieces.count('k'))


        ###  getting attacks
        n_targets_by_white, n_targets_by_black = self.get_attacked_squares(board)
        total_vision_by_white, total_vision_by_black = 0, 0
        n_0x_defends_by_white, n_0x_defends_by_black = 0, 0
        n_overattacks_by_white, n_overattacks_by_black = 0, 0
        central_control_by_white, central_control_by_black = 0, 0
        for r in RANKS:
            for f in FILES:
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


    def get_pawn_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of pawn's position

            r : int
                Rank of pawn's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''

        moves_dict = []
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        if player == 'white':

            if r == 7:
                promotion = True
            else:
                promotion = False

            ###  checking if square is available for advance
            if board[r+1][f] == ' ':

                if promotion == True:
                    for piece in ['Q', 'R', 'B', 'N']:
                        moves_dict.append({'player': 'white', 
                                           'piece': 'pawn', 
                                           'notation': '%s%i=%s' % (f, r+1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f, r+1)})
                else:
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%s%i' % (f, r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f, r+1)})

                ###  checking for two-square advance
                if (r == 2) and (board[r+2][f] == ' '):
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%s%i' % (f, r+2), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f, r+2)})

            ###  checking for captures to higher files
            if (f != 'h') and (board[r+1][FILES[FILES.index(f)+1]] in PIECES_BLACK.keys()):
                f2 = FILES[FILES.index(f)+1]
                if promotion == True:
                    for piece in ['Q', 'R', 'B', 'N']:
                        moves_dict.append({'player': 'white', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, f2, r+1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f2, r+1)})
                else:
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r+1)})

            ###  checking for en passant captures to higher files
            if (f != 'h') and (len(self.df_moves) > 0):
                ep_target = self.df_moves.iloc[-1]['ep_target']
                f2 = FILES[FILES.index(f)+1]
                if (ep_target != '-') and (r == 5) and (ep_target[1] == '6') and (ep_target[0] == f2):
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r+1)})

            ###  checking for captures to lower files
            if (f != 'a') and (board[r+1][FILES[FILES.index(f)-1]] in PIECES_BLACK.keys()):
                f2 = FILES[FILES.index(f)-1]
                if promotion == True:
                    for piece in ['Q', 'R', 'B', 'N']:
                        moves_dict.append({'player': 'white', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, f2, r+1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f2, r+1)})
                else:
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r+1)})

            ###  checking for en passant captures to higher files
            if (f != 'a') and (len(self.df_moves) > 0):
                ep_target = self.df_moves.iloc[-1]['ep_target']
                f2 = FILES[FILES.index(f)-1]
                if (ep_target != '-') and (r == 5) and (ep_target[1] == '6') and (ep_target[0] == f2):
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r+1)})



        elif player == 'black':

            if r == 2:
                promotion = True
            else:
                promotion = False

            ###  checking if square is available for advance
            if board[r-1][f] == ' ':

                if promotion == True:
                    for piece in ['q', 'r', 'b', 'n']:
                        moves_dict.append({'player': 'black', 
                                           'piece': 'pawn', 
                                           'notation': '%s%i=%s' % (f, r-1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f, r-1)})
                else:
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%s%i' % (f, r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f, r-1)})

                ###  checking for two-square advance
                if (r == 7) and (board[r-2][f] == ' '):
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%s%i' % (f, r-2), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f, r-2)})

            ###  checking for captures to higher file
            if (f != 'h') and (board[r-1][FILES[FILES.index(f)+1]] in PIECES_WHITE.keys()):
                f2 = FILES[FILES.index(f)+1]
                if promotion == True:
                    for piece in ['q', 'r', 'b', 'n']:
                        moves_dict.append({'player': 'black', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, f2, r-1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f2, r-1)})
                else:
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r-1)})

            ###  checking for en passant captures to higher files
            if (f != 'h') and (len(self.df_moves) > 0):
                ep_target = self.df_moves.iloc[-1]['ep_target']
                f2 = FILES[FILES.index(f)+1]
                if (ep_target != '-') and (r == 4) and (ep_target[1] == '3') and (ep_target[0] == f2):
                    moves_dict.append({'player': 'balck', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r-1)})

            ###  checking for captures to lower files
            if (f != 'a') and (board[r-1][FILES[FILES.index(f)-1]] in PIECES_WHITE.keys()):
                f2 = FILES[FILES.index(f)-1]
                if promotion == True:
                    for piece in ['q', 'r', 'b', 'n']:
                        moves_dict.append({'player': 'black', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, f2, r-1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (f2, r-1)})
                else:
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r-1)})

            ###  checking for en passant captures to lower files
            if (f != 'a') and (len(self.df_moves) > 0):
                ep_target = self.df_moves.iloc[-1]['ep_target']
                f2 = FILES[FILES.index(f)-1]
                if (ep_target != '-') and (r == 4) and (ep_target[1] == '3') and (ep_target[0] == f2):
                    moves_dict.append({'player': 'balck', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, f2, r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (f2, r-1)})


        return moves_dict


    def get_knight_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of knight's position

            r : int
                Rank of knight's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''

        moves_dict = []
        start_square = '%s%i' % (f, r)
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        if player == 'white':
            player_pieces = PIECES_WHITE.keys()
            opponent_pieces = PIECES_BLACK.keys()
        elif player == 'black':
            player_pieces = PIECES_BLACK.keys()
            opponent_pieces = PIECES_WHITE.keys()



        ###  identifying potential end squares and what pieces occupy them
        if (3 <= r) and (1 <= FILES.index(f)):
            end_square = '%s%i' % (FILES[FILES.index(f)-1], r-2)
            end_square_occupant = board[r-2][FILES[FILES.index(f)-1]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (2 <= r) and (2 <= FILES.index(f)):
            end_square = '%s%i' % (FILES[FILES.index(f)-2], r-1)
            end_square_occupant = board[r-1][FILES[FILES.index(f)-2]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (r <= 7) and (2 <= FILES.index(f)):
            end_square = '%s%i' % (FILES[FILES.index(f)-2], r+1)
            end_square_occupant = board[r+1][FILES[FILES.index(f)-2]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (r <= 6) and (1 <= FILES.index(f)):
            end_square = '%s%i' % (FILES[FILES.index(f)-1], r+2)
            end_square_occupant = board[r+2][FILES[FILES.index(f)-1]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (r <= 6) and (FILES.index(f) <= 6):
            end_square = '%s%i' % (FILES[FILES.index(f)+1], r+2)
            end_square_occupant = board[r+2][FILES[FILES.index(f)+1]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (r <= 7) and (FILES.index(f) <= 5):
            end_square = '%s%i' % (FILES[FILES.index(f)+2], r+1)
            end_square_occupant = board[r+1][FILES[FILES.index(f)+2]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (2 <= r) and (FILES.index(f) <= 5):
            end_square = '%s%i' % (FILES[FILES.index(f)+2], r-1)
            end_square_occupant = board[r-1][FILES[FILES.index(f)+2]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
        if (3 <= r) and (FILES.index(f) <= 6):
            end_square = '%s%i' % (FILES[FILES.index(f)+1], r-2)
            end_square_occupant = board[r-2][FILES[FILES.index(f)+1]]
            ###  if legal, adding move to list
            if end_square_occupant == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            elif end_square_occupant in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'knight', 
                                   'notation': 'N%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})

        return moves_dict


    def get_bishop_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of bishop's position

            r : int
                Rank of bishop's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''

        moves_dict = []
        start_square = '%s%i' % (f, r)
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        if player == 'white':
            player_pieces = PIECES_WHITE.keys()
            opponent_pieces = PIECES_BLACK.keys()
        elif player == 'black':
            player_pieces = PIECES_BLACK.keys()
            opponent_pieces = PIECES_WHITE.keys()


        ###  identifying squares via ray casting
        ###  up-right diagonal
        for d in range(1, 8, 1):

            ###  checking if square is valid
            if (r+d in RANKS) and (FILES.index(f)+d in range(8)):
                end_square = '%s%i' % (FILES[FILES.index(f)+d], r+d)
                ###  checking if square is open
                if board[r+d][FILES[FILES.index(f)+d]] == ' ':
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%s%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                ###  checking if square is occupied by current-player piece
                elif board[r+d][FILES[FILES.index(f)+d]] in player_pieces:
                    break
                ###  checking if square is occupied by opponent piece
                elif board[r+d][FILES[FILES.index(f)+d]] in opponent_pieces:
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%sx%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                    break


        ###  down-right diagonal
        for d in range(1, 8, 1):

            ###  checking if square is valid
            if (r-d in RANKS) and (FILES.index(f)+d in range(8)):
                end_square = '%s%i' % (FILES[FILES.index(f)+d], r-d)
                ###  checking if square is open
                if board[r-d][FILES[FILES.index(f)+d]] == ' ':
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%s%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                ###  checking if square is occupied by current-player piece
                elif board[r-d][FILES[FILES.index(f)+d]] in player_pieces:
                    break
                ###  checking if square is occupied by opponent piece
                elif board[r-d][FILES[FILES.index(f)+d]] in opponent_pieces:
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%sx%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                    break


        ###  down-left diagonal
        for d in range(1, 8, 1):

            ###  checking if square is valid
            if (r-d in RANKS) and (FILES.index(f)-d in range(8)):
                end_square = '%s%i' % (FILES[FILES.index(f)-d], r-d)
                ###  checking if square is open
                if board[r-d][FILES[FILES.index(f)-d]] == ' ':
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%s%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                ###  checking if square is occupied by current-player piece
                elif board[r-d][FILES[FILES.index(f)-d]] in player_pieces:
                    break
                ###  checking if square is occupied by opponent piece
                elif board[r-d][FILES[FILES.index(f)-d]] in opponent_pieces:
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%sx%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                    break


        ###  up-left diagonal
        for d in range(1, 8, 1):

            ###  checking if square is valid
            if (r+d in RANKS) and (FILES.index(f)-d in range(8)):
                end_square = '%s%i' % (FILES[FILES.index(f)-d], r+d)
                ###  checking if square is open
                if board[r+d][FILES[FILES.index(f)-d]] == ' ':
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%s%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                ###  checking if square is occupied by current-player piece
                elif board[r+d][FILES[FILES.index(f)-d]] in player_pieces:
                    break
                ###  checking if square is occupied by opponent piece
                elif board[r+d][FILES[FILES.index(f)-d]] in opponent_pieces:
                    moves_dict.append({'player': player, 
                                       'piece': 'bishop', 
                                       'notation': 'B%sx%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                    break

        return moves_dict


    def get_rook_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of rook's position

            r : int
                Rank of rook's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''


        moves_dict = []
        start_square = '%s%i' % (f, r)
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        if player == 'white':
            player_pieces = PIECES_WHITE.keys()
            opponent_pieces = PIECES_BLACK.keys()
        elif player == 'black':
            player_pieces = PIECES_BLACK.keys()
            opponent_pieces = PIECES_WHITE.keys()


        ###  right files
        for f2 in FILES[FILES.index(f)+1:]:
            end_square = '%s%i' % (f2, r)

            ###  checking if square is open
            if board[r][f2] == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            ###  checking if square is occupied by current-player piece
            elif board[r][f2] in player_pieces:
                break
            ###  checking if square is occupied by opponent piece
            elif board[r][f2] in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
                break


        ###  down ranks
        for r2 in RANKS[:r-1][::-1]:
            end_square = '%s%i' % (f, r2)

            ###  checking if square is open
            if board[r2][f] == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            ###  checking if square is occupied by current-player piece
            elif board[r2][f] in player_pieces:
                break
            ###  checking if square is occupied by opponent piece
            elif board[r2][f] in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
                break


        ###  left files
        for f2 in FILES[:FILES.index(f)][::-1]:
            end_square = '%s%i' % (f2, r)

            ###  checking if square is open
            if board[r][f2] == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            ###  checking if square is occupied by current-player piece
            elif board[r][f2] in player_pieces:
                break
            ###  checking if square is occupied by opponent piece
            elif board[r][f2] in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
                break


        ###  up ranks
        for r2 in RANKS[r:]:
            end_square = '%s%i' % (f, r2)

            ###  checking if square is open
            if board[r2][f] == ' ':
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%s%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
            ###  checking if square is occupied by current-player piece
            elif board[r2][f] in player_pieces:
                break
            ###  checking if square is occupied by opponent piece
            elif board[r2][f] in opponent_pieces:
                moves_dict.append({'player': player, 
                                   'piece': 'rook', 
                                   'notation': 'R%sx%s' % (start_square, end_square), 
                                   'start_square': start_square, 
                                   'end_square': end_square})
                break

        return moves_dict


    def get_queen_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of queen's position

            r : int
                Rank of queen's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''


        moves_dict = []
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        ###  grabbing all bishop- and rook-like moves
        moves_dict += [m for m in self.get_bishop_moves(f, r, player, board)]
        moves_dict += [m for m in self.get_rook_moves(f, r, player, board)]

        ###  replacing refs to "bishop" and "rook" with "queen"
        for m in moves_dict:
            m['piece'] = 'queen'
            m['notation'] = 'Q' + m['notation'][1:]

        return moves_dict


    def get_king_moves(self, f, r, player, board=None):
        '''
        Parameters
        ----------
            f : str
                File of king's position

            r : int
                Rank of king's position

            player : str
                Either "white" or "black"

            board : dict
                Full layout of chessboard, if not provided use current board
        '''

        moves_dict = []
        start_square = '%s%i' % (f, r)
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        if player == 'white':
            opponent_pieces = PIECES_BLACK.keys()
        elif player == 'black':
            opponent_pieces = PIECES_WHITE.keys()


        ###  iterating over squares adjacent to king
        for r2 in set(RANKS).intersection(set([r-1, r, r+1])):
            for i_file in set(range(8)).intersection(set([FILES.index(f)-1, FILES.index(f), FILES.index(f)+1])):
                f2 = FILES[i_file]
                end_square = '%s%i' % (f2, r2)
                if board[r2][f2] == ' ':
                    moves_dict.append({'player': player, 
                                       'piece': 'king', 
                                       'notation': 'K%s%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})
                elif board[r2][f2] in opponent_pieces:
                    moves_dict.append({'player': player, 
                                       'piece': 'king', 
                                       'notation': 'K%sx%s' % (start_square, end_square), 
                                       'start_square': start_square, 
                                       'end_square': end_square})

        ###  identifying castling moves
        if (player == 'white'):
            if ('K' in self.castle_rights) and (board[1]['f']+board[1]['g'] == 2*' '):
                moves_dict.append({'player': player, 
                                   'piece': 'king', 
                                   'notation': 'O-O', 
                                   'start_square': 'e1', 
                                   'end_square': 'g1'})
            if ('Q' in self.castle_rights) and (board[1]['b']+board[1]['c']+board[1]['d'] == 3*' '):
                moves_dict.append({'player': player, 
                                   'piece': 'king', 
                                   'notation': 'O-O-O', 
                                   'start_square': 'e1', 
                                   'end_square': 'c1'})
        if (player == 'black'):
            if ('k' in self.castle_rights) and (board[8]['f']+board[8]['g'] == 2*' '):
                moves_dict.append({'player': player, 
                                   'piece': 'king', 
                                   'notation': 'O-O', 
                                   'start_square': 'e8', 
                                   'end_square': 'g8'})
            if ('q' in self.castle_rights) and (board[8]['b']+board[8]['c']+board[8]['d'] == 3*' '):
                moves_dict.append({'player': player, 
                                   'piece': 'king', 
                                   'notation': 'O-O-O', 
                                   'start_square': 'e8', 
                                   'end_square': 'c8'})

        return moves_dict


    def filter_illegal_moves(self, df_candidate_moves, board=None):

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        ###  iterating over candidate moves
        n_attacks_by_white_current, n_attacks_by_black_current = self.get_attacked_squares(board)
        df_candidate_moves['legal'] = True
        for idx, row in df_candidate_moves.iterrows():

            if row['piece'] == 'pawn':
                (next_board, start_square, end_square, en_passants) = self._move_pawn(row['notation'], row['player'], board)

            if row['piece'] == 'knight':
                (next_board, start_square, end_square) = self._move_knight(row['notation'], row['player'], board)

            if row['piece'] == 'bishop':
                (next_board, start_square, end_square) = self._move_bishop(row['notation'], row['player'], board)

            if row['piece'] == 'rook':
                (next_board, start_square, end_square) = self._move_rook(row['notation'], row['player'], board)

            if row['piece'] == 'king':
                (next_board, start_square, end_square) = self._move_king(row['notation'], row['player'], board)

            if row['piece'] == 'queen':
                (next_board, start_square, end_square) = self._move_queen(row['notation'], row['player'], board)

            ###  locating squares occupied by kings
            for r in RANKS:
                for f in FILES:
                    if next_board[r][f] == 'K':
                        r_white_king, f_white_king = r, f
                    if next_board[r][f] == 'k':
                        r_black_king, f_black_king = r, f

            ###  examining if king is under attack after applying candidate move
            n_attacks_by_white, n_attacks_by_black = self.get_attacked_squares(next_board)
            if (row['player'] == 'white') and (n_attacks_by_black[r_white_king][f_white_king] > 0):
                df_candidate_moves.loc[idx, 'legal'] = False
            if (row['player'] == 'black') and (n_attacks_by_white[r_black_king][f_black_king] > 0):
                df_candidate_moves.loc[idx, 'legal'] = False

            ###  for castling, make sure king is not alredy in check of moving through check
            if (row['player'] == 'white') and (row['notation'] == 'O-O'):
                if (n_attacks_by_black_current[1]['e'] > 0) or (n_attacks_by_black_current[1]['f'] > 0):
                    df_candidate_moves.loc[idx, 'legal'] = False
            if (row['player'] == 'black') and (row['notation'] == 'O-O'):
                if (n_attacks_by_white_current[8]['e'] > 0) or (n_attacks_by_white_current[8]['f'] > 0):
                    df_candidate_moves.loc[idx, 'legal'] = False
            if (row['player'] == 'white') and (row['notation'] == 'O-O-O'):
                if (n_attacks_by_black_current[1]['e'] > 0) or (n_attacks_by_black_current[1]['d'] > 0):
                    df_candidate_moves.loc[idx, 'legal'] = False
            if (row['player'] == 'black') and (row['notation'] == 'O-O-O'):
                if (n_attacks_by_white_current[8]['e'] > 0) or (n_attacks_by_white_current[8]['d'] > 0):
                    df_candidate_moves.loc[idx, 'legal'] = False

        return df_candidate_moves.query('legal==True').reset_index(drop=True)


    def get_attacked_squares(self, board=None):

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])


        n_attacks_by_white = {8:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              7:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              6:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              5:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              4:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              3:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              2:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              1:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}}

        n_attacks_by_black = {8:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              7:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              6:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              5:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              4:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              3:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              2:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}, 
                              1:{'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0}}


        ###  iterating over squares
        for r in RANKS:
            for f in FILES:

                ###  hangling pawn attacks
                if (board[r][f] == 'P') and (f != 'h'):
                    n_attacks_by_white[r+1][FILES[FILES.index(f)+1]] += 1
                if (board[r][f] == 'P') and (f != 'a'):
                    n_attacks_by_white[r+1][FILES[FILES.index(f)-1]] += 1
                if (board[r][f] == 'p') and (f != 'h'):
                    n_attacks_by_black[r-1][FILES[FILES.index(f)+1]] += 1
                if (board[r][f] == 'p') and (f != 'a'):
                    n_attacks_by_black[r-1][FILES[FILES.index(f)-1]] += 1

                ###  handling attacks by knights
                if (board[r][f] == 'N'):
                    if (3 <= r) and (1 <= FILES.index(f)):
                        n_attacks_by_white[r-2][FILES[FILES.index(f)-1]] += 1
                    if (2 <= r) and (2 <= FILES.index(f)):
                        n_attacks_by_white[r-1][FILES[FILES.index(f)-2]] += 1
                    if (r <= 7) and (2 <= FILES.index(f)):
                        n_attacks_by_white[r+1][FILES[FILES.index(f)-2]] += 1
                    if (r <= 6) and (1 <= FILES.index(f)):
                        n_attacks_by_white[r+2][FILES[FILES.index(f)-1]] += 1
                    if (r <= 6) and (FILES.index(f) <= 6):
                        n_attacks_by_white[r+2][FILES[FILES.index(f)+1]] += 1
                    if (r <= 7) and (FILES.index(f) <= 5):
                        n_attacks_by_white[r+1][FILES[FILES.index(f)+2]] += 1
                    if (2 <= r) and (FILES.index(f) <= 5):
                        n_attacks_by_white[r-1][FILES[FILES.index(f)+2]] += 1
                    if (3 <= r) and (FILES.index(f) <= 6):
                        n_attacks_by_white[r-2][FILES[FILES.index(f)+1]] += 1
                if (board[r][f] == 'n'):
                    if (3 <= r) and (1 <= FILES.index(f)):
                        n_attacks_by_black[r-2][FILES[FILES.index(f)-1]] += 1
                    if (2 <= r) and (2 <= FILES.index(f)):
                        n_attacks_by_black[r-1][FILES[FILES.index(f)-2]] += 1
                    if (r <= 7) and (2 <= FILES.index(f)):
                        n_attacks_by_black[r+1][FILES[FILES.index(f)-2]] += 1
                    if (r <= 6) and (1 <= FILES.index(f)):
                        n_attacks_by_black[r+2][FILES[FILES.index(f)-1]] += 1
                    if (r <= 6) and (FILES.index(f) <= 6):
                        n_attacks_by_black[r+2][FILES[FILES.index(f)+1]] += 1
                    if (r <= 7) and (FILES.index(f) <= 5):
                        n_attacks_by_black[r+1][FILES[FILES.index(f)+2]] += 1
                    if (2 <= r) and (FILES.index(f) <= 5):
                        n_attacks_by_black[r-1][FILES[FILES.index(f)+2]] += 1
                    if (3 <= r) and (FILES.index(f) <= 6):
                        n_attacks_by_black[r-2][FILES[FILES.index(f)+1]] += 1

                ###  handling attacks by rooks (and queen)
                if (board[r][f] in ['R', 'Q']):
                    ###  right files
                    for f2 in FILES[FILES.index(f)+1:]:
                        n_attacks_by_white[r][f2] += 1
                        if board[r][f2] != ' ':
                            break
                    ###  down ranks
                    for r2 in RANKS[:r-1][::-1]:
                        n_attacks_by_white[r2][f] += 1
                        if board[r2][f] != ' ':
                            break
                    ###  left files
                    for f2 in FILES[:FILES.index(f)][::-1]:
                        n_attacks_by_white[r][f2] += 1
                        if board[r][f2] != ' ':
                            break
                    ###  up ranks
                    for r2 in RANKS[r:]:
                        n_attacks_by_white[r2][f] += 1
                        if board[r2][f] != ' ':
                            break
                if (board[r][f] in ['r', 'q']):
                    ###  right files
                    for f2 in FILES[FILES.index(f)+1:]:
                        n_attacks_by_black[r][f2] += 1
                        if board[r][f2] != ' ':
                            break
                    ###  down ranks
                    for r2 in RANKS[:r-1][::-1]:
                        n_attacks_by_black[r2][f] += 1
                        if board[r2][f] != ' ':
                            break
                    ###  left files
                    for f2 in FILES[:FILES.index(f)][::-1]:
                        n_attacks_by_black[r][f2] += 1
                        if board[r][f2] != ' ':
                            break
                    ###  up ranks
                    for r2 in RANKS[r:]:
                        n_attacks_by_black[r2][f] += 1
                        if board[r2][f] != ' ':
                            break

                ###  handling attacks by bishops (and queen)
                if (board[r][f] in ['B', 'Q']):
                    ###  up-right diagonal
                    for d in range(1, 8, 1):
                        if (r+d in RANKS) and (FILES.index(f)+d in range(8)):
                            n_attacks_by_white[r+d][FILES[FILES.index(f)+d]] += 1
                            if board[r+d][FILES[FILES.index(f)+d]] != ' ':
                                break
                    ###  down-right diagonal
                    for d in range(1, 8, 1):
                        if (r-d in RANKS) and (FILES.index(f)+d in range(8)):
                            n_attacks_by_white[r-d][FILES[FILES.index(f)+d]] += 1
                            if board[r-d][FILES[FILES.index(f)+d]] != ' ':
                                break
                    ###  down-left diagonal
                    for d in range(1, 8, 1):
                        if (r-d in RANKS) and (FILES.index(f)-d in range(8)):
                            n_attacks_by_white[r-d][FILES[FILES.index(f)-d]] += 1
                            if board[r-d][FILES[FILES.index(f)-d]] != ' ':
                                break
                    ###  up-left diagonal
                    for d in range(1, 8, 1):
                        if (r+d in RANKS) and (FILES.index(f)-d in range(8)):
                            n_attacks_by_white[r+d][FILES[FILES.index(f)-d]] += 1
                            if board[r+d][FILES[FILES.index(f)-d]] != ' ':
                                break
                if (board[r][f] in ['b', 'q']):
                    ###  up-right diagonal
                    for d in range(1, 8, 1):
                        if (r+d in RANKS) and (FILES.index(f)+d in range(8)):
                            n_attacks_by_black[r+d][FILES[FILES.index(f)+d]] += 1
                            if board[r+d][FILES[FILES.index(f)+d]] != ' ':
                                break
                    ###  down-right diagonal
                    for d in range(1, 8, 1):
                        if (r-d in RANKS) and (FILES.index(f)+d in range(8)):
                            n_attacks_by_black[r-d][FILES[FILES.index(f)+d]] += 1
                            if board[r-d][FILES[FILES.index(f)+d]] != ' ':
                                break
                    ###  down-left diagonal
                    for d in range(1, 8, 1):
                        if (r-d in RANKS) and (FILES.index(f)-d in range(8)):
                            n_attacks_by_black[r-d][FILES[FILES.index(f)-d]] += 1
                            if board[r-d][FILES[FILES.index(f)-d]] != ' ':
                                break
                    ###  up-left diagonal
                    for d in range(1, 8, 1):
                        if (r+d in RANKS) and (FILES.index(f)-d in range(8)):
                            n_attacks_by_black[r+d][FILES[FILES.index(f)-d]] += 1
                            if board[r+d][FILES[FILES.index(f)-d]] != ' ':
                                break

                ###  handling attacks by kings
                if (board[r][f] == 'K'):
                    ###  iterating over squares adjacent to king
                    for r2 in set(RANKS).intersection(set([r-1, r, r+1])):
                        for i_file in set(range(8)).intersection(set([FILES.index(f)-1, FILES.index(f), FILES.index(f)+1])):
                            f2 = FILES[i_file]
                            if '%s%i'%(f,r) != '%s%i'%(f2,r2):
                                n_attacks_by_white[r2][f2] += 1
                if (board[r][f] == 'k'):
                    ###  iterating over squares adjacent to king
                    for r2 in set(RANKS).intersection(set([r-1, r, r+1])):
                        for i_file in set(range(8)).intersection(set([FILES.index(f)-1, FILES.index(f), FILES.index(f)+1])):
                            f2 = FILES[i_file]
                            if '%s%i'%(f,r) != '%s%i'%(f2,r2):
                                n_attacks_by_black[r2][f2] += 1

        return (n_attacks_by_white, n_attacks_by_black)


    def print_n_attacks(self, board=None):

        n_attacks_by_white, n_attacks_by_black = self.get_attacked_squares(board=board)

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
 

    def print_png_body(self, df_moves=None):

        if df_moves is None:
            df_moves = self.df_moves

        pgn_text = ''
        for idx, row in df_moves.iterrows():
            if row['player'] == 'white':
                pgn_text += '%i. ' % (1+idx//2)

            pgn_text += '%s ' % row['notation']

        print(pgn_text)



if __name__ == '__main__':


    if len(sys.argv) == 1:
        asdf = chessBoard()

    elif len(sys.argv) == 2:
        asdf = chessBoard(sys.argv[1])