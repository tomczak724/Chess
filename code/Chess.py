
import copy
import time
import numpy
import pandas
import imageio
import matplotlib
from matplotlib import pyplot


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



class chessBoard(object):

    def __init__(self, pgn_file=None):

        self.pgn_file = pgn_file
        if self.pgn_file is not None:
            self.df_pgn = self.load_pgn()
            self.current_player = 'white'
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


        ###  setting up lists of legal moves
        self.legal_moves = self.get_legal_moves()
        self.previous_move = {'player':'', 'move':''}
        self.castle_rights = 'KQkq'

        self.df_moves = pandas.DataFrame(columns=['player', 'notation', 'piece', 'start_square', 'end_square', 'capture_flag', 'check_flag', 'fen_board_position', 'ep_target', 'castle_rights', 'halfmove_count'])
        self.halfmove_count = 0
        self.fens = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1']


        ###  initializing figure
        self.fig, self.ax = pyplot.subplots(figsize=(6, 6))
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
        if pgn_file is not None:

            for idx, row in self.df_pgn.iterrows():

                self.next(1)
                if ('#' in row['white']) or ('++' in row['white']):
                    break

                self.next(1)
                if ('#' in row['black']) or ('++' in row['black']):
                    break

        ###  instantiating interactive variables
        self.selected_square = None

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
            if self.selected_square == '%s%i' % (f, r):
                self.selected_square = None
                self._redraw_board()
                return

            ###  checking if square is occupied
            self.selected_square = event.inaxes.get_label()
            print('\nselected_square = %s' % self.selected_square)
            board = copy.deepcopy(self.chess_boards[-1])

            moves = []

            if board[r][f] == 'P':
                moves = self.get_pawn_moves(f, r, 'white', board)

            elif board[r][f] == 'p':
                moves = self.get_pawn_moves(f, r, 'black', board)

            elif board[r][f] == 'N':
                moves = self.get_knight_moves(f, r, 'white', board)

            elif board[r][f] == 'n':
                moves = self.get_knight_moves(f, r, 'black', board)

            elif board[r][f] == 'B':
                moves = self.get_bishop_moves(f, r, 'white', board)

            elif board[r][f] == 'b':
                moves = self.get_bishop_moves(f, r, 'black', board)

            elif board[r][f] == 'R':
                moves = self.get_rook_moves(f, r, 'white', board)

            elif board[r][f] == 'r':
                moves = self.get_rook_moves(f, r, 'black', board)

            elif board[r][f] == 'Q':
                moves = self.get_queen_moves(f, r, 'white', board)

            elif board[r][f] == 'q':
                moves = self.get_queen_moves(f, r, 'black', board)

            elif board[r][f] == 'K':
                moves = self.get_king_moves(f, r, 'white', board)

            elif board[r][f] == 'k':
                moves = self.get_king_moves(f, r, 'black', board)


            ###  highlighting squares
            for move in moves:
                f, r = move['end_square'][0], int(move['end_square'][1])
                self.image_board[r][f].axes.set_facecolor('#ff9999')

            self._redraw_board()


    def load_pgn(self):

        ###  reading and parsing PGN text
        with open(self.pgn_file, 'r') as fopen:
            lines = fopen.readlines()
            text = ''.join(lines)
            text = text.replace('\n', ' ')

            ###  extracting text for moves
            text_moves = text.replace('.', '').split(' 1 ')[-1]
            while '  ' in text_moves:
                text_moves = text_moves.replace('  ', ' ')

            list_moves = ['1'] + text_moves.split(' ')
            return pandas.DataFrame(numpy.array(list_moves).reshape((int(len(list_moves)/3), 3)), columns=['turn', 'white', 'black'])


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
        fen += ' %s ' % self.current_player[0]

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
                print(self.df_pgn.query('turn=="%s"'%self.current_move)[self.current_player].iloc[0])
            self.move_piece(self.df_pgn.query('turn=="%s"'%self.current_move)[self.current_player].iloc[0], 
                            self.current_player, 
                            redraw=True)

            if self.current_player == 'white':
                self.previous_move['player'] = 'white'
                self.previous_move['move'] = self.df_pgn.query('turn=="%s"'%self.current_move)[self.current_player].iloc[0]
                self.current_player = 'black'
            else:
                self.current_player = 'white'
                self.previous_move['move'] = self.df_pgn.query('turn=="%s"'%self.current_move)[self.current_player].iloc[0]
                self.previous_move['player'] = 'black'
                self.current_move += 1

            if verbose is True:
                print('evaluation = %i' % self.evaluate(self.current_player))


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

        if player == 'white':
            next_player = 'b'
        else:
            next_player = 'w'

        self.fens.append('%s %s %s %s %i %i' % (fbp, next_player, self.castle_rights, ep_target, self.halfmove_count, (i_move+3)//2))

        self.halfmove_count += 1

        ###  redraw plot window and appending new board to history
        if redraw == True:
            self.chess_boards.append(new_board)
            self._redraw_board()

        else:
            return new_board


    def _move_pawn(self, m, p, board):

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
                board[r][f] = 'P'

                ###  checking if first move is two squares
                if board[r-1][f] == ' ':
                    board[r-2][f] = ' '
                    start_square = '%s%i' % (f, r-2)

                    ###  checking if en passant is available
                    if (r == 4) and (f != 'h') and (board[4][FILES[FILES.index(f)+1]] == 'p'):
                        en_passants.append('%sx%s3' % (FILES[FILES.index(f)+1], f))

                    if (r == 4) and (f != 'a') and (board[4][FILES[FILES.index(f)-1]] == 'p'):
                        en_passants.append('%sx%s3' % (FILES[FILES.index(f)-1], f))

                ###  if not then it was a one-square advance
                else:
                    board[r-1][f] = ' '
                    start_square = '%s%i' % (f, r-1)

            if p == 'black':
                board[r][f] = 'p'

                ###  checking if first move is two squares
                if board[r+1][f] == ' ':
                    board[r+2][f] = ' '
                    start_square = '%s%i' % (f, r+2)

                    ###  checking if en passant is available
                    if (r == 5) and (f != 'h') and (board[5][FILES[FILES.index(f)+1]] == 'P'):
                        en_passants.append('%sx%s6' % (FILES[FILES.index(f)+1], f))

                    if (r == 5) and (f != 'a') and (board[5][FILES[FILES.index(f)-1]] == 'P'):
                        en_passants.append('%sx%s6' % (FILES[FILES.index(f)-1], f))

                ###  if not then it was a one-square advance
                else:
                    board[r+1][f] = ' '
                    start_square = '%s%i' % (f, r+1)

        ###  pawn captures (not including promotion)
        if (len(m.strip('+').strip('#')) == 4) and (m[1] == 'x'):
            f1, f2, r = m[0], m[2], int(m[3])

            if p == 'white':
                start_square = '%s%i' % (f1, r-1)

                ###  checking for en passant capture
                if (r == 6) and (board[r][f2] == ' ') and (board[r-1][f2] == 'p'):
                    board[r-1][f2] = ' '

                board[r][f2] = 'P'
                board[r-1][f1] = ' '

            if p == 'black':
                start_square = '%s%i' % (f1, r+1)

                ###  checking for en passant capture
                if (r == 3) and (board[r][f2] == ' ') and (board[r+1][f2] == 'P'):
                    board[r+1][f2] = ' '

                board[r][f2] = 'p'
                board[r+1][f1] = ' '

        ###  pawn promotions
        if '=' in m:

            ###  captures with promotion
            if m[1] == 'x':
                f1, f2, r, new_piece = m[0], m[2], int(m[3]), m.split('=')[1][0]

                if p == 'white':
                    start_square = '%s%i' % (f1, r-1)
                    board[r][f2] = new_piece
                    board[r-1][f1] = ' '

                if p == 'black':
                    start_square = '%s%i' % (f1, r+1)
                    board[r][f2] = new_piece.lower()
                    board[r+1][f1] = ' '

            ###  simple promotion
            else:
                f, r, new_piece = m[0], int(m[1]), m.split('=')[1][0]

                if p == 'white':
                    start_square = '%s%i' % (f, r-1)
                    board[r][f2] = new_piece
                    board[r-1][f1] = ' '

                if p == 'black':
                    start_square = '%s%i' % (f, r+1)
                    board[r][f2] = new_piece.lower()
                    board[r+1][f1] = ' '

        return (board, start_square, end_square, en_passants)


    def _move_rook(self, m, p, board):

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'R'
        else:
            piece = 'r'


        ###  resolving fully-disambiguated rook moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board[r1][f1] = ' '
            board[r2][f2] = piece

            ###  identifying start and end squares
            start_square = m[1:3]
            end_square = m[3:5]

        ###  resolving partially-disambiguated rook moves
        elif len(m) == 4:
            rf1, f2, r2 = m[1], m[2], int(m[3])

            ###  identifying starting square
            if rf1 in FILES:
                board[r2][rf1] = ' '
                start_square = '%s%i' % (rf1, r2)
            else:
                board[int(rf1)][f2] = ' '
                start_square = '%s%i' % (f2, int(rf1))

            end_square = '%s%i' % (f2, r2)
            board[r2][f2] = piece


        ###  resolving regular rook moves
        else:
            f2, r2 = m[1], int(m[2])
            end_square = m[1:3]

            ###  search for current location of rook, vacating square
            ###  cycling through RANKS from nearest to farthest
            for dr in range(1, 8, 1):
                if (r2-dr in RANKS) and (board[r2-dr][f2] == piece):
                    board[r2-dr][f2] = ' '
                    start_square = '%s%i' % (f2, r2-dr)
                    break
                if (r2+dr in RANKS) and (board[r2+dr][f2] == piece):
                    board[r2+dr][f2] = ' '
                    start_square = '%s%i' % (f2, r2+dr)
                    break

            ###  cycling through FILES from nearest to farthest
            for df in range(1, 8, 1):
                if (FILES.index(f2)-df in range(8)) and (board[r2][FILES[FILES.index(f2)-df]] == piece):
                    board[r2][FILES[FILES.index(f2)-df]] = ' '
                    start_square = '%s%i' % (FILES[FILES.index(f2)-df], r2)
                    break
                if (FILES.index(f2)+df in range(8)) and (board[r2][FILES[FILES.index(f2)+df]] == piece):
                    board[r2][FILES[FILES.index(f2)+df]] = ' '
                    start_square = '%s%i' % (FILES[FILES.index(f2)+df], r2)
                    break

            board[r2][f2] = piece

        return (board, start_square, end_square)


    def _move_knight(self, m, p, board):

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'N'
        else:
            piece = 'n'


        ###  resolving fully-disambiguated knight moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board[r1][f1] = ' '
            board[r2][f2] = piece

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
                    if board[r1][rf1] == piece:
                        board[r1][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board[int(rf1)][f1] == piece:
                        board[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board[r2][f2] = piece


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
                if board[r1][f1] == piece:
                    board[r1][f1] = ' '
                    start_square = '%s%i' % (f1, r1)
                    break

            board[r2][f2] = piece

        return (board, start_square, end_square)


    def _move_bishop(self, m, p, board):


        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'B'
        else:
            piece = 'b'


        ###  resolving fully-disambiguated bishop moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board[r1][f1] = ' '
            board[r2][f2] = piece

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
                    if board[r1][rf1] == piece:
                        board[r2][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board[int(rf1)][f1] == piece:
                        board[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board[r2][f2] = piece


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
                if board[r1][f1] == piece:
                    board[r1][f1] = ' '
                    start_square = '%s%i' % (f1, r1)

            board[r2][f2] = piece

        return (board, start_square, end_square)


    def _move_queen(self, m, p, board):


        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'Q'
        else:
            piece = 'q'


        ###  resolving fully-disambiguated queen moves
        if len(m) == 5:
            f1, r1, f2, r2 = m[1], int(m[2]), m[3], int(m[4])
            board[r1][f1] = ' '
            board[r2][f2] = piece

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
                    if board[r1][rf1] == piece:
                        board[r2][rf1] = ' '
                        start_square = '%s%i' % (rf1, r1)
                        break
            else:
                ###  cycling through FILES
                for f1 in FILES:
                    if board[int(rf1)][f1] == piece:
                        board[int(rf1)][f1] = ' '
                        start_square = '%s%i' % (f1, int(rf1))
                        break

            board[r2][f2] = piece

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
                    if board[r2+d][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2+d, FILES[FILES.index(f2)+d]
                        break
                    elif board[r2+d][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  right files
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (FILES.index(f2)+d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2, FILES[FILES.index(f2)+d]
                        break
                    elif board[r2][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  down-right diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS) and (FILES.index(f2)+d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2-d][FILES[FILES.index(f2)+d]] == piece:
                        found_queen = True
                        r1, f1 = r2-d, FILES[FILES.index(f2)+d]
                        break
                    elif board[r2-d][FILES[FILES.index(f2)+d]] != ' ':
                        break


            ###  down ranks
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2-d][f2] == piece:
                        found_queen = True
                        r1, f1 = r2-d, f2
                        break
                    elif board[r2-d][f2] != ' ':
                        break


            ###  down-left diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2-d in RANKS) and (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2-d][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2-d, FILES[FILES.index(f2)-d]
                        break
                    elif board[r2-d][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  left files
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2, FILES[FILES.index(f2)-d]
                        break
                    elif board[r2][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  up-left diagonal
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2+d in RANKS) and (FILES.index(f2)-d in range(8)):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2+d][FILES[FILES.index(f2)-d]] == piece:
                        found_queen = True
                        r1, f1 = r2+d, FILES[FILES.index(f2)-d]
                        break
                    elif board[r2+d][FILES[FILES.index(f2)-d]] != ' ':
                        break


            ###  up ranks
            for d in range(1, 8, 1):
                if found_queen == True:
                    break

                ###  checking if square is valid
                if (r2+d in RANKS):

                    ###  checking if square is occupied, truncate search if blocked by another piece
                    if board[r2+d][f2] == piece:
                        found_queen = True
                        r1, f1 = r2+d, f2
                        break
                    elif board[r2+d][f2] != ' ':
                        break


            start_square = '%s%i' % (f1, r1)
            board[r1][f1] = ' '
            board[r2][f2] = piece

        return (board, start_square, end_square)


    def _move_king(self, m, p, board):

        ###  removing extraneous notation
        m = m.strip('+').strip('#').replace('x', '')
        if p == 'white':
            piece = 'K'
        else:
            piece = 'k'

        ###  identifying starting square
        f2, r2 = m[1], int(m[2])
        end_square = m[1:3]

        for r1 in set(RANKS).intersection(set([r2-1, r2, r2+1])):
            for i_file in set(range(8)).intersection(set([FILES.index(f2)-1, FILES.index(f2), FILES.index(f2)+1])):
                f1 = FILES[i_file]
                if board[r1][f1] == piece:
                    start_square = '%s%i' % (f1, r1)
                    board[r1][f1] = ' '

        board[r2][f2] = piece

        return (board, start_square, end_square)


    def _move_castle(self, m, p, board):

        if (p == 'white'):
            if m.strip('+').strip('#') == 'O-O':
                board[1]['e'] = ' '
                board[1]['h'] = ' '
                board[1]['g'] = 'K'
                board[1]['f'] = 'R'
                start_square, end_square = 'e1', 'g1'

            if m.strip('+').strip('#') == 'O-O-O':
                board[1]['e'] = ' '
                board[1]['a'] = ' '
                board[1]['c'] = 'K'
                board[1]['d'] = 'R'
                start_square, end_square = 'e1', 'c1'

        elif (p == 'black'):
            if m.strip('+').strip('#') == 'O-O':
                board[8]['e'] = ' '
                board[8]['h'] = ' '
                board[8]['g'] = 'k'
                board[8]['f'] = 'r'
                start_square, end_square = 'e8', 'g8'

            if m.strip('+').strip('#') == 'O-O-O':
                board[8]['e'] = ' '
                board[8]['a'] = ' '
                board[8]['c'] = 'k'
                board[8]['d'] = 'r'
                start_square, end_square = 'e8', 'c8'

        return (board, start_square, end_square)


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



    def evaluate(self, player, board=None):
        '''
        Evaluate the current position of the board with `player` to move
        '''

        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        ###  counting material value
        pieces = ''
        for r in RANKS:
            pieces += ''.join(list(board[r].values()))

        evaluation = 0
        evaluation += pieces.count('P')
        evaluation += pieces.count('N')*3
        evaluation += pieces.count('B')*3
        evaluation += pieces.count('R')*5
        evaluation += pieces.count('Q')*9
        evaluation += pieces.count('K')*9999

        evaluation -= pieces.count('p')
        evaluation -= pieces.count('n')*3
        evaluation -= pieces.count('b')*3
        evaluation -= pieces.count('r')*5
        evaluation -= pieces.count('q')*9
        evaluation -= pieces.count('k')*9999







        return evaluation


    def get_legal_moves(self, board=None):
        '''
        Description
        -----------
            Returns the list of all legal moves for both players
            given a board setup. In no board is given then
            the current board position is used.
        '''

        moves = {'white':[], 'black':[]}
        list_leagal_moves = []
        df_legal_moves = pandas.DataFrame(['player', 'piece', 'notation', 'start_square', 'end_square'])
        if board is None:
            board = copy.deepcopy(self.chess_boards[-1])

        for r in RANKS:
            for f in FILES:

                if board[r][f] == 'P':
                    list_leagal_moves += self.get_pawn_moves(f, r, 'white', board)
                if board[r][f] == 'N':
                    list_leagal_moves += self.get_knight_moves(f, r, 'white', board)
                if board[r][f] == 'B':
                    list_leagal_moves += self.get_bishop_moves(f, r, 'white', board)
                if board[r][f] == 'R':
                    list_leagal_moves += self.get_rook_moves(f, r, 'white', board)
                if board[r][f] == 'Q':
                    list_leagal_moves += self.get_queen_moves(f, r, 'white', board)
                if board[r][f] == 'K':
                    list_leagal_moves += self.get_king_moves(f, r, 'white', board)

                if board[r][f] == 'p':
                    list_leagal_moves += self.get_pawn_moves(f, r, 'black', board)
                if board[r][f] == 'n':
                    list_leagal_moves += self.get_knight_moves(f, r, 'black', board)
                if board[r][f] == 'b':
                    list_leagal_moves += self.get_bishop_moves(f, r, 'black', board)
                if board[r][f] == 'r':
                    list_leagal_moves += self.get_rook_moves(f, r, 'black', board)
                if board[r][f] == 'q':
                    list_leagal_moves += self.get_queen_moves(f, r, 'black', board)
                if board[r][f] == 'k':
                    list_leagal_moves += self.get_king_moves(f, r, 'black', board)

        return pandas.DataFrame(list_leagal_moves)


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
                if promotion == True:
                    for piece in ['Q', 'R', 'B', 'N']:
                        moves_dict.append({'player': 'white', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, FILES[FILES.index(f)+1], r+1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (FILES[FILES.index(f)+1], r+1)})
                else:
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, FILES[FILES.index(f)+1], r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (FILES[FILES.index(f)+1], r+1)})

            ###  checking for captures to lower files
            if (f != 'a') and (board[r+1][FILES[FILES.index(f)-1]] in PIECES_BLACK.keys()):
                if promotion == True:
                    for piece in ['Q', 'R', 'B', 'N']:
                        moves_dict.append({'player': 'white', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, FILES[FILES.index(f)-1], r+1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (FILES[FILES.index(f)-1], r+1)})
                else:
                    moves_dict.append({'player': 'white', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, FILES[FILES.index(f)-1], r+1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (FILES[FILES.index(f)-1], r+1)})


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
                if promotion == True:
                    for piece in ['q', 'r', 'b', 'n']:
                        moves_dict.append({'player': 'black', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, FILES[FILES.index(f)+1], r-1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (FILES[FILES.index(f)+1], r-1)})
                else:
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, FILES[FILES.index(f)+1], r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (FILES[FILES.index(f)+1], r-1)})

            ###  checking for captures to lower files
            if (f != 'a') and (board[r-1][FILES[FILES.index(f)-1]] in PIECES_WHITE.keys()):
                if promotion == True:
                    for piece in ['q', 'r', 'b', 'n']:
                        moves_dict.append({'player': 'black', 
                                           'piece': 'pawn', 
                                           'notation': '%sx%s%i=%s' % (f, FILES[FILES.index(f)-1], r-1, piece), 
                                           'start_square': '%s%i' % (f, r), 
                                           'end_square': '%s%i' % (FILES[FILES.index(f)-1], r-1)})
                else:
                    moves_dict.append({'player': 'black', 
                                       'piece': 'pawn', 
                                       'notation': '%sx%s%i' % (f, FILES[FILES.index(f)-1], r-1), 
                                       'start_square': '%s%i' % (f, r), 
                                       'end_square': '%s%i' % (FILES[FILES.index(f)-1], r-1)})


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

        return moves_dict




