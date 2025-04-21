
import numpy


def get_pawn_moves(int f, int r, int player, long[:,:] board, int ep_target=0):
    '''
    Parameters
    ----------
        f : int
            Index of the file the pawn occupies

        r : int
            Index of the rank the pawn occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

        ep_target : int
            Two-digit integer indicating a potential target for
            en passant capture

            Example: The value 63 indicates that square "f3" was
            passed through by white's pawn moving to "f4", thus
            any black pawns on "e4" or "g4" may capture en passant

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)
    '''

    cdef int idx_f_ep, idx_r_ep
    cdef int move_counter = 0
    cdef int moves[12][6]
    cdef int i, j
    for i in range(12):
        for j in range(6):
            moves[i][j] = 0


    ###  handling forward movement 1 square (for both white and black)
    if (board[r+player][f] == 0):

        ###  handling promotions (knight, bishop, rook, queen)
        if (r+player == 0) or (r+player == 7):
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r+1+player)
            moves[move_counter][5] = 2*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r+1+player)
            moves[move_counter][5] = 3*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r+1+player)
            moves[move_counter][5] = 4*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r+1+player)
            moves[move_counter][5] = 5*player
            move_counter += 1

        else:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r+1+player)
            move_counter += 1

    ###  handling forward movement 2 squares for white
    if (player == 1) and (r == 1) and (board[r+1][f] == 0) and (board[r+2][f] == 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f+1) + (r+3)
        move_counter += 1

    ###  handling forward movement 2 squares for black
    if (player == -1) and (r == 6) and (board[r-1][f] == 0) and (board[r-2][f] == 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f+1) + (r-1)
        move_counter += 1



    ###  handling captures to higher files (for both white and black)
    if (f < 7) and (player*board[r+player][f+1] < 0):

        ###  handling promotions (knight, bishop, rook, queen)
        if (r+player == 0) or (r+player == 7):
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+2) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 2*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+2) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 3*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+2) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 4*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+2) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 5*player
            move_counter += 1

        else:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+2) + (r+1+player)
            moves[move_counter][3] = 1
            move_counter += 1

    ###  handling captures to lower files (for both white and black)
    if (f > 0) and (player*board[r+player][f-1] < 0):

        ###  handling promotions (knight, bishop, rook, queen)
        if (r+player == 0) or (r+player == 7):
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+0) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 2*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+0) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 3*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+0) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 4*player
            move_counter += 1

            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+0) + (r+1+player)
            moves[move_counter][3] = 1
            moves[move_counter][5] = 5*player
            move_counter += 1

        else:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+0) + (r+1+player)
            moves[move_counter][3] = 1
            move_counter += 1


    ###  handling en passant captures
    if (ep_target > 0):

        ###  extracting rank and file indices for ep target
        idx_r_ep = ep_target % 10 - 1
        idx_f_ep = (ep_target - (idx_r_ep+1)) // 10 - 1

        ###  checking for pawns that can capture from higher files
        if (idx_f_ep < 7) and (board[idx_r_ep-player][idx_f_ep+1] == player):
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(idx_f_ep+2) + (idx_r_ep+1-player)
            moves[move_counter][2] = 10*(idx_f_ep+1) + (idx_r_ep+1)
            moves[move_counter][3] = 2
            move_counter += 1

        ###  checking for pawns that can capture from lower files
        if (idx_f_ep > 0) and (board[idx_r_ep-player][idx_f_ep-1] == player):
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(idx_f_ep+0) + (idx_r_ep+1-player)
            moves[move_counter][2] = 10*(idx_f_ep+1) + (idx_r_ep+1)
            moves[move_counter][3] = 2
            move_counter += 1

    return moves


def get_knight_moves(int f, int r, int player, long[:,:] board):
    '''
    Parameters
    ----------
        f : int
            Index of the file the knight occupies

        r : int
            Index of the rank the knight occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k


    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)
    '''

    cdef int move_counter = 0
    cdef int moves[8][6]
    cdef int i, j
    for i in range(8):
        for j in range(6):
            moves[i][j] = 0

    cdef int r_up1, r_up2, r_dn1, r_dn2
    cdef int f_up1, f_up2, f_dn1, f_dn2
    r_up1, r_up2 = r+1, r+2
    f_up1, f_up2 = f+1, f+2
    r_dn1, r_dn2 = r-1, r-2
    f_dn1, f_dn2 = f-1, f-2

    ###  checking 1:00 position
    if (f_up1 < 8) and (r_up2 < 8) and (player*board[r_up2][f_up1] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up1+1) + (r_up2+1)
        if player*board[r_up2][f_up1] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 2:00 position
    if (f_up2 < 8) and (r_up1 < 8) and (player*board[r_up1][f_up2] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up2+1) + (r_up1+1)
        if player*board[r_up1][f_up2] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 4:00 position
    if (f_up2 < 8) and (r_dn1 > -1) and (player*board[r_dn1][f_up2] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up2+1) + (r_dn1+1)
        if player*board[r_dn1][f_up2] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 5:00 position
    if (f_up1 < 8) and (r_dn2 > -1) and (player*board[r_dn2][f_up1] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up1+1) + (r_dn2+1)
        if player*board[r_dn2][f_up1] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 7:00 position
    if (f_dn1 > -1) and (r_dn2 > -1) and (player*board[r_dn2][f_dn1] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn1+1) + (r_dn2+1)
        if player*board[r_dn2][f_dn1] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 8:00 position
    if (f_dn2 > -1) and (r_dn1 > -1) and (player*board[r_dn1][f_dn2] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn2+1) + (r_dn1+1)
        if player*board[r_dn1][f_dn2] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 10:00 position
    if (f_dn2 > -1) and (r_up1 < 8) and (player*board[r_up1][f_dn2] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn2+1) + (r_up1+1)
        if player*board[r_up1][f_dn2] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking 11:00 position
    if (f_dn1 > -1) and (r_up2 < 8) and (player*board[r_up2][f_dn1] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn1+1) + (r_up2+1)
        if player*board[r_up2][f_dn1] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    return moves


def get_bishop_moves(int f, int r, int player, long[:,:] board):
    '''
    Parameters
    ----------
        f : int
            Index of the file the bishop occupies

        r : int
            Index of the rank the bishop occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)
    '''


    ###  defining variables
    ###  r_up, r_dn, f_up, f_dn for indices of ranks and files to check
    ###  blk_uprt, blk_uplf, blk_dnrt, blk_dnlf for bools to indicate if direction is blocked
    cdef int r_up, r_dn, f_up, f_dn, d
    cdef int blk_uprt, blk_uplf, blk_dnrt, blk_dnlf
    blk_uprt, blk_uplf, blk_dnrt, blk_dnlf = 0, 0, 0, 0

    cdef int move_counter = 0
    cdef int moves[13][6]
    cdef int i, j
    for i in range(13):
        for j in range(6):
            moves[i][j] = 0


    ###  identifying squares via ray casting
    for d in range(1, 8, 1):
        r_up, r_dn = r+d, r-d
        f_up, f_dn = f+d, f-d

        ###  checking direction of increasing ranks, increasing files
        if (r_up < 8) and (f_up < 8) and (blk_uprt == 0):

            ###  checking if square is vacant
            if board[r_up][f_up] == 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_up+1) + (r_up+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_up][f_up] < 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_up+1) + (r_up+1)
                moves[move_counter][3] = 1
                move_counter += 1
                blk_uprt = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_up][f_up] > 0:
                blk_uprt = 1


        ###  checking direction of increasing ranks, decreasing files
        if (r_up < 8) and (f_dn > -1) and (blk_uplf == 0):

            ###  checking if square is vacant
            if board[r_up][f_dn] == 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_dn+1) + (r_up+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_up][f_dn] < 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_dn+1) + (r_up+1)
                moves[move_counter][3] = 1
                move_counter += 1
                blk_uplf = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_up][f_dn] > 0:
                blk_uplf = 1


        ###  checking direction of decreasing ranks, increasing files
        if (r_dn > -1) and (f_up < 8) and (blk_dnrt == 0):

            ###  checking if square is vacant
            if board[r_dn][f_up] == 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_up+1) + (r_dn+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_dn][f_up] < 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_up+1) + (r_dn+1)
                moves[move_counter][3] = 1
                move_counter += 1
                blk_dnrt = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_dn][f_up] > 0:
                blk_dnrt = 1


        ###  checking direction of decreasing ranks, decreasing files
        if (r_dn > -1) and (f_dn > -1) and (blk_dnlf == 0):

            ###  checking if square is vacant
            if board[r_dn][f_dn] == 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_dn+1) + (r_dn+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_dn][f_dn] < 0:
                moves[move_counter][0] = player
                moves[move_counter][1] = 10*(f+1) + (r+1)
                moves[move_counter][2] = 10*(f_dn+1) + (r_dn+1)
                moves[move_counter][3] = 1
                move_counter += 1
                blk_dnlf = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_dn][f_dn] > 0:
                blk_dnlf = 1

    return moves


def get_rook_moves(int f, int r, int player, long[:,:] board):
    '''
    Parameters
    ----------
        f : int
            Index of the file the rook occupies

        r : int
            Index of the rank the rook occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)
    '''

    ###  defining variables
    cdef int f2, r2
    cdef int move_counter = 0
    cdef int moves[14][6]
    cdef int i, j
    for i in range(14):
        for j in range(6):
            moves[i][j] = 0

    ###  checking increasing files (i.e. to the right from white's view)
    for f2 in range(f+1, 8):

        ###  checking if square is vacant
        if board[r][f2] == 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f2+1) + (r+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r][f2] < 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f2+1) + (r+1)
            moves[move_counter][3] = 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking decreasing files (i.e. to the left from white's view)
    for f2 in range(f-1, -1, -1):

        ###  checking if square is vacant
        if board[r][f2] == 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f2+1) + (r+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r][f2] < 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f2+1) + (r+1)
            moves[move_counter][3] = 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking increasing ranks (i.e. upward from white's view)
    for r2 in range(r+1, 8):

        ###  checking if square is vacant
        if board[r2][f] == 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r2+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r2][f] < 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r2+1)
            moves[move_counter][3] = 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking decreasing ranks (i.e. downward from white's view)
    for r2 in range(r-1, -1, -1):

        ###  checking if square is vacant
        if board[r2][f] == 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r2+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r2][f] < 0:
            moves[move_counter][0] = player
            moves[move_counter][1] = 10*(f+1) + (r+1)
            moves[move_counter][2] = 10*(f+1) + (r2+1)
            moves[move_counter][3] = 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    return moves


def get_queen_moves(int f, int r, int player, long[:,:] board):
    '''
    Parameters
    ----------
        f : int
            Index of the file the queen occupies

        r : int
            Index of the rank the queen occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)
    '''

    cdef int move_counter = 0
    cdef int moves_rook[14][6]
    cdef int moves_bishop[13][6]
    cdef int moves[27][6]
    cdef int i, j
    for i in range(27):
        for j in range(6):
            moves[i][j] = 0

    ###  grabbing all bishop-like moves
    moves_bishop = get_bishop_moves(f, r, player, board)
    for i in range(13):
        ###  break if there are no more bishop moves
        if moves_bishop[i][0] == 0:
            break
        for j in range(6):
            moves[move_counter][j] = moves_bishop[i][j]
        move_counter += 1

    ###  grabbing all rook-like moves
    moves_rook = get_rook_moves(f, r, player, board)
    for i in range(14):
        ###  break if there are no more rook moves
        if moves_rook[i][0] == 0:
            break
        for j in range(6):
            moves[move_counter][j] = moves_rook[i][j]
        move_counter += 1

    return moves


def get_king_moves(int f, int r, int player, long[:,:] board):
    '''
    Parameters
    ----------
        f : int
            Index of the file the king occupies

        r : int
            Index of the rank the king occupies

        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)

            Note: 100 represents king-side castling and 1000
            represents queen-side castling
    '''

    cdef int move_counter = 0
    cdef int moves[8][6]
    cdef int i, j
    for i in range(8):
        for j in range(6):
            moves[i][j] = 0

    cdef int r_up, r_dn, f_up, f_dn
    r_up, f_up, r_dn, f_dn = r+1, f+1, r-1, f-1

    ###  checking straight up
    if (r_up < 8) and (player*board[r_up][f] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f+1) + (r_up+1)
        if player*board[r_up][f] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking straight down
    if (r_dn > -1) and (player*board[r_dn][f] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f+1) + (r_dn+1)
        if player*board[r_dn][f] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking straight left
    if (f_dn > -1) and (player*board[r][f_dn] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn+1) + (r+1)
        if player*board[r][f_dn] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking straight right
    if (f_up < 8) and (player*board[r][f_up] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up+1) + (r+1)
        if player*board[r][f_up] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking up-left diagonal
    if (r_up < 8) and (f_dn > -1) and (player*board[r_up][f_dn] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn+1) + (r_up+1)
        if player*board[r_up][f_dn] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking up-right diagonal
    if (r_up < 8) and (f_up < 8) and (player*board[r_up][f_up] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up+1) + (r_up+1)
        if player*board[r_up][f_up] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking down-right diagonal
    if (r_dn > -1) and (f_up < 8) and (player*board[r_dn][f_up] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_up+1) + (r_dn+1)
        if player*board[r_dn][f_up] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking down-left diagonal
    if (r_dn > -1) and (f_dn > -1) and (player*board[r_dn][f_dn] <= 0):
        moves[move_counter][0] = player
        moves[move_counter][1] = 10*(f+1) + (r+1)
        moves[move_counter][2] = 10*(f_dn+1) + (r_dn+1)
        if player*board[r_dn][f_dn] < 0:
            moves[move_counter][3] = 1
        move_counter += 1

    ###  checking for king-side castling
    if (player == 1) \
       and (board[0][4] == 6) \
       and (board[0][5] == 0) \
       and (board[0][6] == 0) \
       and (board[0][7] == 4):
        moves[move_counter][0] = player
        moves[move_counter][1] = 51
        moves[move_counter][2] = 71
        moves[move_counter][4] = 100
        move_counter += 1
    elif (player == -1) \
         and (board[7][4] == 6) \
         and (board[7][5] == 0) \
         and (board[7][6] == 0) \
         and (board[7][7] == 4):
        moves[move_counter][0] = player
        moves[move_counter][1] = 58
        moves[move_counter][2] = 78
        moves[move_counter][4] = 100
        move_counter += 1

    ###  checking for queen-side castling
    if (player == 1) \
       and (board[0][4] == 6) \
       and (board[0][3] == 0) \
       and (board[0][2] == 0) \
       and (board[0][1] == 0) \
       and (board[0][0] == 4):
        moves[move_counter][0] = player
        moves[move_counter][1] = 51
        moves[move_counter][2] = 31
        moves[move_counter][4] = 1000
        move_counter += 1
    elif (player == -1) \
         and (board[7][4] == 6) \
         and (board[7][3] == 0) \
         and (board[7][2] == 0) \
         and (board[7][1] == 0) \
         and (board[7][0] == 4):
        moves[move_counter][0] = player
        moves[move_counter][1] = 58
        moves[move_counter][2] = 38
        moves[move_counter][4] = 1000
        move_counter += 1

    return moves


def get_legal_moves(int player_on_move, long[:,:] board, int ep_target, bint get_eval=0):
    '''
    Parameters
    ----------
        player_on_move : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

        ep_target : int
            Two-digit integer indicating a potential target for
            en passant capture

            Example: The value 63 indicates that square "f3" was
            passed through by white's pawn moving to "f4", thus
            any black pawns on "e4" or "g4" may capture en passant

        get_eval : bint
            Calculate evaluation of position

    Returns
    -------
        moves : 2d-array of ints
            An array of legal moves where the columns represent:
            column 1 : ID of piece is moving
            column 2 : Start square of piece (e.g. 34 for "c4")
            column 3 : End square of piece (e.g. 67 for "f7")
            column 4 : Capture flag (0=False, 1=True, 2=en passant)
            column 5 : Castling flag (0=False, 100=kingside, 1000=queenside)
            column 6 : Promotion flag (ID of piece being promoted to)

            Note: 100 represents king-side castling and 1000
            represents queen-side castling
    '''

    cdef int r, f, move_counter
    move_counter = 0
    cdef int[12][6] pawn_moves
    cdef int[8][6] knight_moves
    cdef int[13][6] bishop_moves
    cdef int[14][6] rook_moves
    cdef int[27][6] queen_moves
    cdef int[8][6] king_moves
    cdef int[200][6] all_moves  # NOTE: 200 is a rough guess of the max number of candidate moves
    cdef int i, j
    for i in range(200):
        for j in range(6):
            all_moves[i][j] = 0

    for r in range(8):
        for f in range(8):

            if board[r][f] == 1*player_on_move:
                pawn_moves = get_pawn_moves(f, r, player_on_move, board, ep_target)
                for i in range(12):
                    ###  truncate if there are no more moves
                    if pawn_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = pawn_moves[i][j]
                    move_counter += 1

            if board[r][f] == 2*player_on_move:
                knight_moves = get_knight_moves(f, r, player_on_move, board)
                for i in range(8):
                    ###  truncate if there are no more moves
                    if knight_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = knight_moves[i][j]
                    move_counter += 1

            if board[r][f] == 3*player_on_move:
                bishop_moves = get_bishop_moves(f, r, player_on_move, board)
                for i in range(13):
                    ###  truncate if there are no more moves
                    if bishop_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = bishop_moves[i][j]
                    move_counter += 1

            if board[r][f] == 4*player_on_move:
                rook_moves = get_rook_moves(f, r, player_on_move, board)
                for i in range(14):
                    ###  truncate if there are no more moves
                    if rook_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = rook_moves[i][j]
                    move_counter += 1

            if board[r][f] == 5*player_on_move:
                queen_moves = get_queen_moves(f, r, player_on_move, board)
                for i in range(27):
                    ###  truncate if there are no more moves
                    if queen_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = queen_moves[i][j]
                    move_counter += 1

            if board[r][f] == 6*player_on_move:
                king_moves = get_king_moves(f, r, player_on_move, board)
                for i in range(8):
                    ###  truncate if there are no more moves
                    if king_moves[i][0] == 0:
                        break
                    ###  else add all values from row
                    for j in range(6):
                        all_moves[move_counter][j] = king_moves[i][j]
                    move_counter += 1

    ###  filtering out illegal moves
    cdef int[200][6] legal_moves  # NOTE: 200 is a rough guess of the max number of candidate moves
    for i in range(200):
        for j in range(6):
            legal_moves[i][j] = 0

    legal_moves = filter_illegal_moves(numpy.array(all_moves), board)

    return legal_moves


def get_attacked_squares(int player, long[:,:] board):
    '''
    Parameters
    ----------
        player : int
            Either 1 for white or -1 for black

        board : 2d-array of ints
            Full layout of chessboard
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k

    Returns
    -------
        n_attacks_per_square : 2d-array of ints
            An 8x8 array of representing the number of times each
            given square of the board is attacked by `player` pieces
    '''

    cdef int r, f, r2, f2, d
    cdef int[8][8] n_attacks_per_square
    cdef int i, j
    for i in range(8):
        for j in range(8):
            n_attacks_per_square[i][j] = 0


    ###  iterating over squares
    for r in range(8):
        for f in range(8):

            ###  handling pawn attacks
            if (board[r][f] == player) and (f < 7):
                n_attacks_per_square[r+player][f+1] += 1
            if (board[r][f] == player) and (f > 0):
                n_attacks_per_square[r+player][f-1] += 1


            ###  handling attacks by knights
            if (board[r][f] == 2*player):
                if (2 <= r) and (1 <= f):
                    n_attacks_per_square[r-2][f-1] += 1
                if (1 <= r) and (2 <= f):
                    n_attacks_per_square[r-1][f-2] += 1
                if (r <= 6) and (2 <= f):
                    n_attacks_per_square[r+1][f-2] += 1
                if (r <= 5) and (1 <= f):
                    n_attacks_per_square[r+2][f-1] += 1
                if (r <= 5) and (f <= 6):
                    n_attacks_per_square[r+2][f+1] += 1
                if (r <= 6) and (f <= 5):
                    n_attacks_per_square[r+1][f+2] += 1
                if (1 <= r) and (f <= 5):
                    n_attacks_per_square[r-1][f+2] += 1
                if (2 <= r) and (f <= 6):
                    n_attacks_per_square[r-2][f+1] += 1


            ###  handling attacks by rooks (and queen)
            if (board[r][f] == 4*player) or (board[r][f] == 5*player):
                ###  right files
                for f2 in range(f+1, 8):
                    n_attacks_per_square[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  down ranks
                for r2 in range(r-1, -1, -1):
                    n_attacks_per_square[r2][f] += 1
                    if board[r2][f] != 0:
                        break
                ###  left files
                for f2 in range(f-1, -1, -1):
                    n_attacks_per_square[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  up ranks
                for r2 in range(r+1, 8):
                    n_attacks_per_square[r2][f] += 1
                    if board[r2][f] != 0:
                        break


            ###  handling attacks by bishops (and queen)
            if (board[r][f] == 3) or (board[r][f] == 5):
                ###  up-right diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f+d < 8):
                        n_attacks_per_square[r+d][f+d] += 1
                        if board[r+d][f+d] != 0:
                            break
                ###  down-right diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f+d < 8):
                        n_attacks_per_square[r-d][f+d] += 1
                        if board[r-d][f+d] != 0:
                            break
                ###  down-left diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f-d > -1):
                        n_attacks_per_square[r-d][f-d] += 1
                        if board[r-d][f-d] != 0:
                            break
                ###  up-left diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f-d > -1):
                        n_attacks_per_square[r+d][f-d] += 1
                        if board[r+d][f-d] != 0:
                            break


            ###  handling attacks by kings
            if (board[r][f] == 6*player):
                for r2 in range(r-1, r+2):
                    for f2 in range(f-1, f+2):
                        if (r2 == r) and (f2 == f):
                            pass
                        elif (r2 >= 0) and (r2 <= 7) and (f2 >= 0) and (f2 <= 7):
                            n_attacks_per_square[r2][f2] += 1

    return n_attacks_per_square


def move_piece(long[:] move, long[:,:] board):
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

    Returns
    -------
        new_board : 2d-array of ints
            Layout of chessboard after piece is moved
            0 = vacant square
            1, 2, 3, 4, 5, 6 = P, N, B, R, Q, K
            -1, -2, -3, -4, -5, -6 = p, n, b, r, q, k
    '''

    cdef int piece
    cdef int f_start, r_start, f_end, r_end
    cdef int[8][8] new_board
    cdef int i, j
    for i in range(8):
        for j in range(8):
            new_board[i][j] = board[i][j]

    ###  parsing start / end squares and piece being moved
    r_start = move[1] % 10 - 1
    f_start = (move[1] - r_start) // 10 - 1
    piece = new_board[r_start][f_start]
    r_end = move[2] % 10 - 1
    f_end = (move[2] - r_end) // 10 - 1


    ###  if promotion
    if move[5] != 0:
        new_board[r_end][f_end] = move[5]
        new_board[r_start][f_start] = 0

    ###  if en passaant
    elif move[3] == 2:
        new_board[r_end][f_end] = piece
        new_board[r_start][f_start] = 0
        new_board[r_end][f_start] = 0

    ###  if castling
    elif move[4] == 100:
        new_board[r_start][4] = 0
        new_board[r_start][5] = 4
        new_board[r_start][6] = 6
        new_board[r_start][7] = 0
    elif move[4] == 1000:
        new_board[r_start][4] = 0
        new_board[r_start][3] = 4
        new_board[r_start][2] = 6
        new_board[r_start][0] = 0

    ###  else ordinary move
    else:
        new_board[r_end][f_end] = piece
        new_board[r_start][f_start] = 0

    return new_board


def filter_illegal_moves(long[:,:] candidate_moves, long[:,:] board):


    cdef int player = candidate_moves[0][0]
    cdef int opponent = -player
    if opponent == 0:
        print('No candidate moves found... is it checkmate?')
        return 0

    ###  instantiating arrays indicating attacked squares by opponent
    cdef int[8][8] n_attacks_by_opp_current_board
    cdef int[8][8] n_attacks_by_opp_new_board
    n_attacks_by_opp_current_board = get_attacked_squares(opponent, board)

    ###  locating square occupied by king
    cdef int r_king, f_king
    cdef int i, j
    for i in range(8):
        for j in range(8):
            if board[i][j] == 6*player:
                r_king, f_king = i, j
                break

    ###  iterating over candidate moves
    cdef int move_counter = 0
    cdef int f_start, r_start, f_end, r_end
    cdef int[200][6] legal_moves
    for i in range(200):
        for j in range(6):
            legal_moves[i][j] = 0

    cdef int[8][8] new_board
    cdef long[:] move
    for move in candidate_moves:

        ###  truncate if no more moves
        if move[0] == 0:
            break

        ###  parsing start / end squares and piece being moved
        r_start = move[1] % 10 - 1
        f_start = (move[1] - r_start) // 10 - 1
        r_end = move[2] % 10 - 1
        f_end = (move[2] - r_start) // 10 - 1

        ###  checking for king-side castling
        if (move[4] == 100) and \
           (n_attacks_by_opp_current_board[r_end][4] + \
            n_attacks_by_opp_current_board[r_end][5] + \
            n_attacks_by_opp_current_board[r_end][6] == 0):
            for j in range(6):
                legal_moves[move_counter][j] = move[j]
            move_counter += 1

        ###  checking for queen-side castling
        elif (move[4] == 1000) and \
             (n_attacks_by_opp_current_board[r_end][4] + \
              n_attacks_by_opp_current_board[r_end][3] + \
              n_attacks_by_opp_current_board[r_end][2] == 0):
            for j in range(6):
                legal_moves[move_counter][j] = move[j]
            move_counter += 1

        ###  else applying move to board, counting squares attacked by opponent
        else:
            new_board = move_piece(move, board)
            n_attacks_by_opp_new_board = get_attacked_squares(opponent, numpy.array(new_board))

            ###  if piece being moved is not the king, check the king's current location
            if board[r_start][f_start] != 6*player:
                if n_attacks_by_opp_new_board[r_king][f_king] == 0:
                    for j in range(6):
                        legal_moves[move_counter][j] = move[j]
                    move_counter += 1

            ###  else check where the king is moving to
            else:
                if n_attacks_by_opp_new_board[r_end][f_end] == 0:
                    for j in range(6):
                        legal_moves[move_counter][j] = move[j]
                    move_counter += 1

    return legal_moves





'''
if True:
    niter = 1
    board = asdf.chess_boards[-1]
    int_board = asdf.get_int_chessboard(asdf.chess_boards[-1])
    ###
    t0 = time.time()
    for i in range(niter):
        x = asdf.get_attacked_squares(board)
    t1 = time.time()
    print('Ran Python version %i iterations took %.4f seconds' % (niter, t1-t0))
    print('')
    t2 = time.time()
    for j in range(niter):
        y = asdf.get_attacked_squares_cy(int_board)
    t3 = time.time()
    print('Ran Cython version %i iterations took %.4f seconds' % (niter, t3-t2))
    ###
    print('\nWHITE - python')
    for r in range(8, 0, -1):
        s = ''
        for f in FILES:
            s += ' %i ' % x[0][r][f]
        print(s)
    print('\nBLACK - python')
    for r in range(8, 0, -1):
        s = ''
        for f in FILES:
            s += ' %i ' % x[1][r][f]
        print(s)

    print('\nWHITE - cython')
    for r in range(7, -1, -1):
        s = ''
        for f in range(7, -1, -1):
            s += ' %i ' % y[0][r][f]
        print(s)
    print('\nBLACK - cython')
    for r in range(7, -1, -1):
        s = ''
        for f in range(7, -1, -1):
            s += ' %i ' % y[1][r][f]
        print(s)
'''
