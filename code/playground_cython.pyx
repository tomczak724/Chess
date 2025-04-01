


def get_pawn_moves(int f, int r, int player, long[:,:] board, int ep_target=0):
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

        ep_target : int
            Two-digit integer indicating a potential target for
            en passant capture

            Example: The value 63 indicates that square "f3" was
            passed through by white's pawn moving to "f4", thus
            any black pawns on "e4" or "g4" may capture en passant

    Returns
    -------
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''

    cdef int idx_f_ep, idx_r_ep
    cdef int move_counter = 0
    cdef int moves[4]
    cdef int i
    for i in range(4):
        moves[i] = 0


    ###  handling forward movement 1 square (for both white and black)
    if (board[r+player][f] == 0):
        moves[move_counter] = 100*(f+1) + 10*(r+2*player)
        move_counter += 1

    ###  handling forward movement 2 squares for white
    if (player == 1) and (board[r+1][f] == 0) and (board[r+2][f] == 0):
        moves[move_counter] = 100*(f+1) + 10*(r+3)
        move_counter += 1

    ###  handling forward movement 2 squares for black
    if (player == -1) and (board[r-1][f] == 0) and (board[r-2][f] == 0):
        moves[move_counter] = 100*(f+1) + 10*(r-3)
        move_counter += 1


    ###  handling captures to higher files (for both white and black)
    if (f < 7) and (player*board[r+player][f+1] < 0):
        moves[move_counter] = 100*(f+2) + 10*(r+2*player) + 1
        move_counter += 1

    ###  handling captures to lower files (for both white and black)
    if (f > 0) and (player*board[r+player][f-1] < 0):
        moves[move_counter] = 100*(f+0) + 10*(r+2*player) + 1
        move_counter += 1


    ###  handling en passant captures
    if (ep_target > 0):
        ###  extracting rank and file indices for ep target
        idx_r_ep = ep_target % 10 - 1
        idx_f_ep = (ep_target - (idx_r_ep+1)) // 10 - 1

        ###  checking for pawns that can capture from higher files
        if (idx_f_ep < 7) and (board[idx_r_ep-player][idx_f_ep+1] == player):
            moves[move_counter] = 100*(idx_f_ep+1) + 10*(idx_r_ep+1) + 1
            move_counter += 1

        ###  checking for pawns that can capture from lower files
        if (idx_f_ep > 0) and (board[idx_r_ep-player][idx_f_ep-1] == player):
            moves[move_counter] = 100*(idx_f_ep+1) + 10*(idx_r_ep+1) + 1
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
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''

    cdef int move_counter = 0
    cdef int moves[8]
    cdef int i
    for i in range(8):
        moves[i] = 0

    cdef int r_up1, r_up2, r_dn1, r_dn2
    cdef int f_up1, f_up2, f_dn1, f_dn2
    r_up1, r_up2 = r+1, r+2
    f_up1, f_up2 = f+1, f+2
    r_dn1, r_dn2 = r-1, r-2
    f_dn1, f_dn2 = f-1, f-2

    ###  checking 1:00 position
    if (f_up1 < 8) and (r_up2 < 8) and (player*board[r_up2][f_up1] <= 0):
        moves[move_counter] = 100*(f_up1+1) + 10*(r_up2+1)
        if player*board[r_up2][f_up1] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 2:00 position
    if (f_up2 < 8) and (r_up1 < 8) and (player*board[r_up1][f_up2] <= 0):
        moves[move_counter] = 100*(f_up2+1) + 10*(r_up1+1)
        if player*board[r_up1][f_up2] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 4:00 position
    if (f_up2 < 8) and (r_dn1 > -1) and (player*board[r_dn1][f_up2] <= 0):
        moves[move_counter] = 100*(f_up2+1) + 10*(r_dn1+1)
        if player*board[r_dn1][f_up2] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 5:00 position
    if (f_up1 < 8) and (r_dn2 > -1) and (player*board[r_dn2][f_up1] <= 0):
        moves[move_counter] = 100*(f_up1+1) + 10*(r_dn2+1)
        if player*board[r_dn2][f_up1] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 7:00 position
    if (f_dn1 > -1) and (r_dn2 > -1) and (player*board[r_dn2][f_dn1] <= 0):
        moves[move_counter] = 100*(f_dn1+1) + 10*(r_dn2+1)
        if player*board[r_dn2][f_dn1] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 8:00 position
    if (f_dn2 > -1) and (r_dn1 > -1) and (player*board[r_dn1][f_dn2] <= 0):
        moves[move_counter] = 100*(f_dn2+1) + 10*(r_dn1+1)
        if player*board[r_dn1][f_dn2] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 10:00 position
    if (f_dn2 > -1) and (r_up1 < 8) and (player*board[r_up1][f_dn2] <= 0):
        moves[move_counter] = 100*(f_dn2+1) + 10*(r_up1+1)
        if player*board[r_up1][f_dn2] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking 11:00 position
    if (f_dn1 > -1) and (r_up2 < 8) and (player*board[r_up2][f_dn1] <= 0):
        moves[move_counter] = 100*(f_dn1+1) + 10*(r_up2+1)
        if player*board[r_up2][f_dn1] < 0:
            moves[move_counter] += 1
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
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''


    ###  defining variables
    ###  r_up, r_dn, f_up, f_dn for indices of ranks and files to check
    ###  blk_uprt, blk_uplf, blk_dnrt, blk_dnlf for bools to indicate if direction is blocked
    cdef int r_up, r_dn, f_up, f_dn, d, move_counter
    cdef int blk_uprt, blk_uplf, blk_dnrt, blk_dnlf
    cdef int moves[13]
    cdef int i
    for i in range(13):
        moves[i] = 0

    move_counter = 0
    blk_uprt, blk_uplf, blk_dnrt, blk_dnlf = 0, 0, 0, 0


    ###  identifying squares via ray casting
    for d in range(1, 8, 1):
        r_up, r_dn = r+d, r-d
        f_up, f_dn = f+d, f-d

        ###  checking direction of increasing ranks, increasing files
        if (r_up < 8) and (f_up < 8) and (blk_uprt == 0):

            ###  checking if square is vacant
            if board[r_up][f_up] == 0:
                moves[move_counter] = 100*(f_up+1) + 10*(r_up+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_up][f_up] < 0:
                moves[move_counter] = 100*(f_up+1) + 10*(r_up+1) + 1
                move_counter += 1
                blk_uprt = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_up][f_up] > 0:
                blk_uprt = 1


        ###  checking direction of increasing ranks, decreasing files
        if (r_up < 8) and (f_dn > -1) and (blk_uplf == 0):

            ###  checking if square is vacant
            if board[r_up][f_dn] == 0:
                moves[move_counter] = 100*(f_dn+1) + 10*(r_up+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_up][f_dn] < 0:
                moves[move_counter] = 100*(f_dn+1) + 10*(r_up+1) + 1
                move_counter += 1
                blk_uplf = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_up][f_dn] > 0:
                blk_uplf = 1


        ###  checking direction of decreasing ranks, increasing files
        if (r_dn > -1) and (f_up < 8) and (blk_dnrt == 0):

            ###  checking if square is vacant
            if board[r_dn][f_up] == 0:
                moves[move_counter] = 100*(f_up+1) + 10*(r_dn+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_dn][f_up] < 0:
                moves[move_counter] = 100*(f_up+1) + 10*(r_dn+1) + 1
                move_counter += 1
                blk_dnrt = 1

            ###  else check if it is occupied by player's piece
            elif player*board[r_dn][f_up] > 0:
                blk_dnrt = 1


        ###  checking direction of decreasing ranks, decreasing files
        if (r_dn > -1) and (f_dn > -1) and (blk_dnlf == 0):

            ###  checking if square is vacant
            if board[r_dn][f_dn] == 0:
                moves[move_counter] = 100*(f_dn+1) + 10*(r_dn+1)
                move_counter += 1

            ###  else check if it is occupied by opponent piece
            elif player*board[r_dn][f_dn] < 0:
                moves[move_counter] = 100*(f_dn+1) + 10*(r_dn+1) + 1
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
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''

    ###  defining variables
    cdef int f2, r2, move_counter
    move_counter = 0
    cdef int moves[14]
    cdef int i
    for i in range(14):
        moves[i] = 0

    ###  checking increasing files (i.e. to the right from white's view)
    for f2 in range(f+1, 8):

        ###  checking if square is vacant
        if board[r][f2] == 0:
            moves[move_counter] = 100*(f2+1) + 10*(r+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r][f2] < 0:
            moves[move_counter] = 100*(f2+1) + 10*(r+1) + 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking decreasing files (i.e. to the left from white's view)
    for f2 in range(f-1, -1, -1):

        ###  checking if square is vacant
        if board[r][f2] == 0:
            moves[move_counter] = 100*(f2+1) + 10*(r+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r][f2] < 0:
            moves[move_counter] = 100*(f2+1) + 10*(r+1) + 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking increasing ranks (i.e. upward from white's view)
    for r2 in range(r+1, 8):

        ###  checking if square is vacant
        if board[r2][f] == 0:
            moves[move_counter] = 100*(f+1) + 10*(r2+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r2][f] < 0:
            moves[move_counter] = 100*(f+1) + 10*(r2+1) + 1
            move_counter += 1
            break

        ###  else square must be occupied by player's piece
        else:
            break

    ###  checking decreasing ranks (i.e. downward from white's view)
    for r2 in range(r-1, -1, -1):

        ###  checking if square is vacant
        if board[r2][f] == 0:
            moves[move_counter] = 100*(f+1) + 10*(r2+1)
            move_counter += 1

        ###  else check if it is occupied by opponent piece
        elif player*board[r2][f] < 0:
            moves[move_counter] = 100*(f+1) + 10*(r2+1) + 1
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
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''

    cdef int m
    cdef int move_counter = 0
    cdef int moves[27]
    cdef int i
    for i in range(27):
        moves[i] = 0

    ###  grabbing all bishop-like moves
    for m in get_bishop_moves(f, r, player, board):
        if m > 0:
            moves[move_counter] = m
            move_counter += 1

    ###  grabbing all rook-like moves
    for m in get_rook_moves(f, r, player, board):
        if m > 0:
            moves[move_counter] = m
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
        moves : array
            An array of legal moves. Each element of the array
            will be a three-digit number where the first element
            represents the file, the second element represents 
            the rank, and the third element is a binary digit 
            representing if the move involves a capture.

            Note: 100 represents king-side castling and 1000
            represents queen-side castling

            Example: The value 351 represents a piece moving 
            to "c5" where it captures an opponent's piece
    '''

    cdef int move_counter = 0
    cdef int moves[8]
    cdef int i
    for i in range(8):
        moves[i] = 0

    cdef int r_up, r_dn, f_up, f_dn
    r_up = r+1
    f_up = f+1
    r_dn = r-1
    f_dn = f-1

    ###  checking straight up
    if (r_up < 8) and (player*board[r_up][f] <= 0):
        moves[move_counter] = 100*(f+1) + 10*(r_up+1)
        if player*board[r_up][f] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking straight down
    if (r_dn > -1) and (player*board[r_dn][f] <= 0):
        moves[move_counter] = 100*(f+1) + 10*(r_dn+1)
        if player*board[r_dn][f] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking straight left
    if (f_dn > -1) and (player*board[r][f_dn] <= 0):
        moves[move_counter] = 100*(f_dn+1) + 10*(r+1)
        if player*board[r][f_dn] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking straight right
    if (f_up < 8) and (player*board[r][f_up] <= 0):
        moves[move_counter] = 100*(f_up+1) + 10*(r+1)
        if player*board[r][f_up] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking up-left diagonal
    if (r_up < 8) and (f_dn > -1) and (player*board[r_up][f_dn] <= 0):
        moves[move_counter] = 100*(f_dn+1) + 10*(r_up+1)
        if player*board[r_up][f_dn] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking up-right diagonal
    if (r_up < 8) and (f_up < 8) and (player*board[r_up][f_up] <= 0):
        moves[move_counter] = 100*(f_up+1) + 10*(r_up+1)
        if player*board[r_up][f_up] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking down-right diagonal
    if (r_dn > -1) and (f_up < 8) and (player*board[r_dn][f_up] <= 0):
        moves[move_counter] = 100*(f_up+1) + 10*(r_dn+1)
        if player*board[r_dn][f_up] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking down-left diagonal
    if (r_dn > -1) and (f_dn > -1) and (player*board[r_dn][f_dn] <= 0):
        moves[move_counter] = 100*(f_dn+1) + 10*(r_dn+1)
        if player*board[r_dn][f_dn] < 0:
            moves[move_counter] += 1
        move_counter += 1

    ###  checking for king-side castling
    if (player == 1) \
       and (board[0][4] == 6) \
       and (board[0][5] == 0) \
       and (board[0][6] == 0) \
       and (board[0][7] == 4):
        moves[move_counter] = 100
        move_counter += 1
    elif (player == -1) \
         and (board[7][4] == 6) \
         and (board[7][5] == 0) \
         and (board[7][6] == 0) \
         and (board[7][7] == 4):
        moves[move_counter] = 100
        move_counter += 1

    ###  checking for queen-side castling
    if (player == 1) \
       and (board[0][4] == 6) \
       and (board[0][3] == 0) \
       and (board[0][2] == 0) \
       and (board[0][1] == 0) \
       and (board[0][0] == 4):
        moves[move_counter] = 1000
        move_counter += 1
    elif (player == -1) \
         and (board[7][4] == 6) \
         and (board[7][3] == 0) \
         and (board[7][2] == 0) \
         and (board[7][1] == 0) \
         and (board[7][0] == 4):
        moves[move_counter] = 1000
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
        all_moves : array
            An array of legal moves. Each element of the array
            will be a four-digit number where the first element
            represents the piece, the second element represents 
            the file, the third element represents the rank, and 
            the fourth element is a binary digit representing if 
            the move involves a capture

            Example: The value 4351 represents a rook moving 
            to "c5" where it captures an opponent's piece
    '''

    cdef int m, r, f, move_counter
    move_counter = 0
    cdef int[4] pawn_moves
    cdef int[8] knight_moves
    cdef int[13] bishop_moves
    cdef int[14] rook_moves
    cdef int[27] queen_moves
    cdef int[8] king_moves
    cdef int[137] all_moves
    cdef int i
    for i in range(137):
        all_moves[i] = 0

    ###  handling moves for white
    if player_on_move == 1:
        for r in range(8):
            for f in range(8):

                if board[r][f] == 1:
                    pawn_moves = get_pawn_moves(f, r, player_on_move, board, ep_target)
                    for m in pawn_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 1000 + m
                        move_counter += 1
                if board[r][f] == 2:
                    knight_moves = get_knight_moves(f, r, player_on_move, board)
                    for m in knight_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 2000 + m
                        move_counter += 1
                if board[r][f] == 3:
                    bishop_moves = get_bishop_moves(f, r, player_on_move, board)
                    for m in bishop_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 3000 + m
                        move_counter += 1
                if board[r][f] == 4:
                    rook_moves = get_rook_moves(f, r, player_on_move, board)
                    for m in rook_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 4000 + m
                        move_counter += 1
                if board[r][f] == 5:
                    queen_moves = get_queen_moves(f, r, player_on_move, board)
                    for m in queen_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 5000 + m
                        move_counter += 1
                if board[r][f] == 6:
                    king_moves = get_king_moves(f, r, player_on_move, board)
                    for m in king_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = 6000 + m
                        move_counter += 1


    ###  handling moves for black
    if player_on_move == -1:
        for r in range(8):
            for f in range(8):

                if board[r][f] == -1:
                    pawn_moves = get_pawn_moves(f, r, player_on_move, board, ep_target)
                    for m in pawn_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -1000 - m
                        move_counter += 1
                if board[r][f] == -2:
                    knight_moves = get_knight_moves(f, r, player_on_move, board)
                    for m in knight_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -2000 - m
                        move_counter += 1
                if board[r][f] == -3:
                    bishop_moves = get_bishop_moves(f, r, player_on_move, board)
                    for m in bishop_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -3000 - m
                        move_counter += 1
                if board[r][f] == -4:
                    rook_moves = get_rook_moves(f, r, player_on_move, board)
                    for m in rook_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -4000 - m
                        move_counter += 1
                if board[r][f] == -5:
                    queen_moves = get_queen_moves(f, r, player_on_move, board)
                    for m in queen_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -5000 - m
                        move_counter += 1
                if board[r][f] == -6:
                    king_moves = get_king_moves(f, r, player_on_move, board)
                    for m in king_moves:
                        if m == 0:
                            break
                        all_moves[move_counter] = -6000 - m
                        move_counter += 1

    return all_moves


def get_attacked_squares(long[:,:] board):

    cdef int r, f, r2, f2, d
    cdef int[8][8] n_attacks_by_white
    cdef int[8][8] n_attacks_by_black

    cdef int i, j
    for i in range(8):
        for j in range(8):
            n_attacks_by_white[i][j] = 0
            n_attacks_by_black[i][j] = 0


    ###  iterating over squares
    for r in range(8):
        for f in range(8):
            
            ###  handling pawn attacks
            if (board[r][f] == 1) and (f < 7):
                n_attacks_by_white[r+1][f+1] += 1
            if (board[r][f] == 1) and (f > 0):
                n_attacks_by_white[r+1][f-1] += 1
            if (board[r][f] == -1) and (f < 7):
                n_attacks_by_black[r-1][f+1] += 1
            if (board[r][f] == -1) and (f > 0):
                n_attacks_by_black[r-1][f-1] += 1


            ###  handling attacks by knights
            if (board[r][f] == 2):
                if (2 <= r) and (1 <= f):
                    n_attacks_by_white[r-2][f-1] += 1
                if (1 <= r) and (2 <= f):
                    n_attacks_by_white[r-1][f-2] += 1
                if (r <= 6) and (2 <= f):
                    n_attacks_by_white[r+1][f-2] += 1
                if (r <= 5) and (1 <= f):
                    n_attacks_by_white[r+2][f-1] += 1
                if (r <= 5) and (f <= 6):
                    n_attacks_by_white[r+2][f+1] += 1
                if (r <= 6) and (f <= 5):
                    n_attacks_by_white[r+1][f+2] += 1
                if (1 <= r) and (f <= 5):
                    n_attacks_by_white[r-1][f+2] += 1
                if (2 <= r) and (f <= 6):
                    n_attacks_by_white[r-2][f+1] += 1
            elif (board[r][f] == -2):
                if (2 <= r) and (1 <= f):
                    n_attacks_by_black[r-2][f-1] += 1
                if (1 <= r) and (2 <= f):
                    n_attacks_by_black[r-1][f-2] += 1
                if (r <= 6) and (2 <= f):
                    n_attacks_by_black[r+1][f-2] += 1
                if (r <= 5) and (1 <= f):
                    n_attacks_by_black[r+2][f-1] += 1
                if (r <= 5) and (f <= 6):
                    n_attacks_by_black[r+2][f+1] += 1
                if (r <= 6) and (f <= 5):
                    n_attacks_by_black[r+1][f+2] += 1
                if (1 <= r) and (f <= 5):
                    n_attacks_by_black[r-1][f+2] += 1
                if (2 <= r) and (f <= 6):
                    n_attacks_by_black[r-2][f+1] += 1


            ###  handling attacks by rooks (and queen)
            if (board[r][f] == 4) or (board[r][f] == 5):
                ###  right files
                for f2 in range(f+1, 8):
                    n_attacks_by_white[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  down ranks
                for r2 in range(r-1, -1, -1):
                    n_attacks_by_white[r2][f] += 1
                    if board[r2][f] != 0:
                        break
                ###  left files
                for f2 in range(f-1, -1, -1):
                    n_attacks_by_white[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  up ranks
                for r2 in range(r+1, 8):
                    n_attacks_by_white[r2][f] += 1
                    if board[r2][f] != 0:
                        break
            if (board[r][f] == -4) or (board[r][f] == -5):
                ###  right files
                for f2 in range(f+1, 8):
                    n_attacks_by_black[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  down ranks
                for r2 in range(r-1, -1, -1):
                    n_attacks_by_black[r2][f] += 1
                    if board[r2][f] != 0:
                        break
                ###  left files
                for f2 in range(f-1, -1, -1):
                    n_attacks_by_black[r][f2] += 1
                    if board[r][f2] != 0:
                        break
                ###  up ranks
                for r2 in range(r+1, 8):
                    n_attacks_by_black[r2][f] += 1
                    if board[r2][f] != 0:
                        break


            ###  handling attacks by bishops (and queen)
            if (board[r][f] == 3) or (board[r][f] == 5):
                ###  up-right diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f+d < 8):
                        n_attacks_by_white[r+d][f+d] += 1
                        if board[r+d][f+d] != 0:
                            break
                ###  down-right diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f+d < 8):
                        n_attacks_by_white[r-d][f+d] += 1
                        if board[r-d][f+d] != 0:
                            break
                ###  down-left diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f-d > -1):
                        n_attacks_by_white[r-d][f-d] += 1
                        if board[r-d][f-d] != 0:
                            break
                ###  up-left diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f-d > -1):
                        n_attacks_by_white[r+d][f-d] += 1
                        if board[r+d][f-d] != 0:
                            break
            if (board[r][f] == -3) or (board[r][f] == -5):
                ###  up-right diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f+d < 8):
                        n_attacks_by_black[r+d][f+d] += 1
                        if board[r+d][f+d] != 0:
                            break
                ###  down-right diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f+d < 8):
                        n_attacks_by_black[r-d][f+d] += 1
                        if board[r-d][f+d] != 0:
                            break
                ###  down-left diagonal
                for d in range(1, 8, 1):
                    if (r-d > -1) and (f-d > -1):
                        n_attacks_by_black[r-d][f-d] += 1
                        if board[r-d][f-d] != 0:
                            break
                ###  up-left diagonal
                for d in range(1, 8, 1):
                    if (r+d < 8) and (f-d > -1):
                        n_attacks_by_black[r+d][f-d] += 1
                        if board[r+d][f-d] != 0:
                            break


            ###  handling attacks by kings
            if (board[r][f] == 6):
                for r2 in range(r-1, r+2):
                    for f2 in range(f-1, f+2):
                        if (r2 == r) and (f2 == f):
                            pass
                        elif (r2 >= 0) and (r2 <= 7) and (f2 >= 0) and (f2 <= 7):
                            n_attacks_by_white[r2][f2] += 1
            if (board[r][f] == -6):
                for r2 in range(r-1, r+2):
                    for f2 in range(f-1, f+2):
                        if (r2 == r) and (f2 == f):
                            pass
                        elif (r2 >= 0) and (r2 <= 7) and (f2 >= 0) and (f2 <= 7):
                            n_attacks_by_black[r2][f2] += 1

    return (n_attacks_by_white, n_attacks_by_black)



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
