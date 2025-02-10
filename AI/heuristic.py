def heuristic_best_move(board, board_score):
    """Find and make the best move based on board score evaluation."""
    best_move = None
    best_score = float('inf')

    # Evaluate each legal move
    for move in board.legal_moves:
        # Make the move
        board.push(move)
        
        score = board_score.get_score()
        
        # Undo the move
        board.pop()
        
        # Update best move if this score is better
        if score < best_score:
            best_score = score
            best_move = move

    # Make the best move if one was found
    if best_move:
        print(f"best_move = {best_move} , best_score = {best_score}")
        return best_move
    return None