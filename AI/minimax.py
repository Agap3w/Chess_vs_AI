import chess
from typing import Optional, Tuple
import random

def minimax_best_move(board, board_score, depth=3):
    """
    Find the best move using minimax algorithm with alpha-beta pruning.
    
    Args:
        board: Current chess board state
        board_score: BoardScore instance for position evaluation
        depth: Maximum depth to search
        
    Returns:
        Best move found or None if no legal moves
    """
    
    def alpha_beta(board, depth, alpha, beta, maximizing):
        """
        Recursive alpha-beta pruning implementation.
        
        Args:
            board: Current board state
            depth: Remaining depth to search
            alpha: Best score for maximizing player
            beta: Best score for minimizing player
            maximizing: True if maximizing player's turn
            
        Returns:
            Tuple of (best score, best move)
        """
        # Base cases
        if depth == 0 or board.is_game_over():
            return board_score.get_score(), None
            
        best_move = None
        legal_moves = list(board.legal_moves)
        
        # Sort moves to improve pruning (captures and checks first)
        legal_moves.sort(key=lambda move: (
            board.is_capture(move),
            board.gives_check(move)
        ), reverse=True)
        
        # per ora AI gioca solo col nero quindi questa parte non servirebbe ma mi porto avanti
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
                    
            return max_eval, best_move
                    
        else: # se minimizing (=AI gioca col nero)
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
                    
            return min_eval, best_move

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    # White maximizes, Black minimizes
    is_maximizing = board.turn == chess.WHITE
    _, best_move = alpha_beta(
        board,
        depth,
        float('-inf'),
        float('inf'),
        is_maximizing
    )
    
    # If multiple moves have the same evaluation, choose randomly among them
    if best_move is None:
        return random.choice(legal_moves)
        
    return best_move