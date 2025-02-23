import tensorflow as tf
import numpy as np
import chess
import chess.pgn
from AI.resources.supervised_CNN_train import EnhancedChessCNN


# Costanti (input, output, trained model)
BOARD_SHAPE = (8, 8, 15)
MOVE_VECTOR_SIZE = 4096
model_path = r"C:\Users\Matte\main_matte_py\Chess_vs_AI\AI\resources\supervised_model.h5"

def board_to_efficient_matrix(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board into an efficient 8x8x15 one-hot matrix representation (input format of trained model).
    
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Kingside castling rights (0: no rights, 1: has rights)
    13: Queenside castling rights (0: no rights, 1: has rights)
    14: Turn (0: white to move, 1: black to move)
    """
    # Initialize 8x8x15 matrix
    matrix = np.zeros((8, 8, 15), dtype=np.float32)
    
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    # Fill piece positions (channels 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            plane = piece_to_plane[piece.symbol()]
            matrix[rank, file, plane] = 1.0
    
    # Fill castling rights (channels 12-13)
    if board.turn:  # White's turn
        has_kingside = bool(board.castling_rights & chess.BB_H1)
    else:  # Black's turn
        has_kingside = bool(board.castling_rights & chess.BB_H8)
    matrix[:, :, 12] = has_kingside

    if board.turn:  
        has_queenside = bool(board.castling_rights & chess.BB_A1)
    else:  
        has_queenside = bool(board.castling_rights & chess.BB_A8)
    matrix[:, :, 13] = has_queenside
    
    # Fill turn (channel 14)
    matrix[:, :, 14] = int(not board.turn)  # 0 for white, 1 for black
    
    return matrix

def create_legal_moves_mask(board: chess.Board) -> np.ndarray:
    """Create a mask of legal moves in the 4096 one-hot format"""
    legal_moves_mask = np.zeros(MOVE_VECTOR_SIZE, dtype=np.int8)
    
    for move in board.legal_moves:
        move_idx = move.from_square * 64 + move.to_square
        legal_moves_mask[move_idx] = 1
    
    return legal_moves_mask

def index_to_move(idx: int) -> chess.Move:
    """Convert a move index (0-4095) to a chess.Move object"""
    from_sq = idx // 64
    to_sq = idx % 64
    
    from_square = chess.square(from_sq % 8, from_sq // 8)
    to_square = chess.square(to_sq % 8, to_sq // 8)
    
    return chess.Move(from_square, to_square)

def load_model_with_weights(model_path):
    """
    Reconstruct the model architecture and load trained weights.
    
    This function:
    1. Creates a new instance of our CNN model
    2. Builds the model graph with a dummy input
    3. Loads the trained weights
    """
    # Create new model instance
    model = EnhancedChessCNN()
    
    # Build model architecture using a dummy input
    # Shape must match your training input: (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, 8, 8, 15))
    _ = model(dummy_input)
    
    # Load trained weights
    model.load_weights(model_path)
    return model

class ChessMovePredictor:
    def __init__(self):
        """Initialize the predictor with a trained model"""
        try:
            # Try to load the model using the new method
            self.model = load_model_with_weights(model_path)
            print(f"Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def predict_move(self, board: chess.Board, top_k=5):
        """
        Predict the best move for the given board position
        
        Args:
            board: A chess.Board object
            top_k: Number of top moves to consider
        
        Returns:
            The best legal move as a chess.Move object
        """
        # Convert board to input format
        x = board_to_efficient_matrix(board)
        x = np.expand_dims(x, axis=0).astype(np.float32)
        
        # Get legal moves mask
        legal_moves_mask = create_legal_moves_mask(board)
        
        # Get model prediction
        predictions = self.model.predict(x, verbose=0)[0]
        
        # Apply legal moves mask
        masked_predictions = predictions * legal_moves_mask
        
        # Sort and get top-k legal moves
        top_k_indices = np.argsort(masked_predictions)[-top_k:][::-1]
        top_k_scores = masked_predictions[top_k_indices]
        
        # Convert indices to chess moves
        top_moves = [index_to_move(idx) for idx in top_k_indices]
        
        # Print top moves and their scores
        print("\nTop predicted moves:")
        for i, (move, score) in enumerate(zip(top_moves, top_k_scores)):
            print(f"{i+1}. {move.uci()} (confidence: {score:.4f})")
        
        # Return the highest-scoring legal move
        best_move = top_moves[0]
        
        # Special handling for promotion
        # Check if this is a pawn reaching the last rank
        if (board.piece_at(best_move.from_square) 
            and board.piece_at(best_move.from_square).piece_type == chess.PAWN):
            
            if (chess.square_rank(best_move.to_square) == 7 and board.turn == chess.WHITE) or \
               (chess.square_rank(best_move.to_square) == 0 and board.turn == chess.BLACK):
                # Add promotion to queen
                best_move = chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)
                print(f"Promotion detected: {best_move.uci()}")
        
        return best_move

def supervised_move(board: chess.Board, predictor, turn: bool = chess.BLACK) -> chess.Move:
    """
    Predicts the best move for a given board position and turn.

    Args:
        board: The chess board.
        model_path: the path of the model
        turn: The color to move (chess.WHITE or chess.BLACK). Defaults to chess.BLACK.

    Returns:
        The best move as a chess.Move object.
    """
    # Set the board's turn to the specified turn
    board.turn = turn

    # Convert board to input format
    x = board_to_efficient_matrix(board)
    x = np.expand_dims(x, axis=0).astype(np.float32)

    # Get legal moves mask
    legal_moves_mask = create_legal_moves_mask(board)

    # Get model prediction
    predictions = predictor.model.predict(x, verbose=0)[0]

    # Apply legal moves mask
    masked_predictions = predictions * legal_moves_mask

    # Get the best move index
    best_move_index = np.argmax(masked_predictions)

    # Convert index to chess move
    best_move = index_to_move(best_move_index)

    # Special handling for promotion
    if (board.piece_at(best_move.from_square) and board.piece_at(best_move.from_square).piece_type == chess.PAWN):
        if (chess.square_rank(best_move.to_square) == 7 and board.turn == chess.WHITE) or \
           (chess.square_rank(best_move.to_square) == 0 and board.turn == chess.BLACK):
            best_move = chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)

    return best_move