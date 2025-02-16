import tensorflow as tf
import chess
import numpy as np
import logging

MAX_MOVES= 218

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Matte\main_matte_py\Chess_vs_AI\AI\resources\chess_model", compile=False)

def board_to_efficient_matrix(board: chess.Board) -> np.ndarray:
    """Convert a chess board to an efficient 8x8x12 one-hot matrix representation."""
    # 12 planes: 6 piece types × 2 colors
    matrix = np.zeros((8, 8, 12), dtype=np.int8)
    
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            plane = piece_to_plane[piece.symbol()]
            matrix[rank, file, plane] = 1
    
    return np.expand_dims(matrix, axis=0) 

# Function to test the model on a board
def test_model(board: chess.Board):
    input_data = board_to_efficient_matrix(board)
    
    # Get model predictions
    predictions = model.predict(input_data)['move']
    
    # Get the legal moves
    legal_moves = list(board.legal_moves)
    
    # Create a mask for the legal moves
    move_mask = np.zeros(MAX_MOVES)
    for i in range(len(legal_moves)):
        move_mask[i] = 1
    
    # Apply the mask to the predictions (just as during training)
    predictions_masked = predictions * move_mask # element-wise moltiplication neutrale per legal moves (*1) ma azzera le illegal moves (*0)
    predictions_normalized = predictions_masked / (np.sum(predictions_masked) + 1e-7) # Normalize predictions perché ora somma !=1 (apply softmax)
    
    # Get the best move index (the highest prediction)
    best_move_idx = np.argmax(predictions_normalized)
    
    # Find the best move
    best_move = legal_moves[best_move_idx]
    logging.info(f"Best move predicted: {best_move}")
    
    return best_move

# Create a chess board and test the model
board = chess.Board()
best_move = test_model(board)

# Apply the move to the board
board.push(best_move)
logging.info(f"Move applied: {best_move}")
logging.info(f"Board after move:\n{board}")
