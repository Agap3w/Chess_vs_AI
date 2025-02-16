import chess

# Initialize the chess board
board = chess.Board()

# Get all legal moves
legal_moves = board.legal_moves

# Print the list of legal moves
for move in legal_moves:
    print(move)