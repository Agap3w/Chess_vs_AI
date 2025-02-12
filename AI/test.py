import numpy as np
import chess.pgn
import io
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

# Global constant for piece values
PIECE_VALUES = {
    'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0,
    'p': -1/6, 'n': -2/6, 'b': -3/6, 'r': -4/6, 'q': -5/6, 'k': -1.0
}

def read_pgn_file(pgn_file):
    """Generator that reads a PGN file game-by-game"""
    with open(pgn_file, 'r', encoding='utf-8') as f:
        game_str = ""
        for line in f:
            if line.strip() == "" and game_str.startswith('[Event'):
                yield game_str
                game_str = ""
            else:
                game_str += line
        if game_str.startswith('[Event'):
            yield game_str  # Last game

def board_to_matrix(board):
    """Convert a chess.Board to matrix representation using vectorized operations"""
    matrix = np.zeros((8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        rank = 7 - (square >> 3)  # Get rank (row)
        file = square & 7         # Get file (column)
        matrix[rank][file] = PIECE_VALUES[piece.symbol()]
    
    return matrix

def process_game(game_str):
    """Process a single game and return its features and label"""
    try:
        game = chess.pgn.read_game(io.StringIO(game_str))
        if game is None:
            print("Failed to read game")
            return None
        
        # Check ELO requirements
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if not (1800 <= white_elo <= 2200 and 1800 <= black_elo <= 2200):
            print("ELO requirements not met")
            return None
            
        # Convert result to one-hot encoded label
        result = game.headers["Result"]
        if result == "1-0":
            label = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # White wins
        elif result == "0-1":
            label = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Black wins
        else:  # Draw
            label = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Draw
            
        # Process moves
        board = game.board()
        positions = []
        
        for move in game.mainline_moves():
            board.push(move)
            positions.append(board_to_matrix(board))
        
        print("Game processed successfully")
        return np.array(positions, dtype=np.float32), label
        
    except Exception as e:
        print(f"Error processing game: {e}")
        return None

def process_chunk(chunk_data):
    """Process a chunk of games"""
    results = []
    for game_str in chunk_data:
        result = process_game(game_str)
        if result is not None:
            results.append(result)
    return results

def create_dataset(pgn_file, max_games=500000, chunk_size=1000):
    """Create the dataset from PGN file without loading it all into RAM"""
    print(f"Starting to process {max_games} games from {pgn_file}")
    print(f"Using chunk size of {chunk_size}")

    games = []
    games_processed = 0

    print("Reading PGN file...")
    for game_str in read_pgn_file(pgn_file):
        if games_processed >= max_games:
            break
        games.append(game_str)
        games_processed += 1

    # Debug: Print number of games found
    print(f"Number of games read: {len(games)}")
    
    # Process games in parallel
    num_cores = max(1, mp.cpu_count() // 2)
    chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]

    all_positions = []
    all_labels = []
    
    print(f"Processing games using {num_cores} cores...")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for chunk_results in tqdm(executor.map(process_chunk, chunks), total=len(chunks)):
            for positions, label in chunk_results:
                all_positions.append(positions)
                all_labels.append(label)

    # Convert to final numpy arrays
    X = np.concatenate(all_positions, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.float32)

    print(f"\nProcessed {games_processed} games")

def save_dataset(X, y, output_dir="chess_dataset"):
    """Save the processed dataset"""
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(f"{output_dir}/chess_data.npz", X=X, y=y)
    print(f"Dataset saved to {output_dir}/chess_data.npz")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    pgn_file = r"C:\Users\Matte\Desktop\temp chess\lichess_db_standard_rated_2018-06.pgn"

    if not os.path.exists(pgn_file):
        print(f"Error: Input file not found: {pgn_file}")
        exit(1)
        
    try:
        print("Testing with 100 games...")
        X, y = create_dataset(pgn_file, max_games=100, chunk_size=10)
        save_dataset(X, y, output_dir=r"C:\Users\Matte\Desktop\temp chess\chessDB_lite_test")
    
    except Exception as e:
        print(f"Error during processing: {e}")
