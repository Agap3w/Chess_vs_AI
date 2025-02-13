"""
Input: Zstandard compressed PGN files (.zst)

What it does:
1.  extract the file
2.  filter by ELO range 1800-2300 1MM matches, storing 
    - (X) each match in a matrix(n,8x8x12) where n=num moves, 8x8 is the board representation and 12 are the pieces one-hot encoded
    - (Y) each match result in a matrix (3,) one-hot encoded for white win, black win, draw
3.  Convert into TFRecord format, splitting 1MM matches into 100 shards of 10,000 matches each
4. compress with .gz

This code use multiprocessing to speed up the game processing.

Example Usage:
    python create_tfrecords.py \
        --pgn_path path/to/your/games.pgn.zst \
        --output_path path/to/output/directory \
        --num_games 1000000 \  # Optional: Number of games to process
        --shard_size 10000    # Optional: Number of games per shard
"""


import zstandard
import io
import chardet
import logging
import chess
import chess.pgn
import numpy as np
import multiprocessing as mp
import io
import tensorflow as tf
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOARD_SHAPE = (8, 8, 12)  # Define the constant board shape globally

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
    
    return matrix

def game_to_matrices(game: chess.pgn.Game) -> List[np.ndarray]:
    """Convert a chess game to a sequence of board matrices."""
    matrices = []
    board = game.board()
    
    # Add initial position
    matrices.append(board_to_efficient_matrix(board))
    
    # Add position after each move
    for move in game.mainline_moves():
        board.push(move)
        matrices.append(board_to_efficient_matrix(board))
    
    return matrices

def result_to_vector(result: str) -> np.ndarray:
    """Convert game result to one-hot encoded vector."""
    result_map = {
        "1-0": [1, 0, 0],    # White wins
        "0-1": [0, 1, 0],    # Black wins
        "1/2-1/2": [0, 0, 1] # Draw
    }
    return np.array(result_map.get(result, [0, 0, 1]), dtype=np.int8) # int8 type

def process_game(game_text: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Process a single game and return (game_matrices, result_vector, num_moves)."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            return None
            
        # Check Elo requirements
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        except ValueError:
            return None
        
        if not (1800 <= white_elo <= 2300 and 1800 <= black_elo <= 2300):
            return None
        
        # Convert game to matrices
        matrices = game_to_matrices(game)
        result_vector = result_to_vector(game.headers.get("Result", "*"))
        return np.array(matrices), result_vector, len(matrices)
    except Exception as e:
        logging.warning(f"Error processing game: {str(e)}")
        return None

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _uint16_feature(value):
    """Returns an int64_list from a uint16 (converted)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))  # Convert to int

def create_tfrecord_writer(output_path: str, shard_index: int) -> tf.io.TFRecordWriter:
    """Create TFRecord writer with GZIP compression."""
    filename = f"{output_path}_shard_{shard_index:03d}.tfrecord.gz"  # .gz extension
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    return tf.io.TFRecordWriter(filename, options=options)

def write_game_to_tfrecord(writer, matrices, result, num_moves):
    """Write game to TFRecord (corrected)."""
    board_states_bytes_list = []  # List to store byte strings
    for matrix in matrices:
        board_states_bytes_list.append(matrix.tobytes())  # Convert each matrix to bytes

    feature = {
        'board_states': tf.train.Feature(bytes_list=tf.train.BytesList(value=board_states_bytes_list)),  # Use a list of bytes
        'result': _bytes_feature(result.tobytes()),
        'num_moves': _uint16_feature(num_moves),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

def split_pgn_into_games(pgn_path: str):
    """Split Zstandard compressed PGN file into individual games."""
    try:
        with open(pgn_path, 'rb') as f_zst:  # Open in binary mode
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f_zst) as zf:  # Decompress with zstandard
                raw_data = zf.read(4096)  # Read a chunk to detect encoding
                encoding_info = chardet.detect(raw_data)
                encoding = encoding_info['encoding']

                try:
                    reader = io.TextIOWrapper(zf, encoding=encoding, errors='replace')  # Decode with detected encoding, handle errors
                    current_game = []
                    for line in reader:
                        if line.startswith('[Event ') and current_game:
                            yield ''.join(current_game)
                            current_game = []
                        current_game.append(line)
                    if current_game:
                        yield ''.join(current_game)

                except UnicodeDecodeError:
                    logging.error(f"Decoding error with detected encoding {encoding}. Trying 'latin-1'.")
                    try:
                        reader = io.TextIOWrapper(zf, encoding='latin-1', errors='replace')  # Fallback to latin-1
                        current_game = []
                        for line in reader:
                            if line.startswith('[Event ') and current_game:
                                yield ''.join(current_game)
                                current_game = []
                            current_game.append(line)
                        if current_game:
                            yield ''.join(current_game)
                    except Exception as e:
                        logging.error(f"Error decoding with both {encoding} and 'latin-1': {e}")
                        return

    except FileNotFoundError:
        logging.error(f"File not found: {pgn_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred while decompressing/reading the file: {e}")
        return

def process_pgn_file(pgn_path: str, output_path: str, num_games: int = 1_000_000, shard_size: int = 10000):
    """Process PGN, create TFRecord with optimizations."""
    logging.info(f"Starting processing of {pgn_path}")
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    chunk_size = 1000
    current_chunk = []
    processed_games = 0
    current_shard = 0
    writer = create_tfrecord_writer(output_path, current_shard)

    try:
        for game_text in split_pgn_into_games(pgn_path):
            if processed_games >= num_games:
                break
            current_chunk.append(game_text)

            if len(current_chunk) >= chunk_size:
                chunk_results = pool.map(process_game, current_chunk)
                valid_results = [r for r in chunk_results if r is not None]

                for matrices, result, num_moves in valid_results:
                    if processed_games >= num_games:
                        break

                    if processed_games % shard_size == 0 and processed_games > 0:
                        writer.close()
                        current_shard += 1
                        writer = create_tfrecord_writer(output_path, current_shard)

                    write_game_to_tfrecord(writer, matrices, result, num_moves) # removed board_shape
                    processed_games += 1

                    if processed_games % 1000 == 0:
                        logging.info(f"Processed and stored {processed_games} games")

                current_chunk = []

        # Process any remaining games in the last chunk
        if current_chunk:
            chunk_results = pool.map(process_game, current_chunk)
            valid_results = [r for r in chunk_results if r is not None]
            for matrices, result, num_moves in valid_results:
                if processed_games >= num_games:
                    break
                write_game_to_tfrecord(writer, matrices, result, num_moves) # removed board_shape
                processed_games += 1

        logging.info(f"Successfully processed {processed_games} games")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
    finally:
        writer.close()
        pool.close()
        pool.join()

    # Write metadata (including shape if constant)
    with open(f"{output_path}_metadata.txt", 'w') as f:
        f.write(f"Total games: {processed_games}\n")
        f.write(f"Total shards: {current_shard + 1}\n")
        f.write(f"Games per shard: {shard_size}\n")
        f.write(f"Board state shape: {BOARD_SHAPE}\n")  # Write the shape ONCE


if __name__ == "__main__":
    pgn_path = r"C:\Users\Matte\Desktop\temp chess\lichess_db_standard_rated_2018-06.pgn.zst" # Your PGN file
    output_path = r"C:\Users\Matte\Desktop\temp chess\Train_Dataset"  # Output directory for TFRecords
    num_games = 1000000  # Number of games to process (optional)
    shard_size = 10000  # Number of games per shard (optional)

    process_pgn_file(pgn_path, output_path, num_games, shard_size)
    print("TFRecord creation complete!") # Indicate completion