import zstandard
import io
import chardet
import logging
import chess
import chess.pgn    
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOARD_SHAPE = (8, 8, 15)  # Define the constant board shape globally
MOVE_VECTOR_SIZE = 4096

def board_to_efficient_matrix(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to an efficient 8x8x15 one-hot matrix representation.
    
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Kingside castling rights (0: no rights, 1: has rights)
    13: Queenside castling rights (0: no rights, 1: has rights)
    14: Turn (0: white to move, 1: black to move)
    """
    # Initialize 8x8x15 matrix
    matrix = np.zeros((8, 8, 15), dtype=np.int8)
    
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
            matrix[rank, file, plane] = 1
    
    # Fill castling rights (channels 12-13)
    # Channel 12: Kingside castling rights
    if board.turn:  # White's turn
        has_kingside = bool(board.castling_rights & chess.BB_H1)
    else:  # Black's turn
        has_kingside = bool(board.castling_rights & chess.BB_H8)
    matrix[:, :, 12] = has_kingside

    # Channel 13: Queenside castling rights
    if board.turn:  # White's turn
        has_queenside = bool(board.castling_rights & chess.BB_A1)
    else:  # Black's turn
        has_queenside = bool(board.castling_rights & chess.BB_A8)
    matrix[:, :, 13] = has_queenside
    
    # Fill turn (channel 14)
    matrix[:, :, 14] = int(not board.turn)  # 0 for white, 1 for black
    
    return matrix

def parse_tfrecord(example):
    """
    Parse TFRecord example with updated board shape.
    """
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'legal_moves_mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature)
    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int8)
    legal_moves_mask = tf.io.decode_raw(example['legal_moves_mask'], tf.int8)

    x = tf.reshape(x, BOARD_SHAPE)  # Using updated shape
    y = tf.reshape(y, (MOVE_VECTOR_SIZE,))
    legal_moves_mask = tf.reshape(legal_moves_mask, (MOVE_VECTOR_SIZE,))

    return x, y, legal_moves_mask

def move_to_one_hot(move: chess.Move) -> np.ndarray:
    """Convert a chess move to a one-hot vector representation."""
    move_str = move.uci()  # Get move in UCI format (e.g., 'e2e4')
    from_sq = chess.SQUARES.index(chess.parse_square(move_str[:2]))
    to_sq = chess.SQUARES.index(chess.parse_square(move_str[2:4]))
    
    one_hot = np.zeros(4096, dtype=np.int8)  # 64 * 64 = 4096
    one_hot[from_sq * 64 + to_sq] = 1
    return one_hot

def game_to_xy_pairs(game: chess.pgn.Game) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:  # Correct type hint
    """Convert game to (X, best_move_vector, legal_moves_mask) triples."""
    result = game.headers.get("Result", "*")
    triples = []
    moves = list(game.mainline_moves())
    board = chess.Board()

    try:
        if result == "1-0":  # White wins
            for i in range(0, len(moves), 2):
                x = board_to_efficient_matrix(board)
                if i < len(moves):
                    move = moves[i]
                    if move in board.legal_moves:
                        legal_moves = list(board.legal_moves)
                        legal_moves_mask = np.zeros(MOVE_VECTOR_SIZE, dtype=np.int8)  # Initialize legal move mask
                        for legal_move in legal_moves:
                            move_index = chess.SQUARES.index(chess.parse_square(legal_move.uci()[:2])) * 64 + chess.SQUARES.index(chess.parse_square(legal_move.uci()[2:4]))
                            legal_moves_mask[move_index] = 1

                        best_move_index = chess.SQUARES.index(chess.parse_square(move.uci()[:2])) * 64 + chess.SQUARES.index(chess.parse_square(move.uci()[2:4]))
                        best_move_vector = np.zeros(MOVE_VECTOR_SIZE, dtype=np.int8)
                        best_move_vector[best_move_index] = 1

                        triples.append((x, best_move_vector, legal_moves_mask))  # Append the triple
                        board.push(move)

                        if i + 1 < len(moves):
                            board.push(moves[i + 1])
                    else:
                        logging.warning(f"Illegal move (White win) in game {game.headers.get('Event', 'Unknown')}: {move} in position {board.fen()}")
                        return []

        elif result == "0-1":  # Black wins
            for i in range(0, len(moves), 2):
                if i < len(moves):
                    white_move = moves[i]
                    if white_move in board.legal_moves:
                        board.push(white_move)
                        if i + 1 < len(moves):
                            black_move = moves[i + 1]
                            x = board_to_efficient_matrix(board)
                            if black_move in board.legal_moves:
                                legal_moves = list(board.legal_moves)
                                legal_moves_mask = np.zeros(MOVE_VECTOR_SIZE, dtype=np.int8)  # Initialize legal move mask
                                for legal_move in legal_moves:
                                    move_index = chess.SQUARES.index(chess.parse_square(legal_move.uci()[:2])) * 64 + chess.SQUARES.index(chess.parse_square(legal_move.uci()[2:4]))
                                    legal_moves_mask[move_index] = 1

                                best_move_index = chess.SQUARES.index(chess.parse_square(black_move.uci()[:2])) * 64 + chess.SQUARES.index(chess.parse_square(black_move.uci()[2:4]))
                                best_move_vector = np.zeros(MOVE_VECTOR_SIZE, dtype=np.int8)
                                best_move_vector[best_move_index] = 1

                                triples.append((x, best_move_vector, legal_moves_mask))  # Append the triple
                                board.push(black_move)
                            else:
                                logging.warning(f"Illegal move (Black win) in game {game.headers.get('Event', 'Unknown')}: {black_move} in position {board.fen()}")
                                return []
                    else:
                        logging.warning(f"Illegal move (White win) in game {game.headers.get('Event', 'Unknown')}: {white_move} in position {board.fen()}")
                        return []

        return triples

    except Exception as e:
        logging.warning(f"Error processing game: {e}")
        return []

def process_game(game_text: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Process a single game and return (game_matrices, result_vector, num_moves)."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            return []
            
        # Check Elo requirements
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        except ValueError:
            return []
        
        if not (1800 <= white_elo <= 2300 and 1800 <= black_elo <= 2300):
            return []
        
        pairs = game_to_xy_pairs(game)
        return pairs
    
    except Exception as e:
        logging.warning(f"Error processing game: {str(e)}")
        return []
    
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord_writer(output_path: str, shard_index: int) -> tf.io.TFRecordWriter:
    """Create TFRecord writer with GZIP compression."""
    filename = f"{output_path}_shard_{shard_index:03d}.tfrecord.gz"  # .gz extension
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    return tf.io.TFRecordWriter(filename, options=options)

def write_triple_to_tfrecord(writer, x, best_move_vector, legal_moves_mask):  # Correct arguments
    x_bytes = x.tobytes()
    best_move_vector_bytes = best_move_vector.tobytes()  # Correct variable name
    legal_moves_mask_bytes = legal_moves_mask.tobytes()

    feature = {
        'x': _bytes_feature(x_bytes),
        'y': _bytes_feature(best_move_vector_bytes),  # Correct variable name
        'legal_moves_mask': _bytes_feature(legal_moves_mask_bytes),
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

def process_pgn_file(pgn_path: str, output_path: str, max_pairs: int = 1_000_000, shard_size: int = 10000):
    """Process PGN, create TFRecord with optimizations."""
    logging.info(f"Starting processing of {pgn_path}")
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    chunk_size = 1000
    current_chunk = []
    processed_pairs = 0
    current_shard = 0
    writer = create_tfrecord_writer(output_path, current_shard)

    try:
        for game_text in split_pgn_into_games(pgn_path):
            if processed_pairs >= max_pairs:
                break
            current_chunk.append(game_text)

            if len(current_chunk) >= chunk_size:
                chunk_results = pool.map(process_game, current_chunk)
                current_chunk = []  # Clear the chunk *before* processing results

                for game_pairs in chunk_results:  # Iterate over the *results* of each game
                    if processed_pairs >= max_pairs:
                        break

                    for x, best_move_vector, legal_moves_mask in game_pairs:  # Correct unpacking
                        if processed_pairs == 0:  # Print once, after the first pair is created
                            print("Shape of x:", x.shape)
                            print("Shape of best_move_vector:", best_move_vector.shape)
                            print("Shape of legal_moves_mask:", legal_moves_mask.shape)
                            print("First example x:\n", x)
                            print("First example best_move_vector:\n", best_move_vector)
                            print("First example legal_moves_mask:\n", legal_moves_mask)

                        if processed_pairs % shard_size == 0 and processed_pairs > 0:
                            writer.close()
                            current_shard += 1
                            writer = create_tfrecord_writer(output_path, current_shard)

                        write_triple_to_tfrecord(writer, x, best_move_vector, legal_moves_mask)
                        processed_pairs += 1

                        if processed_pairs % 100000 == 0:
                            logging.info(f"Processed and stored {processed_pairs} pairs")

                current_chunk = []

        # Process any remaining games in the last chunk
        if current_chunk:
            chunk_results = pool.map(process_game, current_chunk)
            all_pairs = [pair for game_pairs in chunk_results for pair in game_pairs]
            for x, best_move_vector, legal_moves_mask in all_pairs:  # Correct unpacking
                if processed_pairs == 0:  # Print ONCE, after the first pair is created
                    print("Shape of x:", x.shape)
                    print("Shape of best_move_vector:", best_move_vector.shape)
                    print("Shape of legal_moves_mask:", legal_moves_mask.shape)
                    print("First example x:\n", x)
                    print("First example best_move_vector:\n", best_move_vector)
                    print("First example legal_moves_mask:\n", legal_moves_mask)

                if processed_pairs >= max_pairs:
                    break
                write_triple_to_tfrecord(writer, x, best_move_vector, legal_moves_mask)  # Correct call
                processed_pairs += 1

        logging.info(f"Successfully processed {processed_pairs} pairs")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
    finally:
        writer.close()
        pool.close()
        pool.join()

    # Write metadata (including shape if constant)
    with open(f"{output_path}_metadata.txt", 'w') as f:
        f.write(f"Total games: {processed_pairs}\n")
        f.write(f"Total shards: {current_shard + 1}\n")
        f.write(f"Games per shard: {shard_size}\n")
        f.write(f"Board state shape: {BOARD_SHAPE}\n")  # Write the shape ONCE

def parse_tfrecord(example):
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'legal_moves_mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature)
    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int8)  # Decode as int8
    legal_moves_mask = tf.io.decode_raw(example['legal_moves_mask'], tf.int8)  # Decode as int8

    x = tf.reshape(x, BOARD_SHAPE)
    y = tf.reshape(y, (MOVE_VECTOR_SIZE,))
    legal_moves_mask = tf.reshape(legal_moves_mask, (MOVE_VECTOR_SIZE,))

    return x, y, legal_moves_mask

if __name__ == "__main__":
    pgn_path = r"C:\Users\Matte\Desktop\temp_chess\LARGE_lichess_db_standard_rated_2018-06.pgn.zst" # Your PGN file
    output_path = r"C:\Users\Matte\Desktop\temp_chess\e2e4\Train_Dataset\Large\large_e2e4"  # Output directory for TFRecords
    max_pairs = 20000000  # Number of pairs to process (optional)
    shard_size = 200000  # Number of pairs per shard (optional)

    process_pgn_file(pgn_path, output_path, max_pairs, shard_size)
    print("TFRecord creation complete!") # Indicate completion