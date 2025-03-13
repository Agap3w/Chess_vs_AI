import numpy as np
import chess
import functools


# Define move directions (8 queen directions and 8 knight directions)
# These are represented as (rank_delta, file_delta) pairs
QUEEN_DIRECTIONS = [
    (-1, 0),  # North
    (-1, 1),  # North-East
    (0, 1),   # East
    (1, 1),   # South-East
    (1, 0),   # South
    (1, -1),  # South-West
    (0, -1),  # West
    (-1, -1)  # North-West
]

KNIGHT_DIRECTIONS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]

# Define promotion piece types
PROMOTION_PIECE_TYPES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

def encode_move_universal(move):
    """
    Encodes a chess move into the universal encoding format.
    
    Args:
        move: A chess.Move object
    
    Returns:
        tuple: (square_index, plane_index) where:
            - square_index is in range [0, 63]
            - plane_index is in range [0, 72]
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Calculate rank and file deltas
    from_rank, from_file = chess.square_rank(from_square), chess.square_file(from_square)
    to_rank, to_file = chess.square_rank(to_square), chess.square_file(to_square)
    
    rank_delta = to_rank - from_rank
    file_delta = to_file - from_file
    
    # Check if this is a knight move
    is_knight_move = (abs(rank_delta), abs(file_delta)) in [(2, 1), (1, 2)]
    
    # Handle promotion moves
    if move.promotion:
        # Determine the direction of the pawn move
        # For promotions, we have 9 planes (3 piece types × 3 directions)
        if file_delta == 0:  # Straight ahead
            direction = 0
        elif file_delta == 1:  # Capture to the right
            direction = 1
        else:  # file_delta == -1, capture to the left
            direction = 2
            
        # Map the promotion piece type to an index
        if move.promotion == chess.KNIGHT:
            piece_index = 0
        elif move.promotion == chess.BISHOP:
            piece_index = 1
        elif move.promotion == chess.ROOK:
            piece_index = 2
        else:  # QUEEN is handled in the standard queen moves
            return encode_queen_move(from_square, to_square)
            
        # Calculate the plane index for underpromotions
        # 56 queen planes + 8 knight planes + (3 directions * piece_index) + direction
        plane_index = 56 + 8 + (3 * piece_index) + direction
        return from_square, plane_index
    
    # Handle knight moves
    elif is_knight_move:
        # Find which knight direction matches this move
        for i, (dr, df) in enumerate(KNIGHT_DIRECTIONS):
            if (rank_delta, file_delta) == (dr, df):
                # 56 queen planes + knight direction index
                plane_index = 56 + i
                return from_square, plane_index
                
    # Handle queen moves (including rook and bishop moves)
    else:
        return encode_queen_move(from_square, to_square)
    
    # If we can't encode the move, return None
    return None

def encode_queen_move(from_square, to_square):
    """Helper function to encode queen-like moves (queen, rook, bishop, pawn)"""
    from_rank, from_file = chess.square_rank(from_square), chess.square_file(from_square)
    to_rank, to_file = chess.square_rank(to_square), chess.square_file(to_square)
    
    rank_delta = to_rank - from_rank
    file_delta = to_file - from_file
    
    # Determine the direction and distance
    direction = None
    distance = max(abs(rank_delta), abs(file_delta))
    
    # Check which direction matches this move
    for i, (dr, df) in enumerate(QUEEN_DIRECTIONS):
        # Check if the move is in this direction
        if (rank_delta == 0 and file_delta == 0):
            continue  # Skip no-movement
            
        if (dr == 0 and df == 0):
            continue  # Skip the no-movement direction
            
        if ((rank_delta == 0 or file_delta == 0 or abs(rank_delta) == abs(file_delta)) and
            (dr == 0 and df == 0) or
            (dr != 0 and df != 0 and abs(dr) == abs(df) and
             rank_delta * dr >= 0 and file_delta * df >= 0 and
             abs(rank_delta) == abs(file_delta)) or
            (dr != 0 and df == 0 and rank_delta * dr >= 0 and file_delta == 0) or
            (dr == 0 and df != 0 and rank_delta == 0 and file_delta * df >= 0)):
            
            # This is the direction we're moving in
            direction = i
            break
    
    # If we couldn't determine the direction, it's not a valid queen move
    if direction is None:
        # Let's try a simpler approach - normalize the direction
        dr = 0 if rank_delta == 0 else rank_delta // abs(rank_delta)
        df = 0 if file_delta == 0 else file_delta // abs(file_delta)
        
        for i, (qdr, qdf) in enumerate(QUEEN_DIRECTIONS):
            if (dr, df) == (qdr, qdf):
                direction = i
                break
    
    if direction is None:
        return None  # We can't encode this move
        
    # Calculate the plane index
    # Each direction has 7 possible distances (1-7)
    plane_index = direction * 7 + (distance - 1)
    
    return from_square, plane_index

def decode_move_universal(square_index, plane_index):
    """
    Decodes a universal encoding back to a chess move.
    
    Args:
        square_index: Integer in range [0, 63]
        plane_index: Integer in range [0, 72]
        
    Returns:
        chess.Move: The corresponding chess move
    """
    # Extract the from_square coordinates
    from_rank = chess.square_rank(square_index)
    from_file = chess.square_file(square_index)
    
    # Handle queen moves (planes 0-55)
    if plane_index < 56:
        direction = plane_index // 7
        distance = (plane_index % 7) + 1
        
        # Get the direction deltas
        dr, df = QUEEN_DIRECTIONS[direction]
        
        # Calculate the to_square coordinates
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        
        # Check if the coordinates are within the board
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_square = chess.square(to_file, to_rank)
            return chess.Move(square_index, to_square)
    
    # Handle knight moves (planes 56-63)
    elif plane_index < 64:
        knight_direction = plane_index - 56
        dr, df = KNIGHT_DIRECTIONS[knight_direction]
        
        to_rank = from_rank + dr
        to_file = from_file + df
        
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_square = chess.square(to_file, to_rank)
            return chess.Move(square_index, to_square)
    
    # Handle underpromotions (planes 64-72)
    else:
        underpromotion_index = plane_index - 64
        piece_type_index = underpromotion_index // 3
        direction = underpromotion_index % 3
        
        # Calculate the to_square for pawn promotion
        if from_rank == 6:  # White pawn on 7th rank
            to_rank = 7
            if direction == 0:  # Straight ahead
                to_file = from_file
            elif direction == 1:  # Capture right
                to_file = from_file + 1
            else:  # Capture left
                to_file = from_file - 1
        else:  # Black pawn on 2nd rank
            to_rank = 0
            if direction == 0:  # Straight ahead
                to_file = from_file
            elif direction == 1:  # Capture right
                to_file = from_file + 1
            else:  # Capture left
                to_file = from_file - 1
        
        if 0 <= to_file < 8:
            to_square = chess.square(to_file, to_rank)
            promotion_piece = PROMOTION_PIECE_TYPES[piece_type_index]
            return chess.Move(square_index, to_square, promotion=promotion_piece)
    
    return None  # Invalid encoding

def build_move_lookup_tables():
    """
    Builds lookup tables for quick conversion between chess.Move objects and their universal encoding.
    
    Returns:
        tuple: (move_to_encoding, encoding_to_move)
    """
    move_to_encoding = {}
    encoding_to_move = {}
    
    # Generate all possible moves on an empty board
    for from_square in range(64):
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)
        
        # Regular moves
        for direction_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
            for distance in range(1, 8):
                to_rank = from_rank + dr * distance
                to_file = from_file + df * distance
                
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_square = chess.square(to_file, to_rank)
                    move = chess.Move(from_square, to_square)
                    
                    # Calculate the plane index
                    plane_index = direction_idx * 7 + (distance - 1)
                    
                    # Add to lookup tables
                    encoding = (from_square, plane_index)
                    move_to_encoding[move] = encoding
                    encoding_to_move[encoding] = move
        
        # Knight moves
        for knight_idx, (dr, df) in enumerate(KNIGHT_DIRECTIONS):
            to_rank = from_rank + dr
            to_file = from_file + df
            
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_square = chess.square(to_file, to_rank)
                move = chess.Move(from_square, to_square)
                
                # Calculate the plane index
                plane_index = 56 + knight_idx
                
                # Add to lookup tables
                encoding = (from_square, plane_index)
                move_to_encoding[move] = encoding
                encoding_to_move[encoding] = move
        
        # Promotions
        if from_rank == 6:  # White pawn on 7th rank
            for direction in range(3):
                to_rank = 7
                if direction == 0:  # Straight ahead
                    to_file = from_file
                elif direction == 1:  # Capture right
                    to_file = from_file + 1
                else:  # Capture left
                    to_file = from_file - 1
                
                if 0 <= to_file < 8:
                    to_square = chess.square(to_file, to_rank)
                    
                    # Add queen promotion to regular moves
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                    encoding = encode_queen_move(from_square, to_square)
                    if encoding:
                        move_to_encoding[move] = encoding
                        encoding_to_move[encoding] = move
                    
                    # Add underpromotions
                    for piece_idx, piece_type in enumerate(PROMOTION_PIECE_TYPES):
                        move = chess.Move(from_square, to_square, promotion=piece_type)
                        
                        # Calculate the plane index
                        plane_index = 64 + (piece_idx * 3) + direction
                        
                        # Add to lookup tables
                        encoding = (from_square, plane_index)
                        move_to_encoding[move] = encoding
                        encoding_to_move[encoding] = move
        
        elif from_rank == 1:  # Black pawn on 2nd rank
            for direction in range(3):
                to_rank = 0
                if direction == 0:  # Straight ahead
                    to_file = from_file
                elif direction == 1:  # Capture right
                    to_file = from_file + 1
                else:  # Capture left
                    to_file = from_file - 1
                
                if 0 <= to_file < 8:
                    to_square = chess.square(to_file, to_rank)
                    
                    # Add queen promotion to regular moves
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                    encoding = encode_queen_move(from_square, to_square)
                    if encoding:
                        move_to_encoding[move] = encoding
                        encoding_to_move[encoding] = move
                    
                    # Add underpromotions
                    for piece_idx, piece_type in enumerate(PROMOTION_PIECE_TYPES):
                        move = chess.Move(from_square, to_square, promotion=piece_type)
                        
                        # Calculate the plane index
                        plane_index = 64 + (piece_idx * 3) + direction
                        
                        # Add to lookup tables
                        encoding = (from_square, plane_index)
                        move_to_encoding[move] = encoding
                        encoding_to_move[encoding] = move
    
    return move_to_encoding, encoding_to_move

# Build lookup tables once and store them
MOVE_TO_ENCODING, ENCODING_TO_MOVE = build_move_lookup_tables()

def _move_to_index(move):
    """
    Convert a chess.Move to an index in the policy output.
    Uses the universal encoding scheme.
    
    Args:
        move: A chess.Move object
    
    Returns:
        int: Index in the range [0, 4671]
    """
    if move in MOVE_TO_ENCODING:
        square_index, plane_index = MOVE_TO_ENCODING[move]
        return square_index * 73 + plane_index
    
    # If the move isn't in our lookup table, try to encode it directly
    encoding = encode_move_universal(move)
    if encoding:
        square_index, plane_index = encoding
        return square_index * 73 + plane_index
    
    # If we can't encode the move, return None
    return None

@functools.lru_cache(maxsize=None)
def move_to_index(move):
    return _move_to_index(move)

def index_to_move(index):
    """
    Convert a policy index back to a chess.Move using universal encoding.
    
    Args:
        index: Integer in range [0, 4671]
        
    Returns:
        chess.Move: The corresponding move, or None if invalid
    """
    if index < 0 or index >= 4672:
        return None
    
    square_index = index // 73
    plane_index = index % 73
    
    # Use existing decoding function
    return decode_move_universal(square_index, plane_index)