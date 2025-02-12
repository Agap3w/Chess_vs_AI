""" This code extract the zst file in which i found the chess database """

import zstandard as zstd
import chess.pgn

# Paths to the compressed and decompressed files
compressed_file = r"C:\Users\Matte\Desktop\temp chess\lichess_db_standard_rated_2013-01.pgn.zst"
decompressed_file = r"C:\Users\Matte\Desktop\temp chess\chessDB_lite.pgn"

# Decompress the .pgn.zst file
with open(compressed_file, 'rb') as compressed:
    with open(decompressed_file, 'wb') as decompressed:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed, decompressed)

# Now you can read the decompressed .pgn file
with open(decompressed_file, 'r') as f:
    for i in range(5):  # Read first 5 games
        game = chess.pgn.read_game(f)
        if game:
            print(game.headers)  # Prints game headers like event, date, players, etc.
            print(game)  # Prints the full game moves