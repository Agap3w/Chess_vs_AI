"""
input: tfrecord.gz

This script inspects TFRecord files containing chess game data. It reads
the TFRecord files, decodes the board state matrices and result vectors,
and prints information about the games, including the shapes of the data
and a slice of the first board state.  It is designed to work with
TFRecord files created by `create_tfrecords.py`.

The script performs the following steps:

1.  Reads TFRecord files from a specified directory.
2.  Parses the TFRecord examples, extracting the board state byte strings,
    result byte strings, and number of moves.
3.  Decodes the byte strings into NumPy arrays, reshaping the board state
    byte strings into 8x8x12 matrices.
4.  Prints the shapes of the board states and result vectors, the number
    of moves, and a slice of the initial board state for each game.

Example Usage:
    python inspect_tfrecord.py \
        --tfrecord_path path/to/tfrecord/file.tfrecord.gz

Or to iterate over all shards:

    python inspect_tfrecord.py \
        --output_path path/to/tfrecord/directory

"""


import tensorflow as tf
import numpy as np
import os

BOARD_SHAPE = (8, 8, 12)

def inspect_tfrecord(tfrecord_path):
    """Inspects a TFRecord file."""

    def _decode_and_reshape(board_states_bytes_list, result_bytes, num_moves):
        board_states = []
        for board_states_bytes in board_states_bytes_list:  # Iterate over the list
            board_state = np.frombuffer(board_states_bytes.numpy(), dtype=np.int8)
            board_state = board_state.reshape(BOARD_SHAPE) # removed the None because now the shape is correct
            board_states.append(board_state)
        board_states = np.stack(board_states) # stack the list of np array into one big np array
        result = np.frombuffer(result_bytes.numpy(), dtype=np.int8)
        return board_states, result, num_moves

    dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type='GZIP')
    dataset = dataset.map(lambda example_proto: tf.io.parse_single_example(
        example_proto,
        {
            'board_states': tf.io.VarLenFeature(tf.string),  # Use VarLenFeature
            'result': tf.io.FixedLenFeature([], tf.string),
            'num_moves': tf.io.FixedLenFeature([], tf.int64),
        }
    ))

    dataset = dataset.map(lambda parsed_features: tf.py_function(
        func=_decode_and_reshape,
        inp=[parsed_features['board_states'].values, parsed_features['result'], parsed_features['num_moves']], # Access .values
        Tout=[tf.int8, tf.int8, tf.int64]
    ))

    for board_states, result, num_moves in dataset.take(5):
        print("Board States Shape:", board_states.shape)
        print("Result Shape:", result.shape)
        print("Number of Moves:", num_moves.numpy())
        print("Board States (first game, first move):")
        print(board_states[0, 0, :, :])  # Print a slice of the board state
        print("Result:", result)
        print("-" * 20)

if __name__ == "__main__":
    input_path = r"C:\Users\Matte\Desktop\temp chess" # Your output path
    shard_number = 0  # Change this to 0, 1, or 2 to select the shard
    tfrecord_file = os.path.join(input_path, f"Train_Dataset_shard_{shard_number:03d}.tfrecord.gz")
    print(f"Inspecting shard: {tfrecord_file}")
    inspect_tfrecord(tfrecord_file)