import tensorflow as tf
import numpy as np
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_tfrecord(example):
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'num_legal_moves': tf.io.FixedLenFeature([], tf.string),  # Add this line
    }
    example = tf.io.parse_single_example(example, feature)
    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int16)
    num_legal_moves = tf.io.decode_raw(example['num_legal_moves'], tf.int16)  # Add this line

    x = tf.reshape(x, (8, 8, 12))
    y = tf.reshape(y, (1,))
    num_legal_moves = tf.reshape(num_legal_moves, (1,))  # Add this line

    return x, y, num_legal_moves  # Return num_legal_moves

def inspect_tfrecord(tfrecord_path, num_examples=2):
    dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord)

    num_examples_checked = 0
    num_errors = 0  # Keep track of errors

    for i, (x, y, num_legal_moves) in enumerate(dataset):
        num_examples_checked += 1

        # NaN/Inf checks (as before):
        x_float = tf.cast(x, tf.float32)
        try: # Try block to catch the exception and count it instead of stopping the execution
            tf.debugging.check_numerics(x_float, "NaN/Inf in x detected!")
            tf.debugging.check_numerics(tf.cast(y, tf.float32), "NaN/Inf in y detected!")
            tf.debugging.check_numerics(tf.cast(num_legal_moves, tf.float32), "NaN/Inf in num_legal_moves detected!")
        except tf.errors.InvalidArgumentError as e:
            logging.error(f"NaN/Inf error in example: {i}, Error: {e}")
            num_errors += 1

        max_legal_moves = 218
        is_y_in_range = tf.reduce_all(tf.logical_and(y >= 0, y < num_legal_moves))
        try:
            tf.debugging.assert_equal(is_y_in_range, True, message=f"y is out of range! y: {y.numpy()}, num_legal_moves: {num_legal_moves.numpy()}, Example: {i}")
        except AssertionError as e:
            logging.error(e)
            num_errors += 1

        is_num_legal_moves_valid = tf.less_equal(num_legal_moves, max_legal_moves)
        try:
            tf.debugging.assert_equal(is_num_legal_moves_valid, True, message=f"num_legal_moves is too large! num_legal_moves: {num_legal_moves.numpy()}, Example: {i}")
        except AssertionError as e:
            logging.error(e)
            num_errors += 1

        if i % 1000 == 0:  # Log every 1000 examples (adjust as needed)
            logging.info(f"Checked {i} examples in {tfrecord_path}. Errors found: {num_errors}")

    logging.info(f"Finished checking {num_examples_checked} examples in {tfrecord_path}. Total errors found: {num_errors}")


if __name__ == "__main__":
    tfrecord_files = glob.glob(r"C:\Users\Matte\Desktop\temp chess\Train_Dataset_shard_*.tfrecord.gz")

    for tfrecord_file in tfrecord_files:
        logging.info(f"Inspecting TFRecord file: {tfrecord_file}")
        inspect_tfrecord(tfrecord_file)