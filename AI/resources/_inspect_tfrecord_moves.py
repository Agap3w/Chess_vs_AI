import tensorflow as tf
import numpy as np  # Import numpy at the top level

def parse_tfrecord(example):
    """Parse a single TFRecord example."""
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),  # Corrected: shape=[]
        'y': tf.io.FixedLenFeature([], tf.string),  # Corrected: shape=[]
        'legal_moves_mask': tf.io.FixedLenFeature([], tf.string),  # Corrected: shape=[]
    }
    example = tf.io.parse_single_example(example, feature)

    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int8)
    legal_moves_mask = tf.io.decode_raw(example['legal_moves_mask'], tf.int8)

    x = tf.reshape(x, (8, 8, 15))  # Replace with your x shape if different
    y = tf.reshape(y, (4096,))       # Replace with your y shape if different
    legal_moves_mask = tf.reshape(legal_moves_mask, (4096,)) # Replace with your mask shape if different

    return x, y, legal_moves_mask


def print_example(tfrecord_file):
    """Parses and prints a single example from a TFRecord file."""
    try:
        dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP") # Add compression type if needed
        for example in dataset.take(1):  # Take only one example
            x, y, legal_moves_mask = parse_tfrecord(example)

            np.set_printoptions(threshold=np.inf)  # Print the entire array, no truncation


            print("Shape of x:", x.shape)
            print("Shape of y:", y.shape)
            print("Shape of legal_moves_mask:", legal_moves_mask.shape)

            print("\nExample x:")
            print(x.numpy())  # Convert to NumPy for printing

            print("\nExample y:")
            print(y.numpy())

            print("\nExample legal_moves_mask:")
            print(legal_moves_mask.numpy())

            # Print some statistics (optional):
            print("\nLegal Moves Mask Stats:")
            print("Number of legal moves:", np.sum(legal_moves_mask.numpy()))
            print("Min Value:", np.min(legal_moves_mask.numpy()))
            print("Max Value:", np.max(legal_moves_mask.numpy()))

            return  # Exit after printing one example

    except Exception as e:
        print(f"Error processing TFRecord file: {e}")

if __name__ == "__main__":
    tfrecord_file = r"C:\Users\Matte\Desktop\temp_chess\e2e4\Train_Dataset\Medium\medium_e2e4_shard_000.tfrecord.gz"  # Replace with your TFRecord file path
    print_example(tfrecord_file)