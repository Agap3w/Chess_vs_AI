import tensorflow as tf
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_tfrecord(example):
    """Parse TFRecord example."""
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'num_legal_moves': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature)
    
    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int16)
    num_legal_moves = tf.io.decode_raw(example['num_legal_moves'], tf.int16)

    x = tf.reshape(x, (8, 8, 12))
    x = tf.cast(x, tf.float32)
    
    return x, y, num_legal_moves

def prepare_data(x, y, num_legal_moves):
    num_legal_moves = tf.cast(num_legal_moves, tf.int32)
    indices = tf.range(num_legal_moves[0], dtype=tf.int32)
    indices = tf.expand_dims(indices, 1)
    
    move_mask = tf.zeros(MAX_MOVES, dtype=tf.float32)
    move_mask = tf.tensor_scatter_nd_update(
        move_mask,
        indices,
        tf.ones(num_legal_moves[0], dtype=tf.float32)
    )
    return {'input_board': x}, {'move': y, 'mask': move_mask}

def create_dataset(tfrecord_pattern: str, batch_size: int = 32, validation_split: float = 0.1):
    """Create dataset from TFRecord files with optimized file-level splitting."""
    # Get and sort files for deterministic splitting
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found matching pattern: {tfrecord_pattern}")
    
    # Calculate split index
    split_index = int(len(tfrecord_files) * (1 - validation_split))
    train_files = tfrecord_files[:split_index]
    val_files = tfrecord_files[split_index:]
    
    def create_split(files):
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    return create_split(train_files), create_split(val_files)

def create_chess_model():
    """Create an improved neural network model."""
    inputs = tf.keras.Input(shape=(8, 8, 12))
    
    # First conv block
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second conv block
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Third conv block
    x = tf.keras.layers.Conv2D(512, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(MAX_MOVES, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def masked_sparse_categorical_crossentropy(y_true, y_pred, mask):
    """Custom loss function that handles move masking."""
    y_pred = y_pred * mask
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis=-1, keepdims=True) + 1e-7)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def train_model(tfrecord_pattern: str, epochs: int = 10, batch_size: int = 64, validation_split: float = 0.1):
    """Train the chess model with improved parameters and validation split."""
    # Create datasets with optimized splitting
    train_dataset, val_dataset = create_dataset(tfrecord_pattern, batch_size, validation_split)
    
    # Estimate steps per epoch based on number of training files
    num_train_files = len(glob.glob(tfrecord_pattern)) * (1 - validation_split)
    estimated_train_size = int(DB_SIZE * (1 - validation_split))
    steps_per_epoch = estimated_train_size // batch_size

    # Create and compile model
    base_model = create_chess_model()
    board_input = tf.keras.Input(shape=(8, 8, 12), name='input_board')
    predictions = base_model(board_input)
    model = tf.keras.Model(
        inputs={'input_board': board_input},
        outputs={'move': predictions}
    )
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=steps_per_epoch * epochs,
        alpha=0.0001
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_top5_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_top5_accuracy')
    
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    val_top5_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='val_top5_accuracy')
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        for metric in [train_loss, train_accuracy, train_top5_accuracy,
                      val_loss, val_accuracy, val_top5_accuracy]:
            metric.reset_states()
        
        # Training phase
        for batch_idx, batch in enumerate(train_dataset):
            inputs, targets = batch
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = masked_sparse_categorical_crossentropy(
                    targets['move'],
                    predictions['move'],
                    targets['mask']
                )
                loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss.update_state(loss)
            train_accuracy.update_state(targets['move'], predictions['move'])
            train_top5_accuracy.update_state(targets['move'], predictions['move'])
            
            if batch_idx % 1000 == 0:
                current_lr = optimizer._decayed_lr(tf.float32).numpy()
                logging.info(
                    f"Batch {batch_idx}, "
                    f"Loss: {train_loss.result():.4f}, "
                    f"Accuracy: {train_accuracy.result():.4f}, "
                    f"Top-5 Accuracy: {train_top5_accuracy.result():.4f}, "
                    f"Learning Rate: {current_lr:.6f}"
                )
        
        # Validation phase
        for val_batch in val_dataset:
            val_inputs, val_targets = val_batch
            val_predictions = model(val_inputs, training=False)
            val_loss_value = masked_sparse_categorical_crossentropy(
                val_targets['move'],
                val_predictions['move'],
                val_targets['mask']
            )
            val_loss.update_state(tf.reduce_mean(val_loss_value))
            val_accuracy.update_state(val_targets['move'], val_predictions['move'])
            val_top5_accuracy.update_state(val_targets['move'], val_predictions['move'])
        
        # End of epoch logging
        logging.info(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss.result():.4f}, "
            f"Train Accuracy: {train_accuracy.result():.4f}, "
            f"Train Top-5 Accuracy: {train_top5_accuracy.result():.4f}, "
            f"Val Loss: {val_loss.result():.4f}, "
            f"Val Accuracy: {val_accuracy.result():.4f}, "
            f"Val Top-5 Accuracy: {val_top5_accuracy.result():.4f}"
        )
    
    return model

if __name__ == "__main__":
    tfrecord_pattern = r"C:\Users\Matte\Desktop\temp chess\Train_Dataset\Medium Shard\Train_Dataset_shard_*.tfrecord.gz"
    MAX_MOVES = 120
    DB_SIZE = 10000000
    
    model = train_model(
        tfrecord_pattern,
        epochs=20,
        batch_size=256,
        validation_split=0.1
    )
    
    # Recompile before saving
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=masked_sparse_categorical_crossentropy,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    model.save("chess_model")
    logging.info("Training complete! Model saved as 'chess_model'")