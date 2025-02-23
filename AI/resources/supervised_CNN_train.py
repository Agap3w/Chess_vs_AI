# import + GPU settings
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
from tensorflow.keras import layers, Model
import datetime
import glob
import os


def create_legal_moves_mask_loss():
    """Custom loss function that only considers legal moves."""
    def legal_moves_masked_loss(y_true, y_pred, legal_moves_mask):
        # Apply legal moves mask to predictions
        masked_pred = y_pred * tf.cast(legal_moves_mask, tf.float32)
        
        # Normalize predictions across legal moves only
        masked_pred = masked_pred / (tf.reduce_sum(masked_pred, axis=1, keepdims=True) + 1e-7)
        
        # Calculate cross-entropy loss only for legal moves
        loss = -tf.reduce_sum(
            y_true * tf.math.log(masked_pred + 1e-7) * tf.cast(legal_moves_mask, tf.float32),
            axis=1
        )
        return tf.reduce_mean(loss)
    return legal_moves_masked_loss

class EnhancedChessCNN(Model):
    def __init__(self):
        super(EnhancedChessCNN, self).__init__()
        
        # Expanded initial features with larger kernel
        self.conv1 = layers.Conv2D(192, 5, padding='same')
        self.bn1 = layers.BatchNormalization()
        
        # Feature extraction blocks with different kernel sizes
        self.local_features = [
            layers.Conv2D(96, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
        
        self.global_features = [
            layers.Conv2D(96, 5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
        
        # Attention block layers
        self.attn_q = layers.Conv2D(192 // 8, 1)  # Assuming 192 channels
        self.attn_k = layers.Conv2D(192 // 8, 1)
        self.attn_v = layers.Conv2D(192, 1)

        # Residual blocks with increased capacity
        self.res_blocks = []
        for _ in range(12):  # Increased from 6
            self.res_blocks.append([
                layers.Conv2D(192, 3, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dropout(0.1),  # Reduced dropout
                layers.Conv2D(192, 3, padding='same'),
                layers.BatchNormalization(),
                layers.Dropout(0.1)
            ])  
        
        # Enhanced policy head
        self.policy_conv1 = layers.Conv2D(192, 3, padding='same')
        self.policy_bn1 = layers.BatchNormalization()
        
        self.policy_conv2 = layers.Conv2D(96, 1, padding='same')
        self.policy_bn2 = layers.BatchNormalization()
        
        self.policy_flat = layers.Flatten()
        self.policy_dense1 = layers.Dense(1536, activation='relu')
        self.policy_dropout = layers.Dropout(0.15)
        self.policy_dense2 = layers.Dense(4096)
    
    def attention_block(self, x, channels):
        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)

        batch_size = tf.shape(x)[0]
        q = tf.reshape(q, [batch_size, -1, channels // 8])
        k = tf.reshape(k, [batch_size, -1, channels // 8])
        v = tf.reshape(v, [batch_size, -1, channels])

        q_dtype = q.dtype
        s = tf.matmul(q, k, transpose_b=True)
        # Cast the scalar to the input's dtype instead of hardcoding float32
        scale = tf.cast(tf.sqrt(tf.cast(channels // 8, tf.float32)), q_dtype)
        s = s / scale
        s = tf.nn.softmax(s)

        out = tf.matmul(s, v)
        out = tf.reshape(out, [batch_size, 8, 8, channels])
        return layers.add([x, out])
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        # Parallel feature processing
        local_x = x
        global_x = x
        
        # Local features
        for layer in self.local_features:
            if isinstance(layer, layers.BatchNormalization):
                local_x = layer(local_x, training=training)
            else:
                local_x = layer(local_x)
                
        # Global features
        for layer in self.global_features:
            if isinstance(layer, layers.BatchNormalization):
                global_x = layer(global_x, training=training)
            else:
                global_x = layer(global_x)
        
        # Combine features
        x = tf.concat([local_x, global_x], axis=-1)
        
        x = self.attention_block(x, 192)  # Assuming 192 channels after concatenation

        # Enhanced residual blocks
        for block in self.res_blocks:
            residual = x
            x = block[0](x)
            x = block[1](x, training=training)
            x = block[2](x)
            x = block[3](x, training=training)
            x = block[4](x)
            x = block[5](x, training=training)
            x = block[6](x, training=training)
            x = layers.add([x, residual])
            x = tf.nn.relu(x)
        
        # Enhanced policy head
        x = self.policy_conv1(x)
        x = self.policy_bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.policy_conv2(x)
        x = self.policy_bn2(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.policy_flat(x)
        x = self.policy_dense1(x)
        x = self.policy_dropout(x, training=training)
        x = self.policy_dense2(x)
        
        # Cast the final output back to float32 for numerical stability in loss calculation
        x = tf.nn.softmax(x)
        x = tf.cast(x, tf.float32)
        
        return x

def parse_tfrecord(example):
    """Parse TFRecord example with validation."""
    feature = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'legal_moves_mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature)
    
    # Decode and reshape features with validation
    x = tf.io.decode_raw(example['x'], tf.int8)
    x = tf.reshape(x, (8, 8, 15))
    x = tf.cast(x, tf.float32)
    
    # Validate board values (should be 0 or 1)
    x = tf.clip_by_value(x, 0.0, 1.0)
    
    y = tf.io.decode_raw(example['y'], tf.int8)
    y = tf.reshape(y, (4096,))
    y = tf.cast(y, tf.float32)
    
    # Validate y (should be one-hot)
    y = tf.clip_by_value(y, 0.0, 1.0)
    
    legal_moves_mask = tf.io.decode_raw(example['legal_moves_mask'], tf.int8)
    legal_moves_mask = tf.reshape(legal_moves_mask, (4096,))
    legal_moves_mask = tf.cast(legal_moves_mask, tf.float32)
    
    # Validate mask (should be 0 or 1)
    legal_moves_mask = tf.clip_by_value(legal_moves_mask, 0.0, 1.0)
    
    return x, y, legal_moves_mask

def prepare_dataset(tfrecord_files, batch_size):
    """Prepare dataset from TFRecord files."""
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

@tf.function
def train_step(model, optimizer, loss_fn, x, y_true, legal_moves_mask):
    """Single training step without mixed precision."""

    # Training step
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y_true, y_pred, legal_moves_mask)

    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Add gradient value checking
    grad_norms = [tf.norm(g) for g in gradients if g is not None]
    is_nan_detected = tf.reduce_any(tf.math.is_nan(grad_norms))

    if is_nan_detected:
        tf.print("WARNING: NaN gradients detected")
        return loss

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def train_model(model, train_dataset, val_dataset, optimizer, loss_fn, epochs, steps_per_epoch, validation_steps, checkpoint_dir, log_dir, lr_schedule):
    # Additional metrics
    train_legal_move_accuracy = tf.keras.metrics.Mean()
    val_legal_move_accuracy = tf.keras.metrics.Mean()
    
    # Modify existing metrics initialization
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
    writer = tf.summary.create_file_writer(log_dir)
    
    # Early stopping setup
    best_val_accuracy = 0
    patience = 6
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Reset all metrics
        for metric in [train_loss, train_accuracy, train_top_k, 
                      val_loss, val_accuracy, val_top_k,
                      train_legal_move_accuracy, val_legal_move_accuracy]:
            metric.reset_states()
        
        # Training loop
        for step, (x, y, mask) in enumerate(train_dataset):
            if step >= steps_per_epoch:
                break
            
            y_indices = tf.argmax(y, axis=1)
            loss = train_step(model, optimizer, loss_fn, x, y, mask)
            predictions = model(x, training=False)
            
            # Update metrics including legal move accuracy
            legal_move_pred = predictions * tf.cast(mask, tf.float32)
            legal_move_pred = legal_move_pred / (tf.reduce_sum(legal_move_pred, axis=1, keepdims=True) + 1e-7)
            legal_move_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(legal_move_pred, axis=1), y_indices), tf.float32)
            )
            
            train_loss.update_state(loss)
            train_accuracy.update_state(y_indices, predictions)
            train_top_k.update_state(y_indices, predictions)
            train_legal_move_accuracy.update_state(legal_move_accuracy)
        
        # Validation loop with similar updates
        for step, (x, y, mask) in enumerate(val_dataset):
            if step >= validation_steps:
                break
            
            y_indices = tf.argmax(y, axis=1)
            predictions = model(x, training=False)
            val_loss.update_state(loss_fn(y, predictions, mask))
            val_accuracy.update_state(y_indices, predictions)
            val_top_k.update_state(y_indices, predictions)
            
            # Calculate legal move accuracy for validation
            legal_move_pred = predictions * tf.cast(mask, tf.float32)
            legal_move_pred = legal_move_pred / (tf.reduce_sum(legal_move_pred, axis=1, keepdims=True) + 1e-7)
            legal_move_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(legal_move_pred, axis=1), y_indices), tf.float32)
            )
            val_legal_move_accuracy.update_state(legal_move_accuracy)

        # Enhanced logging
        with writer.as_default():
            # Existing metrics
            tf.summary.scalar('train/loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train/accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('train/top_5_accuracy', train_top_k.result(), step=epoch)
            tf.summary.scalar('validation/loss', val_loss.result(), step=epoch)
            tf.summary.scalar('validation/accuracy', val_accuracy.result(), step=epoch)
            tf.summary.scalar('validation/top_5_accuracy', val_top_k.result(), step=epoch)
            
            # New metrics
            tf.summary.scalar('train/legal_move_accuracy', train_legal_move_accuracy.result(), step=epoch)
            tf.summary.scalar('validation/legal_move_accuracy', val_legal_move_accuracy.result(), step=epoch)
            tf.summary.scalar('learning_rate', lr_schedule(epoch * steps_per_epoch), step=epoch)
        
        # Early stopping check
        current_val_accuracy = val_accuracy.result()
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            patience_counter = 0
            # Save best model
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.h5'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

def main():
    # Configuration
    DATASET_SIZE =  20000000
    BATCH_SIZE = 1024
    EPOCHS = 20
    STEPS_PER_EPOCH = int(DATASET_SIZE / BATCH_SIZE)
    VALIDATION_STEPS = int((DATASET_SIZE*0.1) / BATCH_SIZE)

    # Paths
    tfrecord_dir = r"C:\Users\Matte\Desktop\temp_chess\e2e4\Train_Dataset\Large"  # Directory containing TFRecord files
    checkpoint_dir = "checkpoints"  # Directory to save model checkpoints
    
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Find all TFRecord files
    train_files = glob.glob(os.path.join(tfrecord_dir, "*_shard_*.tfrecord.gz"))
    if not train_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
    
    # Split into train and validation
    split_idx = int(len(train_files) * 0.9)  # 90% train, 10% validation
    train_files, val_files = train_files[:split_idx], train_files[split_idx:]
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_files, BATCH_SIZE)
    val_dataset = prepare_dataset(val_files, BATCH_SIZE)
    
    # Create model  
    model = EnhancedChessCNN()
    dummy_input = tf.random.normal((1, 8, 8, 15))
    _ = model(dummy_input)
    model.summary()

    # Learning rate schedule (Warmup + Cosine Decay)
    warmup_steps = STEPS_PER_EPOCH  # Warmup for one epoch
    total_steps = STEPS_PER_EPOCH * EPOCHS

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, 
        decay_steps=total_steps - warmup_steps
    )

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    loss_fn = create_legal_moves_mask_loss()
    
    # Train model
    try:
        train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            lr_schedule=lr_schedule

        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()