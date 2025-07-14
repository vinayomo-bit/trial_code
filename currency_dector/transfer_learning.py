"""
Transfer Learning Currency Note Classifier
Using MobileNetV2 pre-trained on ImageNet for better accuracy
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_transfer_learning_model(num_classes, img_size=(224, 224)):
    """Create transfer learning model using MobileNetV2"""
    
    print("Building transfer learning model...")
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    print(f"Base model loaded: {len(base_model.layers)} layers")
    print(f"Trainable parameters: {base_model.count_params():,}")
    
    # Add custom classification layers
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='feature_dense')(x)
    x = Dropout(0.3, name='feature_dropout')(x)
    x = Dense(128, activation='relu', name='classifier_dense')(x)
    x = Dropout(0.2, name='classifier_dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the complete model
    model = Model(inputs, outputs)
    
    print(f"Complete model created with {model.count_params():,} parameters")
    return model, base_model

def prepare_datasets(dataset_path, img_size=(224, 224), batch_size=32):
    """Prepare datasets using tf.keras.utils.image_dataset_from_directory"""
    
    print("Preparing datasets...")
    
    # Training dataset with validation split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, "Train"),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, "Train"),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, "Test"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    print(f"Classes found: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Calculate dataset sizes
    train_size = tf.data.experimental.cardinality(train_ds).numpy()
    val_size = tf.data.experimental.cardinality(val_ds).numpy()
    test_size = tf.data.experimental.cardinality(test_ds).numpy()
    
    print(f"Training batches: {train_size}")
    print(f"Validation batches: {val_size}")
    print(f"Test batches: {test_size}")
    
    return train_ds, val_ds, test_ds, class_names, num_classes

def preprocess_datasets(train_ds, val_ds, test_ds):
    """Apply preprocessing and augmentation"""
    
    print("Applying data preprocessing...")
    
    # Normalization layer (will be applied to all datasets)
    normalization_layer = tf.keras.utils.experimental.preprocessing.Rescaling(1./255)
    
    # Data augmentation (only for training)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ])
    
    # Apply to training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Apply only normalization to validation and test
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def train_transfer_learning_model():
    """Complete transfer learning training pipeline"""
    
    print("=" * 70)
    print("TRANSFER LEARNING CURRENCY NOTE DETECTION")
    print("Using MobileNetV2 Pre-trained on ImageNet")
    print("=" * 70)
    
    # Configuration
    dataset_path = "indian-currency-notes-classifier/versions/1"
    img_size = (224, 224)
    batch_size = 16  # Smaller batch size for stability
    
    # Check dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    try:
        # Prepare datasets
        train_ds, val_ds, test_ds, class_names, num_classes = prepare_datasets(
            dataset_path, img_size, batch_size
        )
        
        # Preprocess datasets
        train_ds, val_ds, test_ds = preprocess_datasets(train_ds, val_ds, test_ds)
        
        # Create model
        model, base_model = create_transfer_learning_model(num_classes, img_size)
        
        # Compile model for initial training
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n📋 Model Summary:")
        model.summary()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                'transfer_learning_currency_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n🚀 Phase 1: Training with frozen base model...")
        print("-" * 50)
        
        # Phase 1: Train with frozen base model
        initial_epochs = 20
        start_time = datetime.now()
        
        history1 = model.fit(
            train_ds,
            epochs=initial_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        phase1_time = datetime.now() - start_time
        print(f"Phase 1 completed in: {phase1_time}")
        
        # Evaluate after phase 1
        print("\n📊 Phase 1 Results:")
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Phase 2: Fine-tune if performance is reasonable
        if val_accuracy > 0.3:  # Only fine-tune if we have some learning
            print("\n🔧 Phase 2: Fine-tuning with unfrozen layers...")
            print("-" * 50)
            
            # Unfreeze the top layers of the base model
            base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = 100
            
            # Freeze all layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            print(f"Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
            
            # Recompile with lower learning rate for fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=0.0001/10),  # Much lower learning rate
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training
            fine_tune_epochs = 15
            total_epochs = initial_epochs + fine_tune_epochs
            
            start_time = datetime.now()
            history2 = model.fit(
                train_ds,
                epochs=total_epochs,
                initial_epoch=history1.epoch[-1],
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1
            )
            
            phase2_time = datetime.now() - start_time
            print(f"Phase 2 completed in: {phase2_time}")
            
            # Combine histories
            history = combine_histories(history1, history2)
        else:
            print("⚠️ Skipping fine-tuning due to low initial performance")
            history = history1
        
        # Final evaluation on test set
        print("\n🎯 Final Evaluation on Test Set:")
        print("-" * 40)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        
        print(f"\n📊 Final Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Save the final model
        final_model_path = 'transfer_learning_currency_final.h5'
        model.save(final_model_path)
        print(f"💾 Model saved: {final_model_path}")
        
        # Plot training history
        plot_training_history(history)
        
        # Test predictions on sample images
        test_sample_predictions(model, test_ds, class_names)
        
        print("\n✅ Transfer Learning Training Completed!")
        print("=" * 70)
        
        return model, history, test_accuracy
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def combine_histories(hist1, hist2):
    """Combine training histories from two phases"""
    combined_history = {}
    for key in hist1.history.keys():
        combined_history[key] = hist1.history[key] + hist2.history[key]
    
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return CombinedHistory(combined_history)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_sample_predictions(model, test_ds, class_names):
    """Test model on sample images and show predictions"""
    print("\n🧪 Sample Predictions:")
    print("-" * 40)
    
    # Get a batch of test images
    for images, labels in test_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        # Show predictions for first 5 images
        for i in range(min(5, len(images))):
            predicted_class = np.argmax(predictions[i])
            confidence = predictions[i][predicted_class]
            true_class = labels[i].numpy()
            
            true_label = class_names[true_class]
            predicted_label = class_names[predicted_class]
            
            status = "✅" if predicted_class == true_class else "❌"
            print(f"{status} True: {true_label:<15} | Predicted: {predicted_label:<15} | Confidence: {confidence:.2%}")
        
        break

if __name__ == "__main__":
    # Run transfer learning training
    result = train_transfer_learning_model()
    
    if result:
        model, history, accuracy = result
        print(f"\n🎉 Training successful! Final accuracy: {accuracy:.2%}")
    else:
        print("\n❌ Training failed!")
