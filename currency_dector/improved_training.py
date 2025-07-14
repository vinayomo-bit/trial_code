"""
Improved training script for currency note detection
This version addresses overfitting and low accuracy issues
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

class ImprovedCurrencyClassifier:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=16):
        """
        Initialize the improved currency classifier using transfer learning
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        print(f"Improved Currency Note Classifier")
        print(f"Using Transfer Learning with MobileNetV2")
        print(f"Image size: {img_size}, Batch size: {batch_size}")
    
    def setup_data_generators(self):
        """Setup improved data generators with better augmentation"""
        
        # More conservative augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Currency notes shouldn't be flipped
            brightness_range=[0.9, 1.1],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'Train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'Train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Test generator
        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'Test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = len(self.train_generator.class_indices)
        print(f"\nDataset loaded:")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
    
    def build_transfer_learning_model(self):
        """Build model using transfer learning with MobileNetV2"""
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nTransfer Learning Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Print summary of custom layers only
        print("\nCustom layers summary:")
        for i, layer in enumerate(self.model.layers[-6:]):
            print(f"{layer.name}: {layer.output_shape}")
    
    def train_model(self, epochs=50):
        """Train with improved strategy"""
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                'improved_currency_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nStarting training for {epochs} epochs...")
        
        # Calculate steps
        steps_per_epoch = max(1, self.train_generator.samples // self.batch_size)
        validation_steps = max(1, self.validation_generator.samples // self.batch_size)
        
        # Phase 1: Train with frozen base
        print("\n=== Phase 1: Training with frozen base model ===")
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=min(20, epochs),
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune top layers
        if epochs > 20:
            print("\n=== Phase 2: Fine-tuning with unfrozen top layers ===")
            
            # Unfreeze top layers of base model
            base_model = self.model.layers[0]
            base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = 100
            
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training
            history2 = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                initial_epoch=len(history1.history['loss']),
                validation_data=self.validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = self.combine_histories(history1, history2)
        else:
            self.history = history1
        
        print(f"\nTraining completed!")
    
    def combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def evaluate_and_save(self):
        """Evaluate model and save results"""
        
        print("\n=== Model Evaluation ===")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"\nFinal Test Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Plot training history
        if self.history:
            self.plot_training_history()
        
        # Save the model
        self.model.save('improved_currency_classifier.h5')
        print(f"\nModel saved as 'improved_currency_classifier.h5'")
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    
    print("=" * 70)
    print("IMPROVED CURRENCY NOTE CLASSIFIER TRAINING")
    print("Using Transfer Learning with MobileNetV2")
    print("=" * 70)
    
    dataset_path = "indian-currency-notes-classifier/versions/1"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Initialize classifier
    classifier = ImprovedCurrencyClassifier(
        dataset_path=dataset_path,
        img_size=(224, 224),
        batch_size=16
    )
    
    # Setup data
    classifier.setup_data_generators()
    
    # Build model
    classifier.build_transfer_learning_model()
    
    # Train model
    classifier.train_model(epochs=40)
    
    # Evaluate and save
    accuracy = classifier.evaluate_and_save()
    
    print(f"\n" + "=" * 70)
    print(f"✅ IMPROVED TRAINING COMPLETED!")
    print(f"📊 Final Accuracy: {accuracy:.2%}")
    print(f"📁 Model saved as: improved_currency_classifier.h5")
    print("=" * 70)

if __name__ == "__main__":
    main()
