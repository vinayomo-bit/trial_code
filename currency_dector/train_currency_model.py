import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
from datetime import datetime

class CurrencyNoteClassifier:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32):
        """
        Initialize the Currency Note Classifier
        
        Args:
            dataset_path: Path to the dataset directory containing Train and Test folders
            img_size: Target image size for training (width, height)
            batch_size: Batch size for training
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Currency note classes mapping
        self.class_names = {
            '1Hundrednote': '₹100',
            '2Hundrednote': '₹200', 
            '2Thousandnote': '₹2000',
            '5Hundrednote': '₹500',
            'Fiftynote': '₹50',
            'Tennote': '₹10',
            'Twentynote': '₹20'
        }
        
        print(f"Initialized Currency Note Classifier")
        print(f"Image size: {img_size}")
        print(f"Batch size: {batch_size}")
        print(f"Classes: {list(self.class_names.values())}")
    
    def setup_data_generators(self):
        """Setup data generators with augmentation for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2  # Use 20% for validation
        )
        
        # Only rescaling for test data
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
        print(f"\nDataset Info:")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class indices: {self.train_generator.class_indices}")
    
    def build_model(self):
        """Build a CNN model for currency note classification"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fifth Convolutional Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
    
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
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
                'best_currency_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nStarting training for {epochs} epochs...")
        start_time = datetime.now()
        
        # Calculate steps per epoch
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.validation_generator.samples // self.batch_size
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"\nTraining completed in: {training_time}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        if self.model is None:
            print("No model available. Train the model first.")
            return
        
        print("\nEvaluating model on test data...")
        
        # Reset test generator
        self.test_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        # Plot confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and print accuracy
        test_accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        return test_accuracy
    
    def save_model(self, filepath='currency_note_classifier.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        print(f"Model saved as {filepath}")
    
    def predict_single_image(self, image_path):
        """Predict currency note for a single image"""
        if self.model is None:
            print("No model available. Train the model first.")
            return
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]
        
        # Get class name
        class_labels = list(self.train_generator.class_indices.keys())
        predicted_class = class_labels[predicted_class_idx]
        currency_value = self.class_names.get(predicted_class, predicted_class)
        
        print(f"Predicted: {currency_value}")
        print(f"Confidence: {confidence:.4f}")
        
        return currency_value, confidence

def main():
    """Main function to train the currency note classifier"""
    
    # Dataset path - adjust this to your dataset location
    dataset_path = "indian-currency-notes-classifier/versions/1"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please make sure the dataset is downloaded and the path is correct.")
        return
    
    # Initialize classifier
    classifier = CurrencyNoteClassifier(
        dataset_path=dataset_path,
        img_size=(224, 224),
        batch_size=32
    )
    
    # Setup data generators
    classifier.setup_data_generators()
    
    # Build model
    classifier.build_model()
    
    # Train model
    classifier.train_model(epochs=50)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    classifier.evaluate_model()
    
    # Save model
    classifier.save_model()
    
    print("\nTraining completed successfully!")
    print("Model saved as 'currency_note_classifier.h5'")
    print("Training plots saved as 'training_history.png' and 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
