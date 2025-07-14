"""
Test script to evaluate the trained currency model and check class mappings
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test_model():
    """Test the trained model and show detailed results"""
    
    print("=" * 60)
    print("CURRENCY MODEL TESTING AND EVALUATION")
    print("=" * 60)
    
    # Load the model
    model_path = 'currency_note_classifier.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
    
    # Show model summary
    print("\n📊 Model Summary:")
    model.summary()
    
    # Test data generator to get class mappings
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'indian-currency-notes-classifier/versions/1/Test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n📋 Class Mapping:")
    print("-" * 40)
    for class_name, class_index in test_generator.class_indices.items():
        print(f"{class_index}: {class_name}")
    
    # Evaluate on test data
    print(f"\n🔍 Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\n📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions for detailed analysis
    print(f"\n🔍 Getting detailed predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Show some sample predictions
    print(f"\n📋 Sample Predictions:")
    print("-" * 50)
    print(f"{'True':<15} {'Predicted':<15} {'Confidence':<12}")
    print("-" * 50)
    
    class_names = list(test_generator.class_indices.keys())
    for i in range(min(20, len(predictions))):
        true_class = class_names[true_classes[i]]
        pred_class = class_names[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]]
        print(f"{true_class:<15} {pred_class:<15} {confidence:.2%}")
    
    # Calculate per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(class_names):
        mask = true_classes == class_idx
        if np.sum(mask) > 0:
            class_accuracy = np.sum(predicted_classes[mask] == class_idx) / np.sum(mask)
            print(f"{class_name:<15}: {class_accuracy:.2%}")
    
    return model, test_generator

if __name__ == "__main__":
    model, test_gen = test_model()
