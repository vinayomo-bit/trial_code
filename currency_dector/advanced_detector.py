"""
Advanced Currency Note Detector with Improved Accuracy
This script provides an enhanced inference system for currency detection
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

class AdvancedCurrencyDetector:
    def __init__(self, model_path='currency_note_classifier.h5'):
        """
        Initialize the advanced currency detector
        """
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        
        # Updated class mapping based on actual training
        self.class_names = {
            0: '₹100',   # 1Hundrednote
            1: '₹200',   # 2Hundrednote
            2: '₹2000',  # 2Thousandnote
            3: '₹500',   # 5Hundrednote
            4: '₹50',    # Fiftynote
            5: '₹10',    # Tennote
            6: '₹20'     # Twentynote
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Model loaded: {self.model_path}")
                print(f"📊 Model input shape: {self.model.input_shape}")
                print(f"📊 Model output classes: {len(self.class_names)}")
            else:
                print(f"❌ Model not found: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing"""
        try:
            # Load image
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Cannot load image: {image_path}")
            else:
                img = image_path
            
            # Convert BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict_with_confidence(self, image_path, min_confidence=0.3):
        """
        Predict currency with confidence threshold
        """
        if self.model is None:
            print("❌ No model loaded")
            return None, None
        
        # Preprocess
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, None
        
        try:
            # Predict
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[0][predicted_class_idx]
            
            # Get currency
            predicted_currency = self.class_names.get(predicted_class_idx, "Unknown")
            
            # Check confidence threshold
            if confidence < min_confidence:
                print(f"⚠️ Low confidence prediction: {confidence:.2%}")
                print("Consider using a better image or retraining the model")
            
            return predicted_currency, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None
    
    def test_all_samples(self):
        """Test the model on all available test samples"""
        test_path = "indian-currency-notes-classifier/versions/1/Test"
        
        if not os.path.exists(test_path):
            print("❌ Test directory not found")
            return
        
        print("\n🧪 Testing model on all test samples...")
        print("=" * 60)
        
        total_tests = 0
        correct_predictions = 0
        class_stats = {}
        
        # Test each class
        for class_folder in os.listdir(test_path):
            class_path = os.path.join(test_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Get expected currency value
            expected_currency = self.get_expected_currency(class_folder)
            
            print(f"\n📁 Testing {class_folder} (Expected: {expected_currency})")
            print("-" * 40)
            
            class_correct = 0
            class_total = 0
            
            # Test images in this class
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            
            for img_file in image_files[:5]:  # Test first 5 images per class
                predicted_currency, confidence = self.predict_with_confidence(img_file)
                
                if predicted_currency:
                    total_tests += 1
                    class_total += 1
                    
                    is_correct = predicted_currency == expected_currency
                    if is_correct:
                        correct_predictions += 1
                        class_correct += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    filename = os.path.basename(img_file)
                    print(f"{status} {filename}: {predicted_currency} ({confidence:.1%})")
            
            # Store class statistics
            if class_total > 0:
                class_accuracy = class_correct / class_total
                class_stats[class_folder] = {
                    'accuracy': class_accuracy,
                    'correct': class_correct,
                    'total': class_total
                }
                print(f"📊 Class accuracy: {class_accuracy:.1%} ({class_correct}/{class_total})")
        
        # Overall results
        if total_tests > 0:
            overall_accuracy = correct_predictions / total_tests
            print(f"\n🎯 OVERALL RESULTS:")
            print("=" * 40)
            print(f"Total tests: {total_tests}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Overall accuracy: {overall_accuracy:.1%}")
            
            # Class-wise breakdown
            print(f"\n📊 Class-wise Performance:")
            for class_name, stats in class_stats.items():
                expected = self.get_expected_currency(class_name)
                print(f"{expected:<8}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        
        return overall_accuracy if total_tests > 0 else 0
    
    def get_expected_currency(self, class_folder):
        """Map class folder name to expected currency"""
        mapping = {
            '1Hundrednote': '₹100',
            '2Hundrednote': '₹200',
            '2Thousandnote': '₹2000',
            '5Hundrednote': '₹500',
            'Fiftynote': '₹50',
            'Tennote': '₹10',
            'Twentynote': '₹20'
        }
        return mapping.get(class_folder, class_folder)
    
    def predict_single_with_display(self, image_path):
        """Predict and display results for a single image"""
        predicted_currency, confidence = self.predict_with_confidence(image_path)
        
        if predicted_currency:
            # Load and display image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            
            # Color code based on confidence
            if confidence > 0.7:
                color = 'green'
                conf_level = 'High'
            elif confidence > 0.4:
                color = 'orange'
                conf_level = 'Medium'
            else:
                color = 'red'
                conf_level = 'Low'
            
            plt.title(f'Predicted: {predicted_currency}\n'
                     f'Confidence: {confidence:.1%} ({conf_level})', 
                     fontsize=16, fontweight='bold', color=color)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return predicted_currency, confidence
        
        return None, None

def main():
    """Main function for testing"""
    print("=" * 60)
    print("ADVANCED CURRENCY DETECTION TESTING")
    print("=" * 60)
    
    # Initialize detector
    detector = AdvancedCurrencyDetector()
    
    if detector.model is None:
        print("❌ Cannot proceed without a model")
        print("Please train a model first using one of the training scripts")
        return
    
    print(f"\n🎯 Advanced Currency Detector Ready!")
    
    # Test on all samples
    accuracy = detector.test_all_samples()
    
    print(f"\n📋 Model Assessment:")
    if accuracy > 0.8:
        print("🎉 Excellent model performance!")
    elif accuracy > 0.6:
        print("✅ Good model performance")
    elif accuracy > 0.4:
        print("⚠️ Moderate performance - consider retraining")
    else:
        print("❌ Poor performance - model needs retraining")
    
    # Test individual image if available
    test_folder = "indian-currency-notes-classifier/versions/1/Test"
    if os.path.exists(test_folder):
        # Find a sample image
        for currency_class in os.listdir(test_folder):
            class_path = os.path.join(test_folder, currency_class)
            if os.path.isdir(class_path):
                images = glob.glob(os.path.join(class_path, "*.jpg"))
                if images:
                    print(f"\n🧪 Testing individual image:")
                    detector.predict_single_with_display(images[0])
                    break

if __name__ == "__main__":
    main()
