"""
Currency Note Detection - Inference Script
Use this script to detect currency notes in images using the trained model
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class CurrencyDetector:
    def __init__(self, model_path='currency_note_classifier.h5'):
        """
        Initialize the currency detector with a trained model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        
        # Currency note classes mapping (based on actual class indices)
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
                print(f"✅ Model loaded successfully from {self.model_path}")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("Please train the model first by running 'run_training.py'")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image_path, show_image=True):
        """
        Predict currency note for a given image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image with prediction
            
        Returns:
            Tuple of (predicted_currency, confidence)
        """
        if self.model is None:
            print("❌ No model loaded. Cannot make predictions.")
            return None, None
        
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, None
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[0][predicted_class_idx]
            
            # Get currency value
            predicted_currency = self.class_names.get(predicted_class_idx, "Unknown")
            
            # Display results
            print(f"\n🔍 Prediction Results:")
            print(f"💰 Currency Note: {predicted_currency}")
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Show image with prediction
            if show_image:
                self.display_prediction(image_path, predicted_currency, confidence)
            
            return predicted_currency, confidence
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None, None
    
    def display_prediction(self, image_path, predicted_currency, confidence):
        """Display image with prediction results"""
        try:
            # Load and display image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f'Predicted: {predicted_currency}\nConfidence: {confidence:.2%}', 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying image: {str(e)}")
    
    def predict_batch(self, image_folder):
        """
        Predict currency notes for all images in a folder
        
        Args:
            image_folder: Path to folder containing images
        """
        if not os.path.exists(image_folder):
            print(f"❌ Folder not found: {image_folder}")
            return
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ No image files found in {image_folder}")
            return
        
        print(f"\n🔍 Processing {len(image_files)} images...")
        results = []
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"\n📷 Processing: {image_file}")
            
            currency, confidence = self.predict(image_path, show_image=False)
            if currency:
                results.append({
                    'image': image_file,
                    'currency': currency,
                    'confidence': confidence
                })
        
        # Display summary
        print(f"\n📊 Batch Prediction Summary:")
        print("=" * 50)
        for result in results:
            print(f"{result['image']:<30} | {result['currency']:<8} | {result['confidence']:.2%}")
        
        return results

def main():
    """Main function for testing currency detection"""
    print("=" * 60)
    print("CURRENCY NOTE DETECTION - INFERENCE")
    print("=" * 60)
    
    # Initialize detector
    detector = CurrencyDetector()
    
    if detector.model is None:
        print("\n❌ Cannot proceed without a trained model.")
        print("Please run 'run_training.py' first to train the model.")
        return
    
    print("\n🎯 Currency Detection Ready!")
    print("\nAvailable options:")
    print("1. Test with sample image")
    print("2. Test with your own image")
    print("3. Test batch of images")
    
    # Example usage
    print("\n📋 Example usage:")
    print("detector.predict('path/to/your/image.jpg')")
    print("detector.predict_batch('path/to/image/folder')")
    
    # Test if sample images exist
    test_folder = "indian-currency-notes-classifier/versions/1/Test"
    if os.path.exists(test_folder):
        print(f"\n🧪 Test images available in: {test_folder}")
        
        # Find a sample image to test
        for currency_folder in os.listdir(test_folder):
            currency_path = os.path.join(test_folder, currency_folder)
            if os.path.isdir(currency_path):
                images = [f for f in os.listdir(currency_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_image = os.path.join(currency_path, images[0])
                    print(f"\n🧪 Testing with sample image: {sample_image}")
                    detector.predict(sample_image)
                    break

if __name__ == "__main__":
    main()
