"""
Simple script to train the currency note detection model
Run this script to start training the model
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_currency_model import CurrencyNoteClassifier

def train_currency_model():
    """Train the currency note detection model"""
    
    print("=" * 60)
    print("CURRENCY NOTE DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Dataset path - update this if your dataset is in a different location
    dataset_path = "indian-currency-notes-classifier/versions/1"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nPlease ensure:")
        print("1. The dataset has been downloaded")
        print("2. The path is correct")
        print("3. The dataset contains 'Train' and 'Test' folders")
        return False
    
    try:
        print("✅ Dataset found!")
        print(f"📁 Dataset location: {os.path.abspath(dataset_path)}")
        
        # Initialize the classifier
        print("\n🔧 Initializing Currency Note Classifier...")
        classifier = CurrencyNoteClassifier(
            dataset_path=dataset_path,
            img_size=(224, 224),  # Standard image size for training
            batch_size=16         # Reduced batch size for better compatibility
        )
        
        # Setup data generators
        print("\n📊 Setting up data generators...")
        classifier.setup_data_generators()
        
        # Build the model
        print("\n🏗️ Building the model...")
        classifier.build_model()
        
        # Start training
        print("\n🚀 Starting model training...")
        print("This may take a while depending on your hardware...")
        classifier.train_model(epochs=30)  # Reduced epochs for faster training
        
        # Plot training results
        print("\n📈 Generating training plots...")
        classifier.plot_training_history()
        
        # Evaluate the model
        print("\n🔍 Evaluating model performance...")
        accuracy = classifier.evaluate_model()
        
        # Save the model
        print("\n💾 Saving the trained model...")
        classifier.save_model('currency_note_classifier.h5')
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final Test Accuracy: {accuracy:.2%}")
        print("📁 Files generated:")
        print("   - currency_note_classifier.h5 (trained model)")
        print("   - best_currency_model.h5 (best model checkpoint)")
        print("   - training_history.png (training plots)")
        print("   - confusion_matrix.png (evaluation results)")
        print("\n🎉 Your currency note detection model is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have sufficient GPU memory (or use CPU)")
        print("2. Try reducing batch_size or image_size")
        print("3. Check that all required packages are installed")
        return False

if __name__ == "__main__":
    success = train_currency_model()
    if success:
        print("\n🚀 Ready to detect currency notes!")
    else:
        print("\n❌ Training failed. Please check the errors above.")
