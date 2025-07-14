# Currency Note Detection System

This project implements a deep learning model to detect and classify Indian currency notes using Convolutional Neural Networks (CNN).

## Features

- **Multi-class Classification**: Detects 7 types of Indian currency notes:
  - ₹10 (Ten rupees)
  - ₹20 (Twenty rupees)
  - ₹50 (Fifty rupees)
  - ₹100 (One hundred rupees)
  - ₹200 (Two hundred rupees)
  - ₹500 (Five hundred rupees)
  - ₹2000 (Two thousand rupees)

- **Deep CNN Architecture**: Custom CNN model with multiple layers and data augmentation
- **High Accuracy**: Trained with data augmentation for robust performance
- **Easy to Use**: Simple scripts for training and inference
- **Visualization**: Training plots and confusion matrix for model evaluation

## Project Structure

```
currency_dector/
├── dataset.py                    # Dataset download script
├── train_currency_model.py       # Main training module
├── run_training.py              # Simple training script
├── currency_detector.py         # Inference script
├── requirements.txt             # Package dependencies
├── README.md                    # This file
└── indian-currency-notes-classifier/  # Dataset folder
    └── versions/1/
        ├── Train/               # Training images
        └── Test/                # Test images
```

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (if not already downloaded):
   ```bash
   python dataset.py
   ```

## Usage

### Training the Model

1. **Run the training script**:
   ```bash
   python run_training.py
   ```

2. **Training will**:
   - Load and preprocess the dataset
   - Create data augmentation
   - Build the CNN model
   - Train for 30 epochs with early stopping
   - Generate training plots
   - Evaluate on test data
   - Save the trained model

3. **Output files**:
   - `currency_note_classifier.h5` - Final trained model
   - `best_currency_model.h5` - Best model checkpoint
   - `training_history.png` - Training accuracy/loss plots
   - `confusion_matrix.png` - Model evaluation results

### Using the Trained Model

1. **For single image prediction**:
   ```python
   from currency_detector import CurrencyDetector
   
   detector = CurrencyDetector()
   currency, confidence = detector.predict('path/to/image.jpg')
   ```

2. **For batch prediction**:
   ```python
   detector = CurrencyDetector()
   results = detector.predict_batch('path/to/image/folder')
   ```

3. **Run the inference script**:
   ```bash
   python currency_detector.py
   ```

## Model Architecture

The CNN model consists of:
- 5 Convolutional blocks with BatchNormalization and MaxPooling
- 2 Dense layers with Dropout for regularization
- Softmax output for 7-class classification
- Adam optimizer with learning rate scheduling
- Early stopping and model checkpointing

## Data Augmentation

Training includes:
- Rotation (±20°)
- Width/Height shift (±20%)
- Shear transformation
- Zoom (±20%)
- Horizontal flip
- Brightness variation
- 80/20 train/validation split

## Performance

The model typically achieves:
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Test Accuracy**: 85%+

*Actual performance may vary based on dataset quality and training parameters.*

## Troubleshooting

### Common Issues:

1. **GPU Memory Error**:
   - Reduce batch_size in the training script
   - Use CPU training if GPU memory is insufficient

2. **Import Errors**:
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (Python 3.7+)

3. **Dataset Not Found**:
   - Ensure the dataset is downloaded by running `python dataset.py`
   - Check the dataset path in the training script

4. **Poor Performance**:
   - Increase training epochs
   - Adjust learning rate
   - Add more data augmentation

## Customization

### Modify Training Parameters:

In `run_training.py`, you can adjust:
- `img_size`: Input image dimensions (default: 224x224)
- `batch_size`: Training batch size (default: 16)
- `epochs`: Number of training epochs (default: 30)

### Model Architecture:

Modify the `build_model()` function in `train_currency_model.py` to:
- Add/remove layers
- Change filter sizes
- Adjust dropout rates
- Try different optimizers

## Future Enhancements

- Real-time detection using webcam
- Mobile app integration
- Support for damaged/torn notes
- Multi-currency support
- Object detection for note localization

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.5+
- 4GB+ RAM (8GB+ recommended)
- GPU (optional, for faster training)

## License

This project is for educational purposes. Please ensure compliance with local regulations when using currency detection systems.

## Contributing

Feel free to contribute by:
- Improving model architecture
- Adding new features
- Enhancing documentation
- Reporting issues

---

**Happy Currency Detection! 💰🔍**
