# 🎨 MNIST CNN Digit Classifier & GUI


## 💡 Project Overview

A handwritten digit recognition project using CNN with precomputed data augmentation, combined with a Tkinter GUI to draw and predict digits in real-time.

Accuracy: ~99% on MNIST

GUI: Draw digits on a 280x280 canvas and predict instantly

Training: Uses precomputed augmentation for faster training
## GUI Screenshot

![](images/guiss.jpg)


## 🎬 Demo
![](images/demo1ss.jpg)
![](images/demo2ss.jpg)

## 🚀 Features
**1) 🧠 CNN Model**

Two convolutional blocks with BatchNorm, MaxPooling, and Dropout

Dense layer with 256 neurons + BatchNorm & Dropout

Softmax output layer (10 classes)

**✨2) Precomputed Data Augmentation**

Random rotation, width/height shift, zoom, shear

Doubles dataset without slowing training

**🖌3) Tkinter GUI**

Canvas: Draw digits (0–9) on 280x280 canvas

Buttons: Predict, Clear, Show probs, Quit

Prediction: Displays top-3 probabilities and confidence

**📦4) Dependencies**

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

Pillow

Tkinter

**Install with:**
```bash
pip install tensorflow numpy matplotlib pillow
```

## 🏗 Usage
**1️⃣ Train the Model**
```bash
python mnist_cnn_augmented.py
```

Saves model to: models/mnist_cnn_best.h5

Saves training curves: models/training_curves.png

**2️⃣ Run GUI**
```bash
python mnist_draw_predict.py
```

Draw digits on the canvas and click Predict

See top-3 predictions and confidence

## 📁 Project Structure
.
```bash
├── mnist_cnn_augmented.py    # Training script with precomputed augmentation
├── mnist_draw_predict.py     # GUI for drawing & predicting digits
├── models/
│   ├── mnist_cnn_best.h5     # Trained CNN model
│   └── training_curves.png   # Loss & accuracy plots
├── mnist_gui_demo.gif        # Animated demo of drawing & predicting
├── mnist_gui_screenshot.png  # Screenshot of GUI
└── README.md                 # Documentation
```

## 📊 Results

Test Accuracy: ~99%

Faster training with precomputed augmentation

Real-time GUI predictions with top-3 probabilities

Training Curves

## 🛠 Optional Improvements

Auto-center digits using center-of-mass

Thresholding to remove stray pixels

Multi-augmented dataset for even higher accuracy (~99.3%)

Keyboard shortcuts and dynamic canvas resizing
  
