# Facial-EmotionRecognition-Using-CNN

This project implements a Convolutional Neural Network (CNN) to classify facial expressions into seven categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**. It uses grayscale images and is capable of both batch image classification and real-time emotion detection using a webcam.

---

## 📁 Project Structure

```
.
├── data/
│   ├── train/        # Training images organized by class
│   └── test/         # Validation images organized by class
├── haarcascade_frontalface_default.xml
├── model_file_30epochs.h5         # Trained CNN model
├── main.py                        # Model training and evaluation
├── test.py                        # Real-time emotion detection via webcam
├── testdata.py                    # Emotion prediction on static images
├── Facial_Expression_Recognition_Report.pdf
└── README.md
```

---

## 🚀 Features

- CNN-based model trained on 48x48 grayscale images
- Real-time emotion detection using OpenCV and webcam
- Evaluation metrics: accuracy, loss curves, classification report, confusion matrix
- Data augmentation for better generalization
- Exported model for easy deployment

---

## 🧠 Model Architecture

- 4 Conv2D layers with ReLU activation + MaxPooling
- Dropout regularization (10–20%)
- Fully connected Dense layer
- Output layer with 7-way softmax classification
- Optimizer: Adam | Loss: Categorical Crossentropy

---

## 📊 Evaluation

- Training Accuracy: ~85–90%
- Validation Accuracy: ~75–80%
- Common misclassifications: *Neutral vs. Sad*, *Surprise vs. Fear*
- Visualized confusion matrix and per-class metrics using `sklearn`

---

## 🛠 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/kabirkohli123/Facial-EmotionRecognition-Using-CNN
.git
cd Facial-EmotionRecognition-Using-CNN
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python main.py
```

### 4. Run Real-time Emotion Detection
```bash
python test.py
```

### 5. Predict Emotion from Image
```bash
python testdata.py
```

---

## 📁 Dataset

This project assumes the dataset is organized as follows:
```
data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── ...
├── test/
    ├── Angry/
    ├── Disgust/
    ├── ...
```

Each folder should contain grayscale `.jpg` or `.png` images.

---

## 📄 Report

A detailed summary of the approach, findings, and challenges can be found in:
📄 [`Facial_Expression_Recognition_Report.pdf`](./Facial_Expression_Recognition_Report.pdf)

---

## 🤔 Future Improvements

- Use face alignment for better preprocessing
- Train deeper models like ResNet or EfficientNet
- Address class imbalance using weighted loss or oversampling
- Optimize for low-latency edge deployment

---

## 👨‍💻 Author

- **Your Name**  
  [GitHub](https://github.com/kabirkohli123) • [LinkedIn]([https://linkedin.com/in/your-link](https://www.linkedin.com/in/kabir-kohli-50965a259/))

