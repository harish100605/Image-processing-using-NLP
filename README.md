# Image-processing-using-NLP
# 👟 Shoe vs Slipper Image Classification using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project implements a **Binary Image Classification** model using **Convolutional Neural Networks (CNN)** with **TensorFlow** and **Keras**. The objective is to automatically classify footwear images into one of two categories:

* 👟 Shoe
* 🩴 Slipper

The model is trained on a custom image dataset stored in **Google Drive** and uses deep learning techniques to learn visual features such as shape, texture, and patterns for accurate classification.

---

## 🚀 Features

* Binary image classification using CNN
* Dataset loading directly from Google Drive
* Automatic training and validation dataset creation
* Multiple CNN architecture experiments
* Image preprocessing and normalization
* Model visualization using Keras
* Prediction on custom images
* Accuracy evaluation using validation data

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Google Colab
* Google Drive

---

## 📂 Dataset Structure

```
photos/
│
├── shoe/
│   ├── shoe1.jpg
│   ├── shoe2.jpg
│   └── ...
│
├── slipper/
│   ├── slipper1.jpg
│   ├── slipper2.jpg
│   └── ...
```

The dataset is automatically divided into:

* **90% Training**
* **10% Validation**

using TensorFlow's `image_dataset_from_directory()` function.

---

## 🧠 Model Architecture

The project experiments with different CNN architectures.

### Model Components

* Convolutional Layers (Conv2D)
* MaxPooling Layers
* Flatten Layer
* Fully Connected Dense Layers
* Batch Normalization
* Dropout Regularization
* Sigmoid Output Layer

The final output layer contains one neuron with **Sigmoid Activation**, making it suitable for binary classification.

---

## ⚙️ Workflow

1. Mount Google Drive
2. Load image dataset
3. Split dataset into training and validation sets
4. Build CNN model
5. Compile the model using Adam optimizer
6. Train for multiple epochs
7. Validate model performance
8. Predict footwear type from unseen images

---

## 📈 Model Training

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

Evaluation Metric:

* Accuracy

Training is performed for multiple epochs while monitoring validation accuracy.

---

## 🔍 Prediction

The trained model accepts an input image of size:

```
200 × 200 × 3
```

After preprocessing, the model predicts the probability of the image belonging to either:

* Shoe
* Slipper

Example Output:

```
Prediction: 0.93
Class: Shoe
```

or

```
Prediction: 0.08
Class: Slipper
```

---

## 📊 Deep Learning Concepts Used

* Image Classification
* Convolutional Neural Networks (CNN)
* Feature Extraction
* Max Pooling
* Binary Classification
* Batch Normalization
* Dropout Regularization
* Forward Propagation
* Backpropagation
* Model Evaluation

---

## 📁 Project Structure

```
Shoe-Slipper-Classification/
│
├── dataset/
│
├── images/
│
├── Shoe_Slipper_Classification.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🎯 Future Improvements

* Increase dataset size for better generalization.
* Apply data augmentation techniques.
* Use Transfer Learning (MobileNetV2, ResNet50, EfficientNet).
* Add confusion matrix and classification report.
* Deploy the model as a web application using Flask or Streamlit.
* Convert the model for mobile deployment using TensorFlow Lite.

---

## 📚 Learning Outcomes

Through this project, the following concepts were explored:

* Building CNN models from scratch
* Image preprocessing using TensorFlow
* Binary image classification
* Deep learning model training and evaluation
* Predicting classes from unseen images
* Working with Google Colab and Google Drive integration

---

## 📌 Conclusion

This project demonstrates the implementation of a Convolutional Neural Network for binary footwear classification. By leveraging TensorFlow and Keras, the model successfully learns distinguishing visual features between shoes and slippers, providing an effective introduction to image classification using deep learning.
