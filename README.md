# AI-Powered Dermatology Skin Disease Detection System with Explainable AI (Grad-CAM)

## 📌 Project Overview

The AI-Powered Dermatology Skin Disease Detection System is a web-based application that uses Deep Learning to classify skin images into different categories. It integrates Explainable AI (XAI) using Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight the regions of an image that influence the model's prediction.

The system aims to support preliminary skin image analysis by providing predicted classifications, confidence scores, visual explanations, and relevant skincare guidance.

**Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis.

## 🎯 Objectives

- Develop a CNN-based skin image classification system.
- Identify skin image categories using image processing and deep learning.
- Generate Grad-CAM heatmaps to explain model predictions.
- Build a user-friendly web interface for uploading and analyzing images.
- Provide preliminary skincare guidance based on predicted categories.
- Develop a scalable backend for model inference and API communication.

## ✨ Features

- Skin image upload and analysis.
- CNN-based image classification.
- Prediction confidence scores.
- Grad-CAM visualization for model interpretability.
- Skincare guidance based on the predicted category.
- REST API integration using FastAPI.
- Interactive frontend for displaying results.

## 🧠 Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow / Keras |
| Model architecture | Convolutional Neural Network (CNN) |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | React, HTML, CSS, JavaScript |
| Image processing | Pillow / NumPy |
| API communication | REST API |

*Confirm these technologies against your actual implementation before submission.*

## 🏗️ System Architecture

1. **Image Upload:** The user uploads a skin image through the web interface.
2. **Preprocessing:** The image is resized, converted to the required format, and normalized.
3. **Model Prediction:** The trained CNN extracts image features and predicts a category.
4. **Confidence Score:** The system displays the model's prediction score.
5. **Grad-CAM Explanation:** A heatmap highlights image regions that contribute to the prediction.
6. **Result Display:** The frontend displays the predicted category, confidence score, heatmap, and relevant guidance.

## 🔬 CNN Model

A Convolutional Neural Network learns visual features from skin images.

- **Convolutional layers:** Extract features such as edges, textures, and patterns.
- **Activation functions:** Introduce non-linearity into the network.
- **Pooling layers:** Reduce feature-map dimensions.
- **Dense layers:** Use extracted features to classify the image.
- **Output layer:** Produces scores or probabilities for the supported categories.

The model's performance depends on the quality, diversity, and labeling of the training dataset.

## 🔍 Explainable AI with Grad-CAM

Grad-CAM stands for Gradient-weighted Class Activation Mapping. It uses gradients associated with a predicted class to create a heatmap showing which image regions influenced the model's output.

The heatmap is overlaid on the original image to help users understand the model's focus. It explains aspects of the model's behavior but does not prove that the prediction is medically correct.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- Node.js and npm
- Git
- A trained CNN model and its required files

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Set Up the Backend

Open a terminal in the backend directory.

```bash
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server using the actual entry-point file in your project. For example, if it is `main.py`:

```bash
uvicorn main:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

### 3. Set Up the Frontend

Open another terminal in the frontend directory.

```bash
npm install
npm run dev
```

Open the local URL displayed in the terminal.

*The commands and directory names may need adjustment to match your repository.*

## 📊 Dataset

The system requires a labeled dataset of skin images representing the categories supported by the model.

Document the following details in the final project version:

- Dataset name and source.
- Number of images in each category.
- Training, validation, and test split.
- Image preprocessing and augmentation techniques.
- Evaluation metrics, including accuracy, precision, recall, and F1-score.

Use only dataset details and performance values that you have verified experimentally.

## 📈 Evaluation

The model should be evaluated using a held-out test dataset. Recommended metrics include:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix
- Inference time

Grad-CAM visualizations can also be inspected to evaluate whether the model focuses on relevant image regions.

## 🔮 Future Enhancements

- Expand the dataset to include more diverse skin tones and image conditions.
- Improve model robustness and classification performance.
- Add secure user accounts and analysis history.
- Integrate additional image-quality checks.
- Evaluate explanations with dermatology experts.
- Explore privacy-preserving training methods.
- Conduct clinical validation before considering real-world medical use.

## 👩‍💻 Author

**Olivia Joyson**

B.Tech Computer Science and Engineering  
Karunya Institute of Technology and Sciences

## 📜 License

Specify an appropriate license before distributing or reusing this project.
