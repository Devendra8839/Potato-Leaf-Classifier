# Potato Leaf Disease Classifier

A deep learning-based application for classifying potato leaf diseases using a Convolutional Neural Network (CNN). This project provides both a FastAPI backend for model inference and a Streamlit frontend for user interaction.

## Features

- **Disease Classification**: Identifies three classes of potato leaves:
  - Early blight
  - Late blight
  - Healthy
- **FastAPI Backend**: RESTful API for model predictions
- **Streamlit Frontend**: Interactive web interface for easy image upload and classification
- **Pre-trained Model**: Keras model trained on potato leaf images

## Project Structure

```
cnn_app/
├── backend.py          # FastAPI application for model inference
├── frontend.py         # Streamlit web interface
├── potato_cnn_model.keras  # Pre-trained CNN model
├── leaf.JPG            # Sample potato leaf image
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.8+
- pip or conda package manager

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Devendra8839/Potato-Leaf-Classifier.git
   cd cnn_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn streamlit pillow numpy keras tensorflow
   ```

## Usage

### Running the Backend API

Start the FastAPI server:
```bash
uvicorn backend:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

**API Endpoints:**
- `POST /predict` - Upload an image to get disease prediction

**Example API Request:**
```bash
curl -X POST -F "file=@leaf.JPG" http://127.0.0.1:8000/predict
```

### Running the Frontend

In a separate terminal, start the Streamlit application:
```bash
streamlit run frontend.py
```

The web interface will be available at `http://localhost:8501`.

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8501`
2. Upload a potato leaf image (JPG, JPEG, or PNG format)
3. Click "Predict Disease" button
4. View the prediction results with confidence scores

## Model Details

- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: Keras/TensorFlow
- **Input Size**: 256x256 pixels
- **Classes**: 3 (Early blight, Late blight, Healthy)
- **Model File**: `potato_cnn_model.keras`

## Example Prediction

When you upload a leaf image, the system will return:
- The predicted disease class
- Confidence percentage
- Probability distribution across all classes
