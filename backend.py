import io
import numpy as np
import keras
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Potato Disease Prediction API")

# Load model once at startup (update filename if needed)
MODEL = keras.models.load_model("potato_cnn_model.keras")

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

# Match your model's expected input dimensions: (256, 256) or (224, 224)
IMG_SIZE = (256, 256)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)

    # Uncomment below if your model does NOT have a built-in Rescaling(1./255) layer
    # img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    predictions = MODEL.predict(input_tensor)[0]
    predicted_idx = int(np.argmax(predictions))

    return {
        "class": CLASS_NAMES[predicted_idx],
        "confidence": round(float(predictions[predicted_idx]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
    }