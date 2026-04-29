#Denne har blitt laget ved hjelp av ChatGPT
import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "model.keras"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "adenocarcinoma",
    "large cell",
    "normal",
    "squamous"
]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)

    # MobileNetV2 preprocess
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    if image is None:
        return {"No image": 1.0}

    x = preprocess_image(image)
    preds = model.predict(x, verbose=0)[0]

    return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

title = "Lung CT Classification Demo"
description = (
    "Upload a lung CT image to get class probabilities from the trained transfer learning model "
    "exported from the notebook workflow. "
    "This is a student project demo and not for medical use."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload CT image"),
    outputs=gr.Label(num_top_classes=4, label="Predicted probabilities"),
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()
