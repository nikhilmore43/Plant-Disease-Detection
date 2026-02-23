from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = tf.keras.models.load_model("best_model.keras")

classes = ["Healthy", "Powdery", "Rust"]

CONFIDENCE_THRESHOLD = 70  # Prevent weak predictions


def predict_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return None, "Invalid image"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Slight blur to reduce artificial sharpness
        img = cv2.GaussianBlur(img, (3, 3), 0)

        img = cv2.resize(img, (299, 299))
        img = img.astype(np.float32)

        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img, verbose=0)[0]

        class_index = np.argmax(predictions)
        confidence = float(predictions[class_index]) * 100
        class_name = classes[class_index]

        probabilities = {
            classes[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(classes))
        }

        # Threshold logic
        if confidence < CONFIDENCE_THRESHOLD:
            class_name = "Uncertain"
        
        return {
            "label": class_name,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        }, None

    except Exception as e:
        return None, str(e)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
            file.save(image_path)

            result, error = predict_image(image_path)

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)