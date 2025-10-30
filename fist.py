from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model once (not every time)
model = tf.keras.models.load_model("eye_disease_model.h5")
class_names = ["normal", "cataract", "glaucoma", "diabetic_retinopathy"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).resize((150,150))
            img_array = np.expand_dims(np.array(img)/255.0, axis=0)
            pred = model.predict(img_array)
            class_idx = np.argmax(pred)
            confidence = np.max(pred)
            return f"Prediction: {class_names[class_idx]} ({confidence*100:.2f}%)"
    return render_template("model.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
