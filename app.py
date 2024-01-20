from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # Initialize result variable
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Read the image directly from the file object
        in_memory_file = io.BytesIO()
        f.save(in_memory_file)
        in_memory_file.seek(0)
        file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        labels = ['50', '100', '200', '500']          # Modify as per your labels
        result = labels[pred_class[0]]
        
    return render_template('index.html', result=result)


def model_predict(img, model):
    img = cv2.resize(img, (128, 128)) # Resize to the input shape expected by your model
    img = img / 255.0
    img = np.array(img).reshape(-1, 128, 128, 1)

    preds = model.predict(img)
    return preds


if __name__ == '__main__':
    app.run(debug=True)
