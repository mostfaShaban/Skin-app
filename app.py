from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the skin disease classification model during initialization
model = load_model(r"C:\Users\abdel\Desktop\Final_Try\skin_cancer_detection7.h5")

# Define the classes for skin diseases
classes = {
    0: ('Actinic keratoses and intraepithelial carcinomae', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('Basal cell carcinoma', 'Basal cell carcinoma'),
    2: ('Benign keratosis-like lesions', 'Benign keratosis-like lesions'),
    3: ('Dermatofibroma', 'Dermatofibroma'),
    4: ('Melanoma', 'Melanoma'),
    5: ('Melanocytic nevi', 'Melanocytic nevi'),
    6: ('Pyogenic granulomas and hemorrhage', 'Pyogenic granulomas and hemorrhage')
}

# Function to preprocess the image for skin disease classification
def preprocess_image(image):
    image = image.resize((90, 120))  # Resize the image to match the model's input shape
    image_array = np.array(image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize the pixel values
    return image_array

@app.route('/')
def index():
    return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'fileup' not in request.files:
        return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error='Please upload an image')
    
    file = request.files['fileup']
    if file.filename == '':
        return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error='Please upload a valid image')
    
    try:
        image = Image.open(file)
        image_array = preprocess_image(image)
        # Make predictions
        predictions = model.predict(np.array([image_array]))
        # Interpret the predictions
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = classes[predicted_class_index][0]
        predicted_class_description = classes[predicted_class_index][1]
        return render_template('index.html', appName="Skin Disease Detection", prediction=predicted_class_name, description=predicted_class_description, image=file)
    except Exception as e:
        return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
