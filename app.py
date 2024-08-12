from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

app = Flask(__name__)

# Load models
cnn_model = pickle.load(open('cnn_model.pkl', 'rb'))
transfer_model = pickle.load(open('transfer_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
gb_model = pickle.load(open('gb_model.pkl', 'rb'))

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # Preprocess the image
        img_array = preprocess_image(temp_path)
        
        # Make predictions using different models
        predictions = {
            'CNN': 'Pneumonia' if cnn_model.predict(img_array)[0][0] > 0.5 else 'Normal',
            'Transfer Learning': 'Pneumonia' if transfer_model.predict(img_array)[0][0] > 0.5 else 'Normal',
            'Random Forest': 'Pneumonia' if rf_model.predict(img_array.reshape(1, -1))[0] > 0.5 else 'Normal',
            'SVM': 'Pneumonia' if svm_model.predict(img_array.reshape(1, -1))[0] > 0.5 else 'Normal',
            'Gradient Boosting': 'Pneumonia' if gb_model.predict(img_array.reshape(1, -1))[0] > 0.5 else 'Normal'
        }
        
        return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)