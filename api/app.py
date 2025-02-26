import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model with the correct relative path
model_path = os.path.join(os.path.dirname(__file__), '../src/best_model.h5')
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Read CSV file directly from the uploaded file
            data = pd.read_csv(file, header=None)
            
            # Check the data format
            print(f"Uploaded data shape: {data.shape}")
            print(data.head())
            
            # Prepare data for the model (expand dimensions for CNN input)
            X_input = np.expand_dims(data.values, axis=-1)  # Add channel dimension
            
            # Make predictions using the pre-loaded model
            predictions = model.predict(X_input)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Return the predictions as JSON
            return jsonify({'predicted_classes': predicted_classes.tolist()}), 200
        
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
