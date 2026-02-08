from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

# Load model
try:
    model = joblib.load('models/model.pkl')
    print("Model loaded.")
except FileNotFoundError:
    print("Model not found. Run src/train_model.py first.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    try:
        # Pull data from form or JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Match training feature names
        # Features: Age, BMI, BloodPressure, Cholesterol, Glucose, Gender
        
        # Map incoming fields to model features
        input_data = {
            'Age': float(data['age']),
            'BMI': float(data['bmi']),
            'BloodPressure': float(data['bp']),
            'Cholesterol': float(data['cholesterol']),
            'Glucose': float(data['glucose']),
            'Gender': int(data['gender'])
        }
        
        df = pd.DataFrame([input_data])
        
        # Prediction
        prediction = model.predict(df)[0]
        prediction_prob = model.predict_proba(df).max()
        
        # Map prediction to label
        labels = {0: "Healthy", 1: "Diabetes Risk", 2: "Heart Disease Risk"}
        result = labels.get(prediction, "Unknown")
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        response = {
            'prediction': result,
            'probability': f"{prediction_prob*100:.2f}%",
            'latency_ms': f"{latency_ms:.2f} ms"
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('index.html', prediction_text=f"Result: {result} (Confidence: {response['probability']})", latency=f"Latency: {response['latency_ms']}")
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
