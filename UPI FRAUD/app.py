from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and encoders
xgb_model = joblib.load('xgb_model.sav')
lr_model = joblib.load('lr_model.sav')
type_encoder = joblib.load('type_encoder.sav')
nameOrig_encoder = joblib.load('nameOrig_encoder.sav')
nameDest_encoder = joblib.load('nameDest_encoder.sav')

def safe_transform(encoder, label):
    try:
        return encoder.transform([label])[0]
    except ValueError:
        return -1  # Unknown label fallback

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        step = int(request.form['step'])
        tx_type = request.form['type'].lower()
        amount = float(request.form['amount'])
        nameOrig = request.form['nameOrig']
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = request.form['nameDest']
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Encode features
        type_encoded = safe_transform(type_encoder, tx_type)
        nameOrig_encoded = safe_transform(nameOrig_encoder, nameOrig)
        nameDest_encoded = safe_transform(nameDest_encoder, nameDest)

        # Feature vector
        features = np.array([[step, type_encoded, amount, nameOrig_encoded,
                              oldbalanceOrg, newbalanceOrig, nameDest_encoded,
                              oldbalanceDest, newbalanceDest]])

        # Predict
        prediction = xgb_model.predict(features)[0]
        label = "Fraud" if prediction == 1 else "Not Fraud"
        return render_template('index.html', prediction_text=f"Prediction: {label}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")
    

if __name__ == '__main__':
    app.run(debug=True)