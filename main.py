from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import io  # Importing StringIO from the io module

app = FastAPI()

# Load the pre-existing model and scaler
MODEL_PATH = "models/weather_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Pre-trained model or scaler file is missing.")

model = joblib.load(MODEL_PATH)  # Load pre-existing model
scaler = joblib.load(SCALER_PATH)  # Load pre-existing scaler

# Initialize Label Encoder
label_encoder = LabelEncoder()

# Request Body for Prediction
class PredictionRequest(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float

@app.post("/predict")
async def predict_rain(data: PredictionRequest):
    """
    Predict whether it will rain or not based on input features.
    """
    try:
        # Create feature array
        features = np.array([
            data.Temperature, 
            data.Humidity, 
            data.Wind_Speed, 
            data.Cloud_Cover, 
            data.Pressure
        ]).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_label = "Rain" if prediction[0] == 1 else "No Rain"
        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """
    Retrain the model using the uploaded CSV file, keeping the existing model's learned features.
    """
    try:
        # Load the new training data from the uploaded CSV file
        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode()))  # Use io.StringIO here

        # Check if required columns exist
        required_columns = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure", "Rain"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")

        # Encode the Rain column if it contains strings
        if data["Rain"].dtype == 'object':
            data["Rain"] = label_encoder.fit_transform(data["Rain"])

        # Separate features and target
        X = data[["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]]
        y = data["Rain"]

        # Split the new data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Retrain the model with the new data
        model.fit(X_train_scaled, y_train)

        ## Use evaluate method for Keras models to get loss and accuracy
        val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)

        # Save the updated model and scaler
        joblib.dump(model, MODEL_PATH)  # Save the updated model
        joblib.dump(scaler, SCALER_PATH)  # Save the updated scaler

        return {"message": "Model retrained successfully.", "validation_accuracy": val_accuracy}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in retraining: {e}")
