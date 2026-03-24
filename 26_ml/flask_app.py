
# flask_app.py - Flask API for Housing Price Prediction
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model at startup (not per request - important for performance)
MODEL_PATH = 'housing_model_v1.joblib'
model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'housing_v1', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    Expects JSON with feature values.
    Returns predicted house price.
    """
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        logger.info(f"Received prediction request: {data}")

        # Convert to DataFrame (handles both single and batch)
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Batch prediction
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input format. Expected dict or list.'}), 400

        # Validate required features
        required_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude', 'ocean_proximity']
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Make prediction
        predictions = model.predict(df)

        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'model_version': 'v1',
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction successful: {response['predictions']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run development server (use gunicorn in production)
    app.run(host='0.0.0.0', port=5000, debug=True)
