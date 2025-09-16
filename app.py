from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from crop_predictor import CropPredictor
from disease_predictor import DiseasePredictor
import os
import pandas as pd

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Initialize the predictors
predictor = CropPredictor()
disease_predictor = DiseasePredictor(dataset_dir='Dataset')

# Load dataset and models
dataset_path = 'Crop_recommendation.csv'
model_path = 'crop_model.joblib'
plant_disease_model_path = 'disease_model.joblib'

# Load dataset first
if os.path.exists(dataset_path):
    predictor.load_data(dataset_path)
    print(f"Dataset loaded successfully from: {dataset_path}")
else:
    print(f"Warning: Dataset file {dataset_path} not found.")

# Try to load the pre-trained model
if os.path.exists(model_path):
    predictor.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
else:
    print(f"Warning: Model file {model_path} not found. Please train the model first.")

# Try to load the plant disease model
if os.path.exists(plant_disease_model_path):
    if disease_predictor.load_model(plant_disease_model_path):
        print(f"Disease model loaded successfully from: {plant_disease_model_path}")
    else:
        print(f"Warning: Failed to load disease model from: {plant_disease_model_path}")
else:
    print(f"Warning: Disease model file {plant_disease_model_path} not found. You can train it via /api/disease/train")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/features')
def serve_features():
    return send_from_directory('.', 'features.html')

@app.route('/api/crops')
def get_available_crops():
    """Get list of available crops in the dataset"""
    try:
        if predictor.df is not None:
            crops = predictor.df['label'].unique().tolist()
            return jsonify({
                'success': True,
                'crops': sorted(crops)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Dataset not loaded'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict-crop', methods=['POST', 'OPTIONS'])
def predict_crop():
    """Predict crop based on soil parameters"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
            
            try:
                float(data[field])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid value for {field}. Must be a number.'
                }), 400

        # Create input array for prediction
        input_data = [
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]

        # Make prediction
        crop, confidence = predictor.predict(input_data)

        if crop is None or confidence is None:
            return jsonify({
                'success': False,
                'error': 'Failed to make prediction'
            }), 500

        # Get the maximum confidence value
        max_confidence = max(confidence)

        # Get soil parameter recommendations
        soil_recommendations = predictor.recommend_soil_parameters(crop, input_data)

        return jsonify({
            'success': True,
            'crop': crop,
            'confidence': max_confidence,
            'soil_recommendations': soil_recommendations
        })

    except Exception as e:
        print(f"Error in predict_crop route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict-soil', methods=['POST', 'OPTIONS'])
def predict_soil():
    """Predict soil parameters based on crop"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        if 'crop' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: crop'
            }), 400

        crop_name = data['crop'].strip()
        
        if not crop_name:
            return jsonify({
                'success': False,
                'error': 'Crop name cannot be empty'
            }), 400

        # Get soil parameter recommendations for the crop
        soil_parameters = predictor.predict_soil_parameters_for_crop(crop_name)

        if soil_parameters is None:
            return jsonify({
                'success': False,
                'error': f'No data found for crop: {crop_name}'
            }), 404

        return jsonify({
            'success': True,
            'crop': crop_name,
            'soil_parameters': soil_parameters
        })

    except Exception as e:
        print(f"Error in predict_soil route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        if predictor.df is None:
            return jsonify({
                'success': False,
                'error': 'Dataset not loaded'
            }), 500

        # Split features and target
        X = predictor.df[predictor.features]
        y = predictor.df[predictor.target]
        
        # Train the model
        success = predictor.train_model(X, y)
        
        if success:
            # Save the model
            save_success = predictor.save_model(model_path)
            if save_success:
                return jsonify({
                    'success': True,
                    'message': 'Model trained and saved successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Model trained but failed to save'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to train model'
            }), 500

    except Exception as e:
        print(f"Error in train_model route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/disease/classes')
def disease_classes():
    try:
        classes = []
        # Prefer classes from loaded model if available
        if disease_predictor.class_names:
            classes = disease_predictor.class_names
        else:
            classes = disease_predictor.scan_classes()
        return jsonify({'success': True, 'classes': classes})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/disease/train', methods=['POST'])
def disease_train():
    try:
        payload = request.get_json(silent=True) or {}
        per_class_limit = payload.get('per_class_limit', 200)
        metrics = disease_predictor.train(per_class_limit=per_class_limit)
        if disease_predictor.save_model(plant_disease_model_path):
            metrics['model_saved'] = True
            metrics['model_path'] = plant_disease_model_path
        else:
            metrics['model_saved'] = False
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/disease/predict', methods=['POST'])
def disease_predict():
    try:
        # Ensure model is available; if not, try to load or train quickly
        if getattr(disease_predictor, 'model', None) is None:
            # Try loading from disk
            if os.path.exists(plant_disease_model_path):
                disease_predictor.load_model(plant_disease_model_path)
            # If still not available, try a quick training with a small per-class cap
            if getattr(disease_predictor, 'model', None) is None:
                try:
                    metrics = disease_predictor.train(per_class_limit=50)
                    disease_predictor.save_model(plant_disease_model_path)
                    print(f"[disease] Quick-trained model. Metrics: {metrics}")
                except Exception as train_err:
                    print(f"[disease] Quick training failed: {train_err}")
                    return jsonify({'success': False, 'error': 'Disease model not available. Please use the training button first.'}), 500

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided (field name should be "image")'}), 400
        file = request.files['image']
        image_bytes = file.read()
        label, confidence = disease_predictor.predict_image(image_bytes)
        if label is None:
            return jsonify({'success': False, 'error': 'Prediction failed. Ensure the model is trained.'}), 500

        # Make a prettier label for UI
        pretty = label.replace('___', ' - ').replace('_', ' ')
        return jsonify({
            'success': True,
            'label': label,
            'pretty_label': pretty,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True) 