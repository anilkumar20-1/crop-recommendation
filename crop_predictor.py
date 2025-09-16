import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import zipfile

class CropPredictor:
    def __init__(self):
        self.model = None
        self.features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target = 'label'
        self.df = None  # Store the dataset for soil parameter analysis
        
    def load_data(self, dataset_path):
        """Load and preprocess the dataset"""
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
            
            # Handle zip file
            if dataset_path.endswith('.zip'):
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    # Extract the CSV file from the zip
                    csv_file = zip_ref.namelist()[0]  # Get the first file in the zip
                    with zip_ref.open(csv_file) as f:
                        df = pd.read_csv(f)
            else:
                # Read the dataset directly if it's a CSV
                df = pd.read_csv(dataset_path)
            
            # Store the dataset for soil parameter analysis
            self.df = df.copy()
            
            # Verify required columns exist
            required_columns = self.features + [self.target]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in dataset: {missing_columns}")
            
            # Split features and target
            X = df[self.features]
            y = df[self.target]
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None, None, None, None

    def train_model(self, X_train, y_train):
        """Train the Decision Tree model"""
        try:
            self.model = DecisionTreeClassifier(random_state=42)
            self.model.fit(X_train, y_train)
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def save_model(self, model_path):
        """Save the trained model"""
        try:
            joblib.dump(self.model, model_path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path):
        """Load a trained model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, input_data):
        """Make predictions for new input data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Convert input data to numpy array and reshape
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_array)
            probability = self.model.predict_proba(input_array)
            
            return prediction[0], probability[0]
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None, None

    def recommend_soil_parameters(self, crop_name, current_params):
        """Recommend optimal soil parameters for a specific crop"""
        try:
            if self.df is None:
                raise ValueError("Dataset not loaded. Please load the dataset first.")
            
            # Filter data for the specific crop
            crop_data = self.df[self.df[self.target] == crop_name]
            
            if len(crop_data) == 0:
                return None
            
            # Calculate optimal ranges for each parameter
            recommendations = {}
            current_values = {
                'N': current_params[0],
                'P': current_params[1], 
                'K': current_params[2],
                'temperature': current_params[3],
                'humidity': current_params[4],
                'ph': current_params[5],
                'rainfall': current_params[6]
            }
            
            for feature in self.features:
                feature_data = crop_data[feature]
                mean_val = feature_data.mean()
                std_val = feature_data.std()
                
                # Define optimal range (mean ± 1 standard deviation)
                min_optimal = max(0, mean_val - std_val)
                max_optimal = mean_val + std_val
                
                current_val = current_values[feature]
                
                # Determine status
                if min_optimal <= current_val <= max_optimal:
                    status = "Optimal"
                elif current_val < min_optimal:
                    status = "Low"
                else:
                    status = "High"
                
                recommendations[feature] = {
                    'current': current_val,
                    'optimal_min': min_optimal,
                    'optimal_max': max_optimal,
                    'mean': mean_val,
                    'status': status,
                    'recommendation': self._get_parameter_recommendation(feature, current_val, min_optimal, max_optimal)
                }
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating soil parameter recommendations: {str(e)}")
            return None
    
    def _get_parameter_recommendation(self, parameter, current_val, min_optimal, max_optimal):
        """Generate specific recommendations for each parameter"""
        recommendations = {
            'N': {
                'low': "Consider adding nitrogen-rich fertilizers like urea or ammonium nitrate",
                'high': "Reduce nitrogen application to prevent excessive vegetative growth",
                'optimal': "Nitrogen levels are within optimal range"
            },
            'P': {
                'low': "Apply phosphorus-rich fertilizers like superphosphate or bone meal",
                'high': "Reduce phosphorus application to prevent nutrient imbalance",
                'optimal': "Phosphorus levels are within optimal range"
            },
            'K': {
                'low': "Add potassium-rich fertilizers like potash or wood ash",
                'high': "Reduce potassium application to prevent salt buildup",
                'optimal': "Potassium levels are within optimal range"
            },
            'temperature': {
                'low': "Consider using greenhouses or row covers to increase temperature",
                'high': "Implement shade structures or irrigation to cool the soil",
                'optimal': "Temperature is within optimal range for this crop"
            },
            'humidity': {
                'low': "Increase irrigation frequency or use mulch to retain moisture",
                'high': "Improve drainage and reduce irrigation to lower humidity",
                'optimal': "Humidity levels are within optimal range"
            },
            'ph': {
                'low': "Add lime to raise soil pH to optimal levels",
                'high': "Add sulfur or organic matter to lower soil pH",
                'optimal': "Soil pH is within optimal range"
            },
            'rainfall': {
                'low': "Implement irrigation systems to supplement rainfall",
                'high': "Improve drainage systems to prevent waterlogging",
                'optimal': "Rainfall is within optimal range"
            }
        }
        
        if current_val < min_optimal:
            return recommendations[parameter]['low']
        elif current_val > max_optimal:
            return recommendations[parameter]['high']
        else:
            return recommendations[parameter]['optimal']

    def predict_soil_parameters_for_crop(self, crop_name):
        """Predict optimal soil parameters for a specific crop"""
        try:
            if self.df is None:
                raise ValueError("Dataset not loaded. Please load the dataset first.")
            
            # Filter data for the specific crop
            crop_data = self.df[self.df[self.target] == crop_name]
            
            if len(crop_data) == 0:
                return None
            
            # Calculate optimal ranges for each parameter
            soil_parameters = {}
            
            for feature in self.features:
                feature_data = crop_data[feature]
                mean_val = feature_data.mean()
                std_val = feature_data.std()
                
                # Define optimal range (mean ± 1 standard deviation)
                min_optimal = max(0, mean_val - std_val)
                max_optimal = mean_val + std_val
                
                # Get specific recommendations for this parameter
                recommendation = self._get_crop_specific_recommendation(feature, crop_name, mean_val)
                
                soil_parameters[feature] = {
                    'optimal_min': min_optimal,
                    'optimal_max': max_optimal,
                    'mean': mean_val,
                    'std': std_val,
                    'recommendation': recommendation
                }
            
            return soil_parameters
            
        except Exception as e:
            print(f"Error predicting soil parameters for crop: {str(e)}")
            return None
    
    def _get_crop_specific_recommendation(self, parameter, crop_name, mean_val):
        """Generate crop-specific recommendations for soil parameters"""
        crop_specific_recommendations = {
            'rice': {
                'N': f"Rice requires {mean_val:.1f} mg/kg nitrogen. Apply nitrogen in split doses - 50% at transplanting, 25% at tillering, and 25% at panicle initiation.",
                'P': f"Rice needs {mean_val:.1f} mg/kg phosphorus. Apply phosphorus at transplanting for better root development.",
                'K': f"Rice requires {mean_val:.1f} mg/kg potassium. Apply potassium at transplanting and panicle initiation stages.",
                'temperature': f"Rice grows best at {mean_val:.1f}°C. Maintain warm conditions, especially during flowering.",
                'humidity': f"Rice prefers {mean_val:.1f}% humidity. Maintain high humidity during vegetative growth.",
                'ph': f"Rice prefers pH {mean_val:.1f}. Slightly acidic to neutral soil is ideal.",
                'rainfall': f"Rice needs {mean_val:.1f} mm rainfall. Ensure consistent water supply throughout growth."
            },
            'maize': {
                'N': f"Maize requires {mean_val:.1f} mg/kg nitrogen. Apply nitrogen in 3-4 split doses during growth stages.",
                'P': f"Maize needs {mean_val:.1f} mg/kg phosphorus. Apply phosphorus at sowing for early root development.",
                'K': f"Maize requires {mean_val:.1f} mg/kg potassium. Apply potassium at sowing and knee-high stage.",
                'temperature': f"Maize grows best at {mean_val:.1f}°C. Warm temperatures are essential for germination and growth.",
                'humidity': f"Maize prefers {mean_val:.1f}% humidity. Moderate humidity is optimal for growth.",
                'ph': f"Maize prefers pH {mean_val:.1f}. Slightly acidic to neutral soil is best.",
                'rainfall': f"Maize needs {mean_val:.1f} mm rainfall. Regular irrigation is important during critical growth stages."
            },
            'chickpea': {
                'N': f"Chickpea requires {mean_val:.1f} mg/kg nitrogen. Being a legume, it can fix atmospheric nitrogen.",
                'P': f"Chickpea needs {mean_val:.1f} mg/kg phosphorus. Apply phosphorus at sowing for root development.",
                'K': f"Chickpea requires {mean_val:.1f} mg/kg potassium. Apply potassium at sowing.",
                'temperature': f"Chickpea grows best at {mean_val:.1f}°C. Cool to warm temperatures are optimal.",
                'humidity': f"Chickpea prefers {mean_val:.1f}% humidity. Moderate humidity is suitable.",
                'ph': f"Chickpea prefers pH {mean_val:.1f}. Neutral to slightly alkaline soil is ideal.",
                'rainfall': f"Chickpea needs {mean_val:.1f} mm rainfall. Avoid waterlogging conditions."
            },
            'cotton': {
                'N': f"Cotton requires {mean_val:.1f} mg/kg nitrogen. Apply nitrogen in split doses during vegetative and flowering stages.",
                'P': f"Cotton needs {mean_val:.1f} mg/kg phosphorus. Apply phosphorus at sowing and flowering.",
                'K': f"Cotton requires {mean_val:.1f} mg/kg potassium. Apply potassium at sowing and boll development.",
                'temperature': f"Cotton grows best at {mean_val:.1f}°C. Warm temperatures are essential for growth.",
                'humidity': f"Cotton prefers {mean_val:.1f}% humidity. Moderate humidity is optimal.",
                'ph': f"Cotton prefers pH {mean_val:.1f}. Neutral to slightly alkaline soil is best.",
                'rainfall': f"Cotton needs {mean_val:.1f} mm rainfall. Regular irrigation is important during boll development."
            }
        }
        
        # Default recommendation if crop not found in specific recommendations
        default_recommendations = {
            'N': f"Apply {mean_val:.1f} mg/kg nitrogen based on soil test results.",
            'P': f"Apply {mean_val:.1f} mg/kg phosphorus based on soil test results.",
            'K': f"Apply {mean_val:.1f} mg/kg potassium based on soil test results.",
            'temperature': f"Maintain temperature around {mean_val:.1f}°C for optimal growth.",
            'humidity': f"Maintain humidity around {mean_val:.1f}% for optimal growth.",
            'ph': f"Maintain soil pH around {mean_val:.1f} for optimal growth.",
            'rainfall': f"Ensure rainfall/irrigation around {mean_val:.1f} mm for optimal growth."
        }
        
        if crop_name.lower() in crop_specific_recommendations:
            return crop_specific_recommendations[crop_name.lower()].get(parameter, default_recommendations[parameter])
        else:
            return default_recommendations[parameter]

def main():
    # Initialize the predictor
    predictor = CropPredictor()
    
    # Get the current directory (curser folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the dataset in current workspace
    dataset_path = os.path.join(current_dir, "Crop_recommendation.csv")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        print("Please make sure the Crop_recommendation.csv file exists at the specified path")
        return
    
    # Train new model
    print(f"\nLoading dataset from: {dataset_path}")
    X_train, X_test, y_train, y_test = predictor.load_data(dataset_path)
    
    if X_train is not None:
        print("Training model...")
        if predictor.train_model(X_train, y_train):
            print("Model trained successfully!")
            
            # Save the model
            model_path = os.path.join(current_dir, "crop_model.joblib")
            if predictor.save_model(model_path):
                print(f"Model saved successfully to: {model_path}")
            
            # Ask user for prediction type
            print("\nChoose prediction type:")
            print("1. Crop recommendation based on soil parameters")
            print("2. Soil parameters recommendation based on crop")
            print("----------------------------------------")
            
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                
                if choice == "1":
                    # Get user input for soil parameters
                    print("\nEnter soil parameters for crop recommendation:")
                    print("----------------------------------------")
                    
                    nitrogen = float(input("Enter Nitrogen (N) content: "))
                    phosphorus = float(input("Enter Phosphorus (P) content: "))
                    potassium = float(input("Enter Potassium (K) content: "))
                    temperature = float(input("Enter Temperature (°C): "))
                    humidity = float(input("Enter Humidity (%): "))
                    ph = float(input("Enter pH Level (0-14): "))
                    rainfall = float(input("Enter Rainfall (mm): "))
                    
                    # Create input array
                    input_data = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                    
                    # Make prediction
                    prediction, probability = predictor.predict(input_data)
                    
                    if prediction is not None:
                        print(f"Confidence: {max(probability)*100:.2f}%")
                        print("\nInput Parameters:")
                        print(f"Nitrogen: {nitrogen} mg/kg")
                        print(f"Phosphorus: {phosphorus} mg/kg")
                        print(f"Potassium: {potassium} mg/kg")
                        print(f"Temperature: {temperature}°C")
                        print(f"Humidity: {humidity}%")
                        print(f"pH Level: {ph}")
                        print(f"Rainfall: {rainfall} mm")
                        print("\nCrop Recommendation:")
                        print("----------------------------------------")
                        print(f"Recommended Crop: {prediction}")
                        
                        # Get soil parameter recommendations
                        print("\nSoil Parameters Analysis:")
                        print("----------------------------------------")
                        soil_recommendations = predictor.recommend_soil_parameters(prediction, input_data)
                        
                        if soil_recommendations:
                            print(f"Optimal soil parameters for {prediction}:")
                            print()
                            
                            for param, data in soil_recommendations.items():
                                status_symbol = "✅" if data['status'] == "Optimal" else "⚠️" if data['status'] == "Low" else "❌"
                                print(f"{status_symbol} {param.upper()}:")
                                print(f"   Current: {data['current']:.2f}")
                                print(f"   Optimal Range: {data['optimal_min']:.2f} - {data['optimal_max']:.2f}")
                                print(f"   Status: {data['status']}")
                                print(f"   Recommendation: {data['recommendation']}")
                                print()
                        else:
                            print("Unable to generate soil parameter recommendations")
                    else:
                        print("Failed to make prediction")
                
                elif choice == "2":
                    # Get user input for crop
                    print("\nEnter crop name for soil parameters recommendation:")
                    print("----------------------------------------")
                    
                    crop_name = input("Enter crop name: ").strip()
                    
                    # Get soil parameter recommendations for the crop
                    soil_parameters = predictor.predict_soil_parameters_for_crop(crop_name)
                    
                    if soil_parameters:
                        print(f"\nOptimal Soil Parameters for {crop_name}:")
                        print("----------------------------------------")
                        
                        for param, data in soil_parameters.items():
                            print(f"🌱 {param.upper()}:")
                            print(f"   Optimal Range: {data['optimal_min']:.2f} - {data['optimal_max']:.2f}")
                            print(f"   Average Value: {data['mean']:.2f}")
                            print(f"   Standard Deviation: {data['std']:.2f}")
                            print(f"   Recommendation: {data['recommendation']}")
                            print()
                    else:
                        print(f"No data found for crop: {crop_name}")
                        print("Available crops in dataset:")
                        if predictor.df is not None:
                            available_crops = predictor.df[predictor.target].unique()
                            for crop in sorted(available_crops):
                                print(f"  - {crop}")
                
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    
            except ValueError:
                print("Error: Please enter valid numerical values")
        else:
            print("Failed to train model")
    else:
        print("Failed to load dataset")

if __name__ == "__main__":
    main() 