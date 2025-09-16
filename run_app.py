#!/usr/bin/env python3
"""
Smart Crop Recommendation System - Application Runner
"""

import os
import sys
from crop_predictor import CropPredictor

def setup_application():
    """Setup the application by training the model if needed"""
    print("🚀 Setting up Smart Crop Recommendation System...")
    
    # Initialize predictor
    predictor = CropPredictor()
    
    # Check if dataset exists
    dataset_path = 'Crop_recommendation.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Please make sure the Crop_recommendation.csv file exists in the current directory")
        return False
    
    # Load dataset
    print("📊 Loading dataset...")
    X_train, X_test, y_train, y_test = predictor.load_data(dataset_path)
    
    if X_train is None:
        print("❌ Failed to load dataset")
        return False
    
    print(f"✅ Dataset loaded successfully! ({len(X_train)} training samples)")
    
    # Check if model exists
    model_path = 'crop_model.joblib'
    if not os.path.exists(model_path):
        print("🤖 Training new model...")
        if predictor.train_model(X_train, y_train):
            if predictor.save_model(model_path):
                print("✅ Model trained and saved successfully!")
            else:
                print("❌ Failed to save model")
                return False
        else:
            print("❌ Failed to train model")
            return False
    else:
        print("✅ Pre-trained model found!")
    
    return True

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("🌱 Smart Crop Recommendation System")
    print("=" * 60)
    
    # Setup application
    if not setup_application():
        print("\n❌ Application setup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n✅ Application setup completed successfully!")
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run Flask app
    try:
        from app import app
        app.run(host='127.0.0.1', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 