# Smart Crop Recommendation System

An AI-powered web application that provides intelligent crop recommendations based on soil parameters and optimal soil conditions for specific crops using advanced machine learning.

## 🌟 Features

### Dual Prediction System
- **Crop Recommendation**: Get AI-powered crop suggestions based on your soil parameters
- **Soil Parameters**: Find optimal soil conditions for your desired crop

### Advanced Analysis
- Real-time crop recommendations with confidence scores
- Detailed soil parameter analysis with status indicators
- Crop-specific management recommendations
- Statistical analysis of optimal ranges

### Modern Interface
- Responsive, mobile-friendly design
- Interactive tabbed interface
- Real-time loading indicators
- Beautiful visual feedback
- Crop images and confidence bars

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)
```bash
# Run the setup script
python run_app.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python crop_predictor.py

# Start the server
python app.py
```

3. Open your browser and go to: `http://127.0.0.1:5000`

## 📊 How It Works

### Crop Prediction
1. Enter your soil parameters:
   - Nitrogen (N) content (mg/kg)
   - Phosphorus (P) content (mg/kg)
   - Potassium (K) content (mg/kg)
   - Temperature (°C)
   - Humidity (%)
   - pH Level (0-14)
   - Rainfall (mm)

2. Get instant recommendations with:
   - Recommended crop with confidence score
   - Detailed soil parameter analysis
   - Status indicators (Optimal/Warning/Critical)
   - Specific management recommendations

### Soil Parameters Prediction
1. Select a crop from the dropdown
2. Get comprehensive soil requirements:
   - Optimal parameter ranges
   - Statistical data (mean, standard deviation)
   - Crop-specific management advice
   - Detailed recommendations for each parameter

## 🛠️ API Endpoints

### Crop Prediction
```http
POST /api/predict-crop
Content-Type: application/json

{
  "nitrogen": 50.0,
  "phosphorus": 40.0,
  "potassium": 30.0,
  "temperature": 25.0,
  "humidity": 70.0,
  "ph": 6.5,
  "rainfall": 100.0
}
```

### Soil Parameters Prediction
```http
POST /api/predict-soil
Content-Type: application/json

{
  "crop": "rice"
}
```

### Get Available Crops
```http
GET /api/crops
```

### Train Model
```http
POST /api/train-model
```

## 📁 Project Structure

```
crop recommendation/
├── crop_predictor.py      # Core ML model and prediction logic
├── app.py                # Flask web server and API endpoints
├── run_app.py            # Application setup and runner
├── index.html            # Modern web interface
├── requirements.txt      # Python dependencies
├── Crop_recommendation.csv # Training dataset
├── crop_model.joblib     # Trained model file
├── python/               # Crop images
└── templates/            # Flask templates (if needed)
```

## 🧠 Machine Learning Features

- **Decision Tree Classifier**: Robust crop prediction model
- **Data Analysis**: Statistical analysis of soil parameters
- **Crop-Specific Recommendations**: Tailored advice for each crop
- **Confidence Scoring**: Reliability indicators for predictions
- **Parameter Optimization**: Optimal range calculations

## 🎨 Technologies Used

### Backend
- **Python 3.7+**: Core programming language
- **Flask**: Web framework for API endpoints
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **joblib**: Model serialization

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Advanced styling with CSS variables
- **Bootstrap 5**: Responsive UI framework
- **JavaScript (ES6+)**: Interactive functionality
- **Font Awesome**: Beautiful icons

## 🔧 Configuration

The application automatically:
- Loads the dataset from `Crop_recommendation.csv`
- Trains the model if no pre-trained model exists
- Saves the trained model as `crop_model.joblib`
- Starts the web server on `http://127.0.0.1:5000`

## 📱 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Mobile)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Crop recommendation dataset
- Icons: Font Awesome
- UI Framework: Bootstrap
- ML Library: scikit-learn 