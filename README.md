# 🩸 Blood Donation Prediction System

A machine learning-powered forecasting system for predicting blood donation volumes using ensemble models. This system combines XGBoost, LSTM, and meta-learning techniques to provide accurate 7-day blood donation forecasts with interactive visualizations.

## 🎯 Key Features

- **High-Accuracy Predictions**: MAE of 25.18 with R² of 0.9803
- **Ensemble Learning**: Combines XGBoost (31.86%) + LSTM (71.83%) + Meta-learner
- **Smart Analytics**: Weekend vs weekday analysis, trend detection, confidence intervals
- **Interactive Dashboard**: Built with Streamlit for user-friendly experience

## 🏗️ System Architecture

```
Input: 14 days historical data → Feature Engineering → Ensemble Models → 7-day Predictions
                                       ↓
                    [XGBoost] + [LSTM] → [Meta-learner] → Final Forecast
```

### Model Performance
- **Mean Absolute Error (MAE)**: 25.18 donations
- **R² Score**: 0.9803

## 📊 Data Requirements

Upload a CSV file with **exactly 14 days** of historical data containing these columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `date` | Date in YYYY-MM-DD format | String | 2024-12-18 |
| `blood_o` | O-type blood donations | Integer | 245 |
| `donations_new` | New donor donations | Integer | 123 |
| `donations_regular` | Regular donor donations | Integer | 432 |
| `daily` | Total daily blood donations | Integer | 568 |
| `max_temperature` | Maximum temperature (°C) | Float | 28.5 |
| `is_holiday` | Holiday indicator (0/1) | Integer | 0 |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/991888lili/blood-donation-forecasting-Deployment.git
   cd blood-donation-forecasting-Deployment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   ```
   http://localhost:8501
   ```

## 📁 Project Structure

```
blood-donation-forecasting/
├── app.py                      # Main Streamlit application
├── prediction_system.py        # Core prediction engine
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Trained model files
│   ├── meta_learner.pkl       # Meta-learning ensemble model
│   ├── xgb_meta.pkl          # XGBoost model
│   ├── lstm_meta.keras       # LSTM neural network
│   ├── scaler_X_meta.pkl     # Feature scaler
│   ├── scaler_y_meta.pkl     # Target scaler
│   └── model_info.json       # Model metadata
├── sample_data/               # Example datasets
│   ├── sample_december_data.csv
│   └── data_description.json
└── screenshots/               # Application screenshots
```

## 🔧 Usage Guide

### Step 1: Data Upload
1. Download the sample data template from the app
2. Prepare your 14-day historical data in the same format
3. Upload your CSV file through the interface

### Step 2: Configure Predictions
- **Start Date**: Choose the first day to predict
- **Duration**: default: 7 days
- **Advanced Options**: Toggle prediction components and confidence analysis

### Step 3: Generate Forecasts
- Click "Generate Predictions" to run the ensemble models
- View detailed results including trends, patterns, and downloadable reports

### Step 4: Analyze Results
- **Daily Predictions**: Tabular view with dates and forecast values
- **Trend Visualization**: Historical data + future predictions
- **Weekend Analysis**: Weekday vs weekend donation patterns
- **Confidence Metrics**: Prediction reliability indicators

## 📈 Model Details

### Ensemble Architecture
1. **XGBoost Model**: Gradient boosting for pattern recognition
2. **LSTM Model**: Recurrent neural network for time series dependencies
3. **Meta-learner**: Optimally combines predictions with learned weights


## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from GitHub


## 🛠️ Technical Stack

- **Backend**: Python 3.9+
- **ML Framework**: scikit-learn, XGBoost, TensorFlow/Keras
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: pandas, NumPy
- **Model Persistence**: joblib, pickle


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Built for NBC (National Blood Center) forecasting needs
- Inspired by time series forecasting best practices
- Ensemble techniques based on modern ML research

---

**⚡ Quick Demo**: Upload the sample data and click "Generate Predictions" to see the system in action!

