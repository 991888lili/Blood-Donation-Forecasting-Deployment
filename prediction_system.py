"""
Blood Donation Prediction System
Real-time prediction module for Streamlit deployment
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import json


def safe_date_operation(date_obj, operation='add', days=0):
    """Safely perform date operations to avoid pandas compatibility issues"""
    # Always convert to string then back to datetime for maximum compatibility
    if hasattr(date_obj, 'strftime'):
        date_str = date_obj.strftime('%Y-%m-%d')
    elif hasattr(date_obj, 'date'):
        date_str = date_obj.date().strftime('%Y-%m-%d')
    else:
        date_str = str(date_obj)[:10]  # Take first 10 chars (YYYY-MM-DD)
    
    # Convert to Python datetime
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Perform the operation
    if operation == 'add':
        return date_obj + timedelta(days=days)
    elif operation == 'subtract':
        return date_obj - timedelta(days=days)
    else:
        return date_obj

class BloodDonationPredictor:
    def __init__(self, model_dir='models/'):
        """Initialize predictor with trained models"""
        self.model_dir = model_dir
        self.feature_names = [
            'donations_new_diff_1d', 'blood_o_diff_1d', 'donations_new_min_7d', 'blood_o_lag_1',
            'donations_regular_diff_7d', 'donations_new_diff_7d', 'donations_regular_min_7d', 
            'donations_new_mean_14d', 'donations_new_lag_1', 'donations_regular_max_7d', 
            'blood_o_lag_7', 'is_weekend', 'holiday_tomorrow', 'covid_impact', 
            'weekday_hot', 'holiday_yesterday'
        ]
        self.load_models()
        self.setup_prediction_params()
    
    def load_models(self):
        """Load all trained models and scalers"""
        try:
            self.xgb_model = joblib.load(f'{self.model_dir}xgb_meta.pkl')
            self.lstm_model = load_model(f'{self.model_dir}lstm_meta.keras')
            self.meta_learner = joblib.load(f'{self.model_dir}meta_learner.pkl')
            self.scaler_X = joblib.load(f'{self.model_dir}scaler_X_meta.pkl')
            self.scaler_y = joblib.load(f'{self.model_dir}scaler_y_meta.pkl')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def setup_prediction_params(self):
        """Setup prediction parameters (optimized values)"""
        self.weekend_boost_factor = 0.75
        self.weekday_effects = {
            0: -0.05,  # Monday: slightly lower
            1: 0.02,   # Tuesday: slightly higher
            2: 0.0,    # Wednesday: baseline
            3: 0.0,    # Thursday: baseline
            4: -0.06,  # Friday: lower
        }
        self.mean_pull_strength = 0.35
        self.daily_variation_factor = 0.4
        self.random_factor = 0.1
        self.trend_continuation_factor = 0.05
    
    def prepare_historical_data(self, data):
        """Prepare historical data for prediction"""
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Calculate baseline metrics
        self.historical_data = data
        self.recent_mean = data['daily'].mean()
        self.baseline_blood_o = data['blood_o'].mean()
        self.baseline_donations_new = data['donations_new'].mean()
        self.baseline_donations_regular = data['donations_regular'].mean()
        
        # Calculate variation parameters
        daily_changes = data['daily'].diff().dropna()
        self.daily_variation = daily_changes.std() * self.daily_variation_factor
        
        # Calculate weekend effects
        # Safe dayofweek calculation
        data['date'] = pd.to_datetime(data['date'])
        data['dayofweek'] = [d.weekday() for d in data['date']]
        weekdays = data[data['dayofweek'] < 5]
        weekends = data[data['dayofweek'] >= 5]
        
        if len(weekdays) > 0 and len(weekends) > 0:
            weekend_avg = weekends['daily'].mean()
            weekday_avg = weekdays['daily'].mean()
            weekend_boost = (weekend_avg / weekday_avg - 1.0) * self.weekend_boost_factor
            
            self.weekday_effects[5] = weekend_boost * 0.9  # Saturday
            self.weekday_effects[6] = weekend_boost * 1.0  # Sunday
        else:
            self.weekday_effects[5] = 0.3  # Default Saturday boost
            self.weekday_effects[6] = 0.4  # Default Sunday boost
        
        print(f"Historical data prepared: {len(data)} days")
        print(f"Baseline mean: {self.recent_mean:.1f}")
        print(f"Weekend boost: Sat {self.weekday_effects[5]*100:+.1f}%, Sun {self.weekday_effects[6]*100:+.1f}%")
    
    def calculate_features(self, target_date):
        """Calculate features for target date from historical data"""
        # Convert target_date to string for safe comparison
        if hasattr(target_date, 'strftime'):
            target_str = target_date.strftime('%Y-%m-%d')
        else:
            target_str = str(target_date)[:10]

        hist_data = self.historical_data[self.historical_data['date'].dt.strftime('%Y-%m-%d') < target_str]
        
        if len(hist_data) < 14:
            raise ValueError(f"Need at least 14 days before {target_date}")
        
        recent_data = hist_data.tail(30)
        
        # Extract time series
        blood_o = recent_data['blood_o'].values
        donations_new = recent_data['donations_new'].values
        donations_regular = recent_data['donations_regular'].values
        
        features = {}
        
        # Ensure target_date is a datetime object for weekday calculation
        target_dt = safe_date_operation(target_date)
        
        # Time-based features
        features['is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        
        # Lag features with stabilization
        stabilization_factor = 0.2
        features['blood_o_lag_1'] = (1 - stabilization_factor) * blood_o[-1] + stabilization_factor * self.baseline_blood_o
        features['donations_new_lag_1'] = (1 - stabilization_factor) * donations_new[-1] + stabilization_factor * self.baseline_donations_new
        features['blood_o_lag_7'] = (1 - stabilization_factor) * (blood_o[-7] if len(blood_o) >= 7 else blood_o[0]) + stabilization_factor * self.baseline_blood_o
        
        # Difference features with dampening
        dampening = 0.7
        features['blood_o_diff_1d'] = (blood_o[-1] - blood_o[-2]) * dampening if len(blood_o) >= 2 else 0
        features['donations_new_diff_1d'] = (donations_new[-1] - donations_new[-2]) * dampening if len(donations_new) >= 2 else 0
        features['donations_new_diff_7d'] = (donations_new[-1] - donations_new[-7]) * dampening if len(donations_new) >= 7 else 0
        features['donations_regular_diff_7d'] = (donations_regular[-1] - donations_regular[-7]) * dampening if len(donations_regular) >= 7 else 0
        
        # Rolling statistics
        features['donations_new_min_7d'] = np.min(donations_new[-7:]) if len(donations_new) >= 7 else np.min(donations_new)
        features['donations_new_mean_14d'] = np.mean(donations_new[-14:]) if len(donations_new) >= 14 else np.mean(donations_new)
        features['donations_regular_min_7d'] = np.min(donations_regular[-7:]) if len(donations_regular) >= 7 else np.min(donations_regular)
        features['donations_regular_max_7d'] = np.max(donations_regular[-7:]) if len(donations_regular) >= 7 else np.max(donations_regular)
        
        # External features
        features['holiday_tomorrow'] = 0  # Simplified for deployment
        features['holiday_yesterday'] = recent_data['is_holiday'].iloc[-1] if len(recent_data) > 0 else 0
        features['covid_impact'] = 0  # Set to 0 for 2025 predictions
        
        # Temperature-based weekday_hot (corrected threshold)
        recent_temp = recent_data['max_temperature'].iloc[-1] if len(recent_data) > 0 else 25.0
        is_weekday = target_dt.weekday() < 5
        features['weekday_hot'] = 1 if (is_weekday and recent_temp > 33) else 0
        
        return features
    
    def predict_single_day(self, target_date):
        """Predict blood donations for a single day"""
        # Ensure target_date is a datetime object
        target_date = safe_date_operation(target_date)
        
        # Calculate features
        features = self.calculate_features(target_date)
        
        # Create feature vector with proper column names
        feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        feature_df = pd.DataFrame(feature_vector, columns=self.feature_names)
        feature_vector_scaled = self.scaler_X.transform(feature_df)
        
        # Model predictions
        xgb_pred = self.xgb_model.predict(feature_df)[0]
        
        lstm_input = feature_vector_scaled.reshape((1, 1, feature_vector_scaled.shape[1]))
        lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0).flatten()
        lstm_pred = self.scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        # Ensemble prediction
        meta_features = np.column_stack([[xgb_pred], [lstm_pred]])
        raw_ensemble = self.meta_learner.predict(meta_features)[0]
        
        # Apply adjustments
        mean_pulled = (1 - self.mean_pull_strength) * raw_ensemble + self.mean_pull_strength * self.recent_mean
        
        # Day-of-week effect
        dow_effect = self.weekday_effects[target_date.weekday()] * self.recent_mean
        
        # Random variation
        daily_random = np.random.normal(0, self.daily_variation * self.random_factor)
        
        # Trend continuation
        trend_continuation = 0
        if hasattr(self, 'last_prediction'):
            recent_trend = self.last_prediction - self.recent_mean
            trend_continuation = recent_trend * self.trend_continuation_factor
        
        # Final prediction
        final_prediction = mean_pulled + dow_effect + daily_random + trend_continuation
        
        # Bounds based on historical data
        hist_min = self.historical_data['daily'].min()
        hist_max = self.historical_data['daily'].max()
        final_prediction = np.clip(final_prediction, hist_min * 0.7, hist_max * 1.3)
        
        # Store for trend continuation
        self.last_prediction = final_prediction
        
        # Update historical data for next prediction
        self.update_historical_data(target_date, final_prediction)
        
        return {
            'date': target_date,
            'prediction': final_prediction,
            'raw_ensemble': raw_ensemble,
            'xgb_pred': xgb_pred,
            'lstm_pred': lstm_pred,
            'components': {
                'mean_pulled': mean_pulled,
                'dow_effect': dow_effect,
                'daily_random': daily_random,
                'trend_continuation': trend_continuation
            },
            'features': features
        }
    
    def update_historical_data(self, target_date, predicted_daily):
        """Update historical data with prediction for next iteration"""
        # Calculate component breakdown based on historical ratios
        hist_data = self.historical_data
        total_daily = hist_data['daily'].sum()
        
        blood_o_ratio = hist_data['blood_o'].sum() / total_daily
        donations_new_ratio = hist_data['donations_new'].sum() / total_daily
        donations_regular_ratio = hist_data['donations_regular'].sum() / total_daily
        
        # Add small variation
        variation = np.random.normal(1.0, 0.02, 3)
        
        new_blood_o = predicted_daily * blood_o_ratio * variation[0]
        new_donations_new = predicted_daily * donations_new_ratio * variation[1]
        new_donations_regular = predicted_daily * donations_regular_ratio * variation[2]
        
        # Rescale to match prediction
        total_components = new_blood_o + new_donations_new + new_donations_regular
        scale_factor = predicted_daily / total_components
        
        new_blood_o *= scale_factor
        new_donations_new *= scale_factor
        new_donations_regular *= scale_factor
        
        # Temperature variation
        recent_temp = hist_data['max_temperature'].iloc[-1]
        new_temp = recent_temp + np.random.normal(0, 1.0)
        new_temp = np.clip(new_temp, 15, 35)
        
        # Create new row with safe date conversion
        target_date_pd = pd.to_datetime(target_date)
        new_row = pd.DataFrame({
            'date': [target_date_pd],
            'daily': [predicted_daily],
            'blood_o': [new_blood_o],
            'donations_new': [new_donations_new],
            'donations_regular': [new_donations_regular],
            'max_temperature': [new_temp],
            'is_holiday': [0]
        })
        
        # Add to historical data
        self.historical_data = pd.concat([self.historical_data, new_row], ignore_index=True)
        
        # Keep reasonable history length
        if len(self.historical_data) > 45:
            self.historical_data = self.historical_data.tail(45).reset_index(drop=True)
    
    def predict_multiple_days(self, start_date, num_days=7):
        """Predict multiple consecutive days"""
        # Convert start_date to Python datetime object
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # Use safe conversion for any other type
            start_date = safe_date_operation(start_date)

        # Ensure historical data dates are pandas datetime
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        
        predictions = []
        
        # Reset trend continuation
        if hasattr(self, 'last_prediction'):
            delattr(self, 'last_prediction')
        
        for day in range(num_days):
            # Use safe date addition
            target_date = safe_date_operation(start_date, 'add', day)
            pred_result = self.predict_single_day(target_date)
            predictions.append(pred_result)
        
        return predictions
    
    def get_prediction_summary(self, predictions):
        """Generate summary statistics for predictions"""
        values = [p['prediction'] for p in predictions]
        
        # Weekend vs weekday analysis
        weekday_preds = [p['prediction'] for p in predictions if p['date'].weekday() < 5]
        weekend_preds = [p['prediction'] for p in predictions if p['date'].weekday() >= 5]
        
        summary = {
            'predictions': [round(v, 0) for v in values],
            'mean_prediction': np.mean(values),
            'std_prediction': np.std(values),
            'min_prediction': np.min(values),
            'max_prediction': np.max(values),
            'weekday_avg': np.mean(weekday_preds) if weekday_preds else None,
            'weekend_avg': np.mean(weekend_preds) if weekend_preds else None,
            'weekend_effect': np.mean(weekend_preds) - np.mean(weekday_preds) if weekday_preds and weekend_preds else None,
            'coefficient_variation': np.std(values) / np.mean(values) * 100,
            'baseline_comparison': np.mean(values) - self.recent_mean,
            'trend_change': values[-1] - values[0] if len(values) > 1 else 0
        }
        
        return summary

def load_prediction_config():
    """Load prediction configuration from JSON"""
    try:
        with open('models/model_info.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Warning: model_info.json not found, using defaults")
        return {}

# Example usage
if __name__ == "__main__":
    # Example of how to use the predictor
    predictor = BloodDonationPredictor()
    
    # Load sample historical data
    sample_data = pd.read_csv('sample_december_data.csv')
    predictor.prepare_historical_data(sample_data)
    
    # Make predictions
    predictions = predictor.predict_multiple_days('2025-01-01', 7)
    summary = predictor.get_prediction_summary(predictions)
    
    print("Predictions:", summary['predictions'])
    print(f"Mean: {summary['mean_prediction']:.0f}")
    print(f"Weekend effect: {summary['weekend_effect']:+.0f}")