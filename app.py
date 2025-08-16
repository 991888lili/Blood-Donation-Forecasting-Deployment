import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io

# Import our prediction system
from prediction_system import BloodDonationPredictor, load_prediction_config

# Page configuration
st.set_page_config(
    page_title="Blood Donation Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #D32F2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #D32F2F;
    }
    .prediction-table {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🩸 Blood Donation Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload Data**: Upload your 14-day historical blood donation data
    2. **Set Parameters**: Choose prediction start date and number of days
    3. **View Results**: Get predictions with detailed analysis and visualizations
    """)
    
    st.header("📊 Model Information")
    try:
        config = load_prediction_config()
        if config:
            st.metric("Model Performance (MAE)", f"{config.get('model_performance', {}).get('ensemble_test_mae', 'N/A')}")
            st.metric("Model R²", f"{config.get('model_performance', {}).get('ensemble_test_r2', 'N/A')}")
    except:
        st.write("Model info not available")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Data Upload")
    
    # Sample data download
    st.subheader("Sample Data Template")
    try:
        sample_data = pd.read_csv('sample_december_data.csv')
        
        # Create download button for sample data
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample Data Template",
            data=csv_buffer.getvalue(),
            file_name="sample_december_data.csv",
            mime="text/csv",
            help="Download this template to see the required data format"
        )
        
        # Show sample data preview
        with st.expander("👀 Preview Sample Data Format"):
            st.dataframe(sample_data.head(), use_container_width=True)
            
    except FileNotFoundError:
        st.error("Sample data file not found. Please ensure sample_december_data.csv exists.")
    
    # File upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with 14 days of historical data",
        type=['csv'],
        help="Upload CSV file with same format as sample data"
    )

with col2:
    st.header("⚙️ Prediction Settings")
    
    # Prediction parameters
    start_date = st.date_input(
        "Prediction Start Date",
        value=datetime(2025, 1, 1),
        help="First day to predict"
    )
    
    num_days = st.selectbox(
        "Number of Days to Predict",
        options=[1, 3, 5, 7, 10, 14],
        index=3,  # Default to 7 days
        help="How many consecutive days to predict"
    )
    
    st.subheader("🔧 Advanced Settings")
    with st.expander("Model Parameters (Optional)"):
        show_components = st.checkbox("Show Prediction Components", value=False)
        show_confidence = st.checkbox("Show Confidence Analysis", value=True)

# Main prediction section
if uploaded_file is not None:
    try:
        # Load and validate data
        data = pd.read_csv(uploaded_file)
        
        # Data validation
        required_columns = ['date', 'blood_o', 'donations_new', 'donations_regular', 'daily', 'max_temperature', 'is_holiday']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Please ensure your data has all required columns. Download the sample template for reference.")
        else:
            # Data info
            st.success(f"✅ Data uploaded successfully! {len(data)} days of historical data loaded.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Data Range", f"{len(data)} days")
            with col2:
                st.metric("📊 Daily Average", f"{data['daily'].mean():.0f}")
            with col3:
                st.metric("📈 Daily Range", f"{data['daily'].min():.0f} - {data['daily'].max():.0f}")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(data, use_container_width=True)
            
            # Prediction button
            st.markdown("---")
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                
                with st.spinner("🤖 Running prediction models..."):
                    try:
                        # Initialize predictor
                        predictor = BloodDonationPredictor()
                        
                        # Prepare data
                        predictor.prepare_historical_data(data)
                        
                        # Make predictions
                        predictions = predictor.predict_multiple_days(start_date, num_days)
                        summary = predictor.get_prediction_summary(predictions)
                        
                        # Success message
                        st.success("🎉 Predictions generated successfully!")
                        
                        # Results section
                        st.markdown("---")
                        st.header("📊 Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Average Prediction", f"{summary['mean_prediction']:.0f}")
                        with col2:
                            st.metric("📊 Prediction Range", f"{summary['min_prediction']:.0f} - {summary['max_prediction']:.0f}")
                        with col3:
                            weekend_effect = summary.get('weekend_effect', 0)
                            st.metric("🏖️ Weekend Effect", f"{weekend_effect:+.0f}" if weekend_effect else "N/A")
                        with col4:
                            baseline_change = summary.get('baseline_comparison', 0)
                            st.metric("📈 vs Historical", f"{baseline_change:+.0f}")
                        
                        # Predictions table
                        st.subheader("📋 Daily Predictions")
                        
                        # Create predictions table
                        pred_table = pd.DataFrame([
                            {
                                'Date': p['date'].strftime('%Y-%m-%d'),
                                'Day': p['date'].strftime('%A'),
                                'Predicted Donations': int(p['prediction']),
                                'Day Type': '🏖️ Weekend' if p['date'].weekday() >= 5 else '💼 Weekday'
                            } for p in predictions
                        ])
                        
                        st.dataframe(pred_table, use_container_width=True, hide_index=True)
                        
                        # Visualization
                        st.subheader("📈 Prediction Visualization")
                        
                        # Prepare data for plotting
                        # Historical data (last 14 days)
                        hist_data = data.tail(14).copy()
                        hist_data['date'] = pd.to_datetime(hist_data['date'])
                        hist_data['type'] = 'Historical'
                        hist_data['day_name'] = hist_data['date'].dt.strftime('%A')
                        
                        # Prediction data
                        pred_data = pd.DataFrame([
                            {
                                'date': p['date'],
                                'daily': p['prediction'],
                                'type': 'Prediction',
                                'day_name': p['date'].strftime('%A')
                            } for p in predictions
                        ])
                        
                        # Combined plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=hist_data['date'],
                            y=hist_data['daily'],
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8)
                        ))
                        
                        # Prediction data
                        fig.add_trace(go.Scatter(
                            x=pred_data['date'],
                            y=pred_data['daily'],
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='#d62728', width=3, dash='dot'),
                            marker=dict(size=10, symbol='diamond')
                        ))
                        
                        # Add vertical line to separate historical and prediction
                        fig.add_vline(
                            x=pred_data['date'].iloc[0],
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Prediction Start"
                        )
                        
                        fig.update_layout(
                            title='Blood Donation Predictions: Historical vs Forecast',
                            xaxis_title='Date',
                            yaxis_title='Daily Blood Donations',
                            hovermode='x unified',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Weekend vs Weekday Analysis
                        st.subheader("🔍 Weekend vs Weekday Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Weekend vs weekday bar chart
                            weekday_avg = summary.get('weekday_avg', 0)
                            weekend_avg = summary.get('weekend_avg', 0)
                            
                            if weekday_avg and weekend_avg:
                                comparison_data = pd.DataFrame({
                                    'Day Type': ['💼 Weekday', '🏖️ Weekend'],
                                    'Average Donations': [weekday_avg, weekend_avg]
                                })
                                
                                fig_bar = px.bar(
                                    comparison_data,
                                    x='Day Type',
                                    y='Average Donations',
                                    color='Day Type',
                                    title='Average Donations: Weekday vs Weekend'
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("Not enough weekend/weekday data for comparison")
                        
                        with col2:
                            # Daily pattern
                            pred_data_with_type = pred_data.copy()
                            pred_data_with_type['day_type'] = pred_data_with_type['date'].apply(
                                lambda x: '🏖️ Weekend' if x.weekday() >= 5 else '💼 Weekday'
                            )
                            
                            fig_pattern = px.bar(
                                pred_data_with_type,
                                x='day_name',
                                y='daily',
                                color='day_type',
                                title='Daily Prediction Pattern',
                                category_orders={'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
                            )
                            st.plotly_chart(fig_pattern, use_container_width=True)
                        
                        # Show components if requested
                        if show_components:
                            st.subheader("🔧 Prediction Components")
                            
                            components_data = []
                            for p in predictions:
                                comp = p['components']
                                components_data.append({
                                    'Date': p['date'].strftime('%Y-%m-%d'),
                                    'Raw Ensemble': f"{p['raw_ensemble']:.0f}",
                                    'Mean Pulled': f"{comp['mean_pulled']:.0f}",
                                    'Day-of-Week': f"{comp['dow_effect']:+.0f}",
                                    'Random': f"{comp['daily_random']:+.0f}",
                                    'Trend': f"{comp['trend_continuation']:+.0f}",
                                    'Final': f"{p['prediction']:.0f}"
                                })
                            
                            components_df = pd.DataFrame(components_data)
                            st.dataframe(components_df, use_container_width=True, hide_index=True)
                        
                        # Confidence analysis
                        if show_confidence:
                            st.subheader("📊 Confidence Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("📏 Coefficient of Variation", f"{summary['coefficient_variation']:.1f}%")
                                st.metric("📊 Standard Deviation", f"{summary['std_prediction']:.0f}")
                            
                            with col2:
                                trend_change = summary.get('trend_change', 0)
                                st.metric("📈 Trend (First to Last)", f"{trend_change:+.0f}")
                                
                                if abs(trend_change) < 50:
                                    st.success("✅ Stable prediction trend")
                                elif abs(trend_change) < 150:
                                    st.warning("⚠️ Moderate trend change")
                                else:
                                    st.error("❌ High trend variation")
                        
                        # Download predictions
                        st.markdown("---")
                        st.subheader("💾 Download Results")
                        
                        # Prepare download data
                        download_data = pd.DataFrame([
                            {
                                'Date': p['date'].strftime('%Y-%m-%d'),
                                'Day_of_Week': p['date'].strftime('%A'),
                                'Predicted_Donations': round(p['prediction'], 0),
                                'Raw_Ensemble': round(p['raw_ensemble'], 2),
                                'XGBoost_Prediction': round(p['xgb_pred'], 2),
                                'LSTM_Prediction': round(p['lstm_pred'], 2)
                            } for p in predictions
                        ])
                        
                        csv_buffer = io.StringIO()
                        download_data.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"blood_donation_predictions_{start_date.strftime('%Y%m%d')}_{num_days}days.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.info("Please check your data format and try again.")
                        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your CSV file format and try again.")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload your historical blood donation data to start making predictions.")
    
    st.markdown("### 📝 Data Requirements")
    st.markdown("""
    Your CSV file should contain **exactly 14 days** of historical data with these columns:
    
    - **date**: Date in YYYY-MM-DD format
    - **blood_o**: O-type blood donations (daily count)
    - **donations_new**: New donor donations (daily count)
    - **donations_regular**: Regular donor donations (daily count)
    - **daily**: Total daily blood donations
    - **max_temperature**: Maximum temperature (Celsius)
    - **is_holiday**: Holiday indicator (0 = regular day, 1 = holiday)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    🩸 Blood Donation Prediction System | Powered by Ensemble ML Models (XGBoost + LSTM + Meta-learner)
</div>
""", unsafe_allow_html=True)