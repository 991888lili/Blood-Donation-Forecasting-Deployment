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
        padding: 2rem;
        background: linear-gradient(135deg, #ffebee 0%, #f3e5f5 100%);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .block-container {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
            
    .js-plotly-plot {
        background: rgba(248, 250, 252, 0.8) !important;
        border-radius: 12px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
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
            # Model weights visualization
            weights = config.get('meta_learner_weights', {})
            if weights:
                st.markdown("**⚖️ Model Weights:**")
                xgb_weight = weights.get('xgb_weight', 0) * 100
                lstm_weight = weights.get('lstm_weight', 0) * 100

                st.progress(lstm_weight / 100, text=f"LSTM: {lstm_weight:.1f}%")
                st.progress(xgb_weight / 100, text=f"XGBoost: {xgb_weight:.1f}%")
    except:
        st.write("Model info not available")

# Main content
col1, col2 = st.columns([1, 1])


with col1:
    st.header("🎉 Welcome to use")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                padding: 1.5rem; 
                border-radius: 12px; 
                border-left: 5px solid #1976D2; 
                margin: 1rem 0;">
        <p style="color: #666; font-size: 0.95rem; margin-bottom: 0;">
            💼 <em>Developed for NBC (National Blood Center) - Optimizing blood collection operations through AI-powered insights</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.subheader("📁 Upload Your Data")
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
        options=[1, 3, 5, 7],
        index=3,  # Default to 7 days
        help="How many consecutive days to predict"
    )
    
    st.subheader("🔧 Advanced Settings")
    with st.expander("Model Parameters (Optional)"):
        show_components = st.checkbox("Show Prediction Components", value=False)
        show_confidence = st.checkbox("Show Confidence Analysis", value=True)
    
# System Status 
st.subheader("📊 System Status")
status_col1, status_col2, status_col3, status_col4 = st.columns([1,1,1,1])
with status_col1:
    st.success("🟢 Models Loaded")

with status_col2:
    st.success("🟢 System Ready")

with status_col3:
    st.info("🔄 Awaiting Data")

with status_col4:
    st.info("⏳ Ready to Predict")

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
                        predictions = predictor.predict_multiple_days(start_date.strftime('%Y-%m-%d'), num_days)
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
                                'Day Type': '🏖️ Weekend' if p['date'].weekday() >= 5 else '💼 Weekday',
                                'Priority Level': (
                                    "🔴 Critical" if p['prediction'] < 450
                                    else "🟡 Moderate" if p['prediction'] < 550
                                    else "🟢 Good"
                                )
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
                        
                        # Add vertical line to separate historical and prediction (commented out)
                        # prediction_start_date = pred_data['date'].iloc[0]
                        # if hasattr(prediction_start_date, 'strftime'):
                        #     prediction_start_str = prediction_start_date.strftime('%Y-%m-%d')
                        # else:
                        #     prediction_start_str = str(prediction_start_date)
                        # fig.add_vline(
                        #     x=prediction_start_str,
                        #     line_dash="dash",
                        #     line_color="gray",
                        #     annotation_text="Prediction Start"
                        # )
                        
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
                            pred_data_with_type['Day_type'] = pred_data_with_type['date'].apply(
                                lambda x: '🏖️ Weekend' if x.weekday() >= 5 else '💼 Weekday'
                            )
                            
                            fig_pattern = px.bar(
                                pred_data_with_type,
                                x='Day_name',
                                y='daily',
                                color='Day_type',
                                title='Daily Prediction Pattern',
                                category_orders={'Day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
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
                                    st.info("📊 Moderate trend change")
                                else:
                                    st.info("📈 Dynamic trend pattern")
                        
                        # Blood donation activity recommendations
                        st.markdown("---")
                        st.subheader("💡 Blood Donation Activity Recommendations")

                        # Calculate recommendations based on predictions
                        pred_values = [p['prediction'] for p in predictions]
                        min_day_idx = pred_values.index(min(pred_values))
                        max_day_idx = pred_values.index(max(pred_values))

                        min_day = predictions[min_day_idx]['date']
                        max_day = predictions[max_day_idx]['date']

                        weekend_days = [p for p in predictions if p['date'].weekday() >= 5]
                        weekday_days = [p for p in predictions if p['date'].weekday() < 5]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🎯 Priority Actions")
                            
                            # High priority recommendations
                            st.markdown("**🔴 High Priority:**")
                            st.markdown(f"• **{min_day.strftime('%A, %B %d')}**: Lowest predicted donations ({pred_values[min_day_idx]:.0f})")
                            st.markdown("  - Strengthen promotional activities")
                            st.markdown("  - Reduce staffing arrangements")
                            st.markdown("  - Contact regular donors")
                            
                            # Medium priority
                            st.markdown("**🟡 Medium Priority:**")
                            if weekend_days:
                                weekend_avg = np.mean([p['prediction'] for p in weekend_days])
                                st.markdown(f"• **Weekend Strategy**: Average {weekend_avg:.0f} donations expected")
                                st.markdown("  - Extend operating hours")
                                st.markdown("  - Prepare additional staff")

                        with col2:
                            st.markdown("#### 📅 Weekly Strategy")
                            
                            # Best collection day
                            st.markdown("**🎉 Peak Collection Day:**")
                            st.markdown(f"• **{max_day.strftime('%A, %B %d')}**: Highest predicted donations ({pred_values[max_day_idx]:.0f})")
                            st.markdown("  - Ensure adequate blood bags")
                            st.markdown("  - Schedule maximum staff")
                            st.markdown("  - Schedule inventory for fewer dates")
                            
                            # Resource allocation
                            total_predicted = sum(pred_values)
                            st.markdown("**📦 Resource Planning:**")
                            st.markdown(f"• **Weekly Total**: {total_predicted:.0f} donations expected")
                            st.markdown(f"• **Daily Average**: {total_predicted/len(pred_values):.0f} donations")
                            st.markdown("  - Blood bag inventory: +20% buffer")
                            st.markdown("  - Staff scheduling: Peak on weekends")


                        # Summary insights
                        st.markdown("#### 🔍 Key Insights")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            weekend_effect = summary.get('weekend_effect', 0)
                            if weekend_effect > 100:
                                st.success(f"**Strong Weekend Boost**: +{weekend_effect:.0f} donations")
                                st.markdown("💡 Focus weekend marketing")
                            else:
                                st.info("**Consistent Pattern**: Similar weekday/weekend donations")

                        with col2:
                            cv = summary['coefficient_variation']
                            if cv < 20:
                                st.success("**Stable Predictions**: Low variation")
                            elif cv < 30:
                                st.info("**Moderate Variation**: Plan flexibility")
                            else:
                                st.warning("**High Variation**: Prepare contingency")

                        with col3:
                            baseline_change = summary.get('baseline_comparison', 0)
                            if baseline_change > 50:
                                st.success(f"**Above Historical**: +{baseline_change:.0f}")
                                st.markdown("📈 Positive trend")
                            elif baseline_change < -50:
                                st.error(f"**Below Historical**: {baseline_change:.0f}")
                                st.markdown("⚠️ Requires action")
                            else:
                                st.info("**Near Historical Average**")
                        
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
                        import traceback
                        full_error = traceback.format_exc()
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.code(full_error)
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