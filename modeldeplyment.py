import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/xiaolongluo/Documents/PPMI/ppmi_env_py312/alphaTrials/')
from alphaTrials.utils import (
    load_testing_data, normalize_testing_data, combine_testing_data,
    extend_testing_data_with_future_timepoints, impute_testing_data_with_mmrm_models,
    load_transformer_model, convert_features_to_tensor, convert_outcomes_to_tensor,
    convert_imputation_status_to_tensor, create_weight_tensor, get_prediction,
    reverse_scale_predictions
)

def run_prediction_pipeline(baseline_df, longitudinal_df):
    """
    Run the complete prediction pipeline on baseline and longitudinal data
    
    Args:
        baseline_df: DataFrame with baseline data
        longitudinal_df: DataFrame with longitudinal data
    
    Returns:
        dict: Dictionary containing prediction results
    """
    st.info("🚀 Starting prediction pipeline...")
    
    # Define variables
    ivar = ['PATNO']
    tvar = ['year']
    Xt = ['LEDD', 'year']
    ynum = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
    tx = ['COHORT_DEFINITION']
    
    # Define variable lists for normalization
    Xcat = ['SEX', 'CHLDBEAR_missing', 'CHLDBEAR', 'HOWLIVE_missing', 'HOWLIVE']
    Xnum = ['ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT']
    
    # Normalize the data using saved scalers
    st.info("📊 Normalizing data...")
    baseline_normalized, dfy_normalized = normalize_testing_data(
        baseline_df, longitudinal_df, Xcat, Xnum, tx, ynum, Xt
    )
    
    # Combine baseline and longitudinal data
    st.info("🔗 Combining baseline and longitudinal data...")
    dfy_with_baseline = combine_testing_data(baseline_normalized, dfy_normalized, ivar, tx)
    
    # Extend dataset with future timepoints
    st.info("⏰ Extending dataset with future timepoints...")
    dfy_extended = extend_testing_data_with_future_timepoints(
        dfy_with_baseline, tvar, ynum, kmax=41, tdelta=0.5
    )
    
    # Impute missing values using saved MMRM models
    st.info("🔮 Imputing missing values with MMRM models...")
    dfy_with_imputed = impute_testing_data_with_mmrm_models(
        dfy_extended, ynum, ivar, Xt
    )
    
    # Generate predictions
    st.info("🤖 Generating predictions...")
    
    # Load the transformer model
    model = load_transformer_model()
    df = dfy_with_imputed.copy()
    
    # Define feature columns
    Xcol = [
        'LEDD', 'year', 'ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT',
        'SEX_10', 'CHLDBEAR_missing_1', 'CHLDBEAR_1.0', 'CHLDBEAR_999.0',
        'HOWLIVE_missing_1', 'HOWLIVE_2.0', 'HOWLIVE_4.0', 'HOWLIVE_999.0',
        'COHORT_DEFINITION_encoded_Parkinsons_Disease', 'COHORT_DEFINITION_encoded_Prodromal', 
        'COHORT_DEFINITION_encoded_SWEDD'
    ]
    
    # Define outcome columns
    iynum = [f'{outcome}_imputed' for outcome in ynum]
    isynum = [f'{outcome}_is_nan' for outcome in ynum]
    
    # Ensure all expected feature columns exist (set missing ones to zero)
    st.info("🔧 Ensuring all expected feature columns exist...")
    missing_columns = [col for col in Xcol if col not in df.columns]
    if missing_columns:
        st.write(f"   Adding missing columns with zero values: {missing_columns}")
        for col in missing_columns:
            df[col] = 0.0
    
    # Convert features to tensor using TensorFlow
    X_tensor = convert_features_to_tensor(df, Xcol)
    
    # Convert outcomes to tensor
    y_tensor = convert_outcomes_to_tensor(df, iynum)
    
    # Convert imputation status to tensor
    mask_tensor = convert_imputation_status_to_tensor(df, isynum)
    
    # Create weight tensor based on cohort and propensity scores
    weight_tensor = create_weight_tensor(df)
    
    df1 = get_prediction(df, X_tensor, y_tensor, model, ynum)
    
    # Reverse scale the predictions back to original scale
    df1_original_scale = reverse_scale_predictions(df1)
    
    st.success("✅ All processing complete!")
    
    return {
        'dfy_with_imputed': dfy_with_imputed,
        'df1_scaled': df1,
        'df1_original_scale': df1_original_scale,
        'Xcol': Xcol,
        'iynum': iynum,
        'isynum': isynum
    }

def plot_predictions_streamlit(df, ynum):
    """
    Plot original and predicted values with confidence intervals for each outcome variable
    
    Args:
        df: DataFrame with original and predicted data
        ynum: List of outcome variable names
    """
    fig, axes = plt.subplots(1, len(ynum), figsize=(6*len(ynum), 6))
    if len(ynum) == 1:
        axes = [axes]
    
    fig.suptitle('Original vs Predicted Values with Confidence Intervals', fontsize=16, fontweight='bold')
    
    for i, outcome in enumerate(ynum):
        ax = axes[i]
        
        # Plot original values
        original_mask = df[f'{outcome}_is_nan'] == 0
        if original_mask.sum() > 0:
            original_years = df.loc[original_mask, 'year']
            original_values = df.loc[original_mask, outcome]
            ax.scatter(original_years, original_values, color='blue', s=50, alpha=0.7, 
                      label='Original', zorder=3)
        
        # Plot predicted values
        pred_years = df['year']
        pred_values = df[f'{outcome}_pred']
        ax.plot(pred_years, pred_values, color='red', linewidth=2, 
                label='Predicted', zorder=2)
        
        # Plot confidence intervals
        lower_values = df[f'{outcome}_lower']
        upper_values = df[f'{outcome}_upper']
        ax.fill_between(pred_years, lower_values, upper_values, 
                       color='red', alpha=0.2, label='95% CI', zorder=1)
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{outcome}')
        ax.set_title(f'{outcome} Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

st.set_page_config(
    page_title="PPMI Data Entry",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 PPMI Data Entry")
st.markdown("Simple data entry for baseline and longitudinal observations")

# Initialize session state
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = {}
if 'longitudinal_data' not in st.session_state:
    st.session_state.longitudinal_data = []

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Baseline Data", "📊 Longitudinal Data", "📋 Summary", "🤖 Predictions"])

# Tab 1: Baseline Data Entry
with tab1:
    st.header("🏠 Baseline Data Entry")
    st.markdown("Enter participant baseline information")
    
    # Create form for baseline data
    with st.form("baseline_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patno = st.text_input("Participant ID (PATNO)", value="101")
            enroll_age = st.number_input("Age at Enrollment", min_value=18, max_value=100, value=65)
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
            childbear = st.selectbox("Female of child bearing potential", options=["Yes", "No"], index=1)
        
        with col2:
            cohort = st.selectbox("Cohort Definition", 
                                options=["SWEDD", "Parkinson's Disease", "Prodromal"], 
                                index=0)
            ledd = st.number_input("LEDD (Levodopa Equivalent Daily Dose)", 
                                 min_value=0.0, max_value=2000.0, value=0.0, step=10.0)
            howlive = st.number_input("How do you live your life day to day?", 
                                    min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        
        # MDS-UPDRS scores
        st.subheader("MDS-UPDRS Scores")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            np1ptot = st.number_input("NP1PTOT (Part I)", min_value=0.0, max_value=52.0, value=10.0)
        with col4:
            np2ptot = st.number_input("NP2PTOT (Part II)", min_value=0.0, max_value=52.0, value=10.0)
        with col5:
            np3ptot = st.number_input("NP3TOT (Part III)", min_value=0.0, max_value=132.0, value=20.0)
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Baseline Data", type="primary")
        
        if submitted:
            # Store baseline data
            st.session_state.baseline_data = {
                'PATNO': patno,
                'ENROLL_AGE': enroll_age,
                'SEX': sex,
                'CHLDBEAR': childbear,
                'COHORT_DEFINITION': cohort,
                'LEDD': ledd,
                'HOWLIVE': howlive,
                'NP1PTOT': np1ptot,
                'NP2PTOT': np2ptot,
                'NP3TOT': np3ptot,
                'year': 0  # Baseline is year 0
            }
            st.success("✅ Baseline data saved successfully!")
    
    # Show current baseline data
    if st.session_state.baseline_data:
        st.subheader("📋 Current Baseline Data")
        baseline_df = pd.DataFrame([st.session_state.baseline_data])
        st.dataframe(baseline_df, use_container_width=True)

# Tab 2: Longitudinal Data Entry
with tab2:
    st.header("📊 Longitudinal Data Entry")
    st.markdown("Add follow-up observations")
    
    if not st.session_state.baseline_data:
        st.warning("⚠️ Please enter baseline data first in the 'Baseline Data' tab.")
    else:
        st.info(f"✅ Baseline data available for participant: {st.session_state.baseline_data['PATNO']}")
        
        # Number of observations
        num_obs = st.number_input("Number of follow-up observations", 
                                min_value=1, max_value=10, value=3)
        
        # Create form for longitudinal data
        with st.form("longitudinal_form"):
            st.subheader("📈 Follow-up Observations")
            
            for i in range(num_obs):
                st.write(f"**Observation {i+1}**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    year = st.number_input(f"Year {i+1}", min_value=0.1, max_value=10.0, 
                                         value=float(i+1), step=0.1, key=f"year_{i}")
                with col2:
                    ledd_obs = st.number_input(f"LEDD {i+1}", min_value=0.0, max_value=2000.0, 
                                             value=0.0, step=10.0, key=f"ledd_{i}")
                with col3:
                    np1_obs = st.number_input(f"NP1PTOT {i+1}", min_value=0.0, max_value=52.0, 
                                            value=10.0, key=f"np1_{i}")
                with col4:
                    np2_obs = st.number_input(f"NP2PTOT {i+1}", min_value=0.0, max_value=52.0, 
                                            value=10.0, key=f"np2_{i}")
                
                col5 = st.columns(1)[0]
                with col5:
                    np3_obs = st.number_input(f"NP3TOT {i+1}", min_value=0.0, max_value=132.0, 
                                            value=20.0, key=f"np3_{i}")
            
            # Submit button
            submitted_long = st.form_submit_button("💾 Save Longitudinal Data", type="primary")
            
            if submitted_long:
                # Store longitudinal data
                longitudinal_obs = []
                
                for i in range(num_obs):
                    obs = {
                        'PATNO': st.session_state.baseline_data['PATNO'],
                        'year': st.session_state[f"year_{i}"],
                        'LEDD': st.session_state[f"ledd_{i}"],
                        'NP1PTOT': st.session_state[f"np1_{i}"],
                        'NP2PTOT': st.session_state[f"np2_{i}"],
                        'NP3TOT': st.session_state[f"np3_{i}"]
                    }
                    longitudinal_obs.append(obs)
                
                st.session_state.longitudinal_data = longitudinal_obs
                st.success(f"✅ {num_obs} longitudinal observations saved successfully!")
        
        # Show current longitudinal data
        if st.session_state.longitudinal_data:
            st.subheader("📋 Current Longitudinal Data")
            long_df = pd.DataFrame(st.session_state.longitudinal_data)
            st.dataframe(long_df, use_container_width=True)

# Tab 3: Summary
with tab3:
    st.header("📋 Data Summary")
    
    if st.session_state.baseline_data and st.session_state.longitudinal_data:
        baseline_processed, dfy, ivar, tvar, Xt, ynum, tx = load_testing_data(
            baseline_data=st.session_state.baseline_data,
            longitudinal_data=st.session_state.longitudinal_data
        )
        
        # Define variable lists for normalization (these should match what normalize_testing_data expects)
        Xcat = ['SEX', 'CHLDBEAR_missing', 'CHLDBEAR', 'HOWLIVE_missing', 'HOWLIVE']
        Xnum = ['ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT']
        
        # Normalize the testing data using saved scalers
        baseline_normalized, dfy_normalized = normalize_testing_data(
            baseline_processed, dfy, Xcat, Xnum, tx, ynum, Xt
        )
        
        # Create combined dataset for display
        baseline_df = pd.DataFrame([st.session_state.baseline_data])
        longitudinal_df = pd.DataFrame(st.session_state.longitudinal_data)
        
        combined_df = longitudinal_df.merge(
            baseline_df[['PATNO', 'ENROLL_AGE', 'SEX', 'CHLDBEAR', 'COHORT_DEFINITION', 'HOWLIVE']], 
            on='PATNO', 
            how='left'
        )
        
        # Add baseline observation (year 0) to the combined dataset
        baseline_obs = baseline_df.copy()
        combined_df = pd.concat([baseline_obs, combined_df], ignore_index=True)
        
        # Sort by PATNO and year
        combined_df = combined_df.sort_values(['PATNO', 'year']).reset_index(drop=True)
        
        
        st.subheader("📊 Complete Dataset")
        st.dataframe(combined_df, use_container_width=True)
        
        # Download CSV files
        st.subheader("💾 Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # Save baseline data to CSV file
            import os
            export_dir = "/Users/xiaolongluo/Documents/PPMI/data_exports"
            os.makedirs(export_dir, exist_ok=True)
            
            baseline_filename = f"baseline_data_{st.session_state.baseline_data['PATNO']}.csv"
            baseline_filepath = os.path.join(export_dir, baseline_filename)
            baseline_df.to_csv(baseline_filepath, index=False)
            
            # Download baseline data as CSV
            baseline_csv = baseline_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Baseline Data (CSV)",
                data=baseline_csv,
                file_name=baseline_filename,
                mime="text/csv",
                help="Download the baseline data for this participant"
            )
            st.success(f"✅ Baseline data saved to: {baseline_filepath}")
        
        with col2:
            # Save longitudinal data to CSV file
            longitudinal_filename = f"longitudinal_data_{st.session_state.baseline_data['PATNO']}.csv"
            longitudinal_filepath = os.path.join(export_dir, longitudinal_filename)
            longitudinal_df.to_csv(longitudinal_filepath, index=False)
            
            # Download longitudinal data as CSV
            longitudinal_csv = longitudinal_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Longitudinal Data (CSV)",
                data=longitudinal_csv,
                file_name=longitudinal_filename,
                mime="text/csv",
                help="Download the longitudinal data for this participant"
            )
            st.success(f"✅ Longitudinal data saved to: {longitudinal_filepath}")
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Observations", len(combined_df))
            st.metric("Participant ID", st.session_state.baseline_data['PATNO'])
        
        with col2:
            st.metric("Years Covered", f"{combined_df['year'].min():.1f} - {combined_df['year'].max():.1f}")
            st.metric("Baseline Age", f"{st.session_state.baseline_data['ENROLL_AGE']} years")
        
        with col3:
            st.metric("Cohort", st.session_state.baseline_data['COHORT_DEFINITION'])
            st.metric("Sex", st.session_state.baseline_data['SEX'])
        
        # Score ranges
        st.subheader("📊 Score Ranges")
        score_cols = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
        for col in score_cols:
            if col in combined_df.columns:
                min_val = combined_df[col].min()
                max_val = combined_df[col].max()
                st.write(f"**{col}**: {min_val:.1f} - {max_val:.1f}")
        
        
    elif st.session_state.baseline_data:
        st.info("ℹ️ Baseline data available. Add longitudinal observations in the 'Longitudinal Data' tab.")
        baseline_df = pd.DataFrame([st.session_state.baseline_data])
        st.subheader("📊 Baseline Data Only")
        st.dataframe(baseline_df, use_container_width=True)
        
        # Show what the combined dataset will look like
        st.subheader("🔮 Preview: Combined Dataset Structure")
        st.write("When you add longitudinal data, the combined dataset will have:")
        st.write("- **Baseline information** (ENROLL_AGE, SEX, CHLDBEAR, COHORT_DEFINITION) available for all time points")
        st.write("- **Time-varying data** (year, LEDD, NP1PTOT, NP2PTOT, NP3TOT) for each observation")
        st.write("- **Sorted by PATNO and year** for proper longitudinal analysis")
        
    else:
        st.info("ℹ️ No data entered yet. Please start with the 'Baseline Data' tab.")

# Tab 4: Predictions
with tab4:
    st.header("🤖 Predictions")
    
    if st.session_state.baseline_data and st.session_state.longitudinal_data:
        st.markdown("Generate predictions and visualize disease progression")
        
        # Convert session state data to DataFrames
        baseline_df = pd.DataFrame([st.session_state.baseline_data])
        longitudinal_df = pd.DataFrame(st.session_state.longitudinal_data)
        
        # Add SEX_10 column to baseline data
        if 'SEX' in baseline_df.columns:
            if baseline_df['SEX'].dtype == 'object':
                baseline_df['SEX_10'] = (baseline_df['SEX'] == 'Male').astype(int)
            else:
                baseline_df['SEX_10'] = (baseline_df['SEX'] == 1.0).astype(int)
        
        # Add 'b_' prefix to ynum columns in baseline data
        ynum_columns = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
        for col in ynum_columns:
            if col in baseline_df.columns:
                baseline_df[f'b_{col}'] = baseline_df[col]
        
        # Button to run predictions
        if st.button("🚀 Run Prediction Pipeline", type="primary"):
            try:
                # Run prediction pipeline
                with st.spinner("Running prediction pipeline... This may take a few minutes."):
                    results = run_prediction_pipeline(baseline_df, longitudinal_df)
                
                # Store results in session state
                st.session_state.prediction_results = results
                
                st.success("✅ Prediction pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error running prediction pipeline: {str(e)}")
                st.exception(e)
        
        # Display results if available
        if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
            results = st.session_state.prediction_results
            df_final = results['df1_original_scale']
            
            st.subheader("📊 Prediction Results")
            
            # Display prediction summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("NP1PTOT Range", 
                         f"{df_final['NP1PTOT_pred'].min():.1f} - {df_final['NP1PTOT_pred'].max():.1f}")
                st.metric("NP1PTOT Mean", f"{df_final['NP1PTOT_pred'].mean():.1f}")
            
            with col2:
                st.metric("NP2PTOT Range", 
                         f"{df_final['NP2PTOT_pred'].min():.1f} - {df_final['NP2PTOT_pred'].max():.1f}")
                st.metric("NP2PTOT Mean", f"{df_final['NP2PTOT_pred'].mean():.1f}")
            
            with col3:
                st.metric("NP3TOT Range", 
                         f"{df_final['NP3TOT_pred'].min():.1f} - {df_final['NP3TOT_pred'].max():.1f}")
                st.metric("NP3TOT Mean", f"{df_final['NP3TOT_pred'].mean():.1f}")
            
            # Display plots
            st.subheader("📈 Prediction Plots")
            try:
                fig = plot_predictions_streamlit(df_final, ['NP1PTOT', 'NP2PTOT', 'NP3TOT'])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error creating plots: {str(e)}")
            
            # Display prediction data
            st.subheader("📋 Prediction Data")
            
            # Select columns to display
            display_columns = ['PATNO', 'year', 'NP1PTOT', 'NP2PTOT', 'NP3TOT', 
                             'NP1PTOT_pred', 'NP2PTOT_pred', 'NP3TOT_pred',
                             'NP1PTOT_lower', 'NP2PTOT_lower', 'NP3TOT_lower',
                             'NP1PTOT_upper', 'NP2PTOT_upper', 'NP3TOT_upper']
            
            available_columns = [col for col in display_columns if col in df_final.columns]
            prediction_display = df_final[available_columns].copy()
            
            st.dataframe(prediction_display, use_container_width=True)
            
            # Download predictions
            st.subheader("💾 Download Predictions")
            prediction_csv = prediction_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Results (CSV)",
                data=prediction_csv,
                file_name=f"predictions_{st.session_state.baseline_data['PATNO']}.csv",
                mime="text/csv",
                help="Download the prediction results for this participant"
            )
    
    elif st.session_state.baseline_data:
        st.warning("⚠️ Please enter longitudinal data in the 'Longitudinal Data' tab before running predictions.")
    else:
        st.info("ℹ️ Please enter baseline and longitudinal data before running predictions.")

# Sidebar
with st.sidebar:
    st.header("📊 Data Status")
    
    if st.session_state.baseline_data:
        st.success("✅ Baseline data entered")
        st.write(f"Participant: {st.session_state.baseline_data['PATNO']}")
    else:
        st.warning("⚠️ No baseline data")
    
    if st.session_state.longitudinal_data:
        st.success(f"✅ {len(st.session_state.longitudinal_data)} longitudinal observations")
    else:
        st.warning("⚠️ No longitudinal data")
    
    if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
        st.success("✅ Predictions available")
    else:
        st.info("ℹ️ No predictions yet")
    
    # Clear data button
    if st.button("🗑️ Clear All Data", key="clear_data_button"):
        st.session_state.baseline_data = {}
        st.session_state.longitudinal_data = []
        if hasattr(st.session_state, 'prediction_results'):
            del st.session_state.prediction_results
        st.rerun()