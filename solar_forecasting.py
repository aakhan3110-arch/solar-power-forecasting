import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. ENVIRONMENTAL TELEMETRY GENERATION
# ==========================================
def generate_solar_data(days=30):
    """
    Generates synthetic solar plant telemetry.
    Models physical properties like diurnal cycles, cloud cover attenuation,
    and cell temperature degradation coefficients.
    """
    print(f"[*] Simulating solar plant telemetry for {days} days...")
    
    # 15-minute intervals over the specified day range
    timestamps = pd.date_range(start="2026-06-01", periods=days * 24 * 4, freq="15min")
    num_intervals = len(timestamps)
    
    # Extract hour of the day to simulate the sun's path (diurnal cycle)
    hour = timestamps.hour + timestamps.minute / 60.0
    
    # Model Solar Irradiance (W/m²) using a sine wave scaled for daylight hours
    # Peak irradiance at noon (~12:00 PM), zero during night hours
    base_irradiance = 800 * np.sin(np.pi * (hour - 6) / 12)
    base_irradiance[base_irradiance < 0] = 0  # No sun before 6 AM or after 6 PM
    
    # Inject stochastic environmental factors
    cloud_cover = np.random.beta(a=2, b=5, size=num_intervals) * 100  # Percentage (%)
    irradiance_attenuation = 1 - (0.75 * (cloud_cover / 100))
    actual_irradiance = base_irradiance * irradiance_attenuation + np.random.normal(0, 15, num_intervals)
    actual_irradiance[actual_irradiance < 0] = 0
    
    # Model Ambient Temperature (°C) correlating with solar peak + thermal lag
    ambient_temp = 22.0 + 12.0 * np.sin(np.pi * (hour - 8) / 12) + np.random.normal(0, 1.5, num_intervals)
    
    # Model PV Cell Temperature (increases under high irradiance)
    cell_temp = ambient_temp + (actual_irradiance * 0.03)
    
    # Calculate Power Output (kW) using a non-linear physical degradation formula:
    # Power = Irradiance * Area * Efficiency * (1 - Temp_Coefficient * (T_cell - T_ref))
    panel_area = 150  # Square meters
    base_efficiency = 0.18
    temp_coefficient = 0.004  # 0.4% efficiency loss per °C above 25°C standard
    
    thermal_loss = 1 - temp_coefficient * (cell_temp - 25.0)
    ideal_power = (actual_irradiance * panel_area * base_efficiency) / 1000 # Convert W to kW
    power_output = ideal_power * thermal_loss
    power_output[power_output < 0] = 0
    
    # Build DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Hour': hour,
        'Cloud_Cover_Pct': cloud_cover,
        'Ambient_Temp_C': ambient_temp,
        'Irradiance_W_m2': actual_irradiance,
        'Cell_Temp_C': cell_temp,
        'Power_Output_kW': power_output
    })
    
    print(f"[+] Successfully compiled {len(df)} telemetry tracking intervals.")
    return df

# ==========================================
# 2. MODEL ENGINEERING & TRAINING
# ==========================================
def train_forecasting_model(df):
    """
    Trains a Gradient Boosting Regressor to predict solar power output
    based on dynamic environmental variables.
    """
    print("\n[*] Initializing Gradient Boosting Regression Pipeline...")
    
    # Features and Target
    X = df[['Hour', 'Cloud_Cover_Pct', 'Ambient_Temp_C', 'Irradiance_W_m2', 'Cell_Temp_C']]
    y = df['Power_Output_kW']
    
    # Train-test split (Chronological sequencing via shuffle=False can also be used, 
    # but stratified random split balances varying weather states well)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n=== Model Evaluation Performance ===")
    print(f" * Mean Absolute Error (MAE):    {mae:.4f} kW")
    print(f" * Root Mean Squared Error (RMSE): {rmse:.4f} kW")
    print(f" * Coefficient of Determination ($R^2$): {r2:.4f}")
    
    return model, X_test, y_test, y_pred

# ==========================================
# 3. ANALYTICAL FORECAST VISUALIZATION
# ==========================================
def plot_forecasting_dashboard(df, model, X_test, y_test, y_pred):
    """
    Compiles an analytical dashboard comparing predictions against physical baselines.
    """
    print("\n[*] Plotting Forecasting Performance Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Solar Power Generation Forecasting Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Time-Series Horizon Comparison (48 hours of consecutive data)
    # Filter a continuous sequence out of the raw dataframe
    horizon_df = df.iloc[400:592].copy() # 48 hours @ 15-min intervals
    X_horizon = horizon_df[['Hour', 'Cloud_Cover_Pct', 'Ambient_Temp_C', 'Irradiance_W_m2', 'Cell_Temp_C']]
    horizon_df['Predicted_Power_kW'] = model.predict(X_horizon)
    
    axes[0, 0].plot(horizon_df['Timestamp'], horizon_df['Power_Output_kW'], label='Actual Power', color='black', alpha=0.8, linewidth=2)
    axes[0, 0].plot(horizon_df['Timestamp'], horizon_df['Predicted_Power_kW'], label='AI Predicted Power', color='gold', linestyle='--', linewidth=2)
    axes[0, 0].set_title('48-Hour Continuous Forecasting Horizon Verification')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Power Generation (kW)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=20)
    axes[0, 0].grid(True, linestyle=':')
    
    # Plot 2: True vs. Predicted Scatter
    axes[0, 1].scatter(y_test, y_pred, alpha=0.3, color='teal', edgecolor='none')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_title('Regression Parity Model Mapping')
    axes[0, 1].set_xlabel('Actual Power Target (kW)')
    axes[0, 1].set_ylabel('Predicted Power Target (kW)')
    axes[0, 1].grid(True, linestyle=':')
    
    # Plot 3: Feature Importances
    importances = model.feature_importances_
    features = X_test.columns
    indices = np.argsort(importances)
    axes[1, 0].barh(range(len(indices)), importances[indices], color='darkorange', align='center')
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([features[i] for i in indices])
    axes[1, 0].set_title('Feature Importance Permutation')
    axes[1, 0].set_xlabel('Relative Weight Split Score')
    
    # Plot 4: Prediction Residual Distribution Error
    residuals = y_test - y_pred
    axes[1, 1].hist(residuals, bins=40, color='crimson', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    axes[1, 1].set_title('Residual Forecast Error Distribution Variance')
    axes[1, 1].set_xlabel('Error Magnitude (Actual - Predicted kW)')
    axes[1, 1].set_ylabel('Sampling Density Frequency')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    solar_df = generate_solar_data(days=45)
    model, X_test, y_test, y_pred = train_forecasting_model(solar_df)
    plot_forecasting_dashboard(solar_df, model, X_test, y_test, y_pred)
