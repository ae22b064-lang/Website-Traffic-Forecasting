# traffic_forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Data Generation (for demonstration purposes) ---
# In a real project, you would load your data from a CSV file.
# We are creating synthetic data to make the script self-contained and runnable.

def generate_synthetic_data():
    """Generates a synthetic time-series dataset with seasonal patterns."""
    dates = pd.date_range(start='2021-06-01', end='2022-06-28', freq='D')
    traffic_data = pd.Series([
        # Simulating a weekly seasonal pattern (higher on weekdays, lower on weekends)
        # Adding a general upward trend
        500 + i * 2 + (100 * (i % 7 < 5)) + (50 * (i % 30 == 0)) for i in range(len(dates))
    ], index=dates)
    
    # Adding some random noise to the data
    traffic_data = traffic_data + pd.Series(np.random.normal(0, 25, len(dates)), index=dates)

    # Let's create a DataFrame that mimics the original article's data structure
    df = pd.DataFrame(traffic_data, columns=['traffic'])
    df.index.name = 'Date'
    df.reset_index(inplace=True)
    return df

# Let's assume the data is in a DataFrame called 'df'
# In a real project, you would load it with something like:
# df = pd.read_csv('website_traffic_data.csv')
# Here we use our synthetic data generator
import numpy as np
df = generate_synthetic_data()

# --- Data Preparation and Analysis ---

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot the historical traffic data to visualize patterns
print("Plotting historical data...")
plt.figure(figsize=(12, 6))
plt.plot(df['traffic'])
plt.title('Historical Website Traffic Data')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.grid(True)
plt.show()

# --- Forecasting with SARIMA ---

# Define the SARIMA model parameters based on the original article
# The parameters (p, d, q) are for the non-seasonal part of the model
# The parameters (P, D, Q, s) are for the seasonal part, with s=7 for a weekly cycle
# Note: In a real project, you would use statistical tests to find optimal parameters.
# For simplicity, we are using the values mentioned in the article's context.
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 7

# Instantiate the SARIMA model
model = SARIMAX(df['traffic'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model to the data
print("Training the SARIMA model...")
results = model.fit(disp=False)
print("Model training complete.")

# --- Predicting Future Traffic ---

# Forecast traffic for the next 50 days
forecast_steps = 50
forecast = results.get_prediction(start=len(df), end=len(df) + forecast_steps - 1)
forecast_values = forecast.predicted_mean
forecast_confidence_intervals = forecast.conf_int()

# Create a date range for the forecasted period
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# --- Visualization of Results ---

print("Plotting results...")
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(df.index, df['traffic'], label='Historical Traffic')

# Plot forecasted data
plt.plot(forecast_dates, forecast_values, label='Forecasted Traffic', color='red')

# Plot the confidence intervals
plt.fill_between(forecast_dates,
                 forecast_confidence_intervals['lower traffic'],
                 forecast_confidence_intervals['upper traffic'],
                 color='pink', alpha=0.3, label='Confidence Interval')

plt.title('Website Traffic Forecasting with SARIMA')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.legend()
plt.grid(True)
plt.show()

print("Script finished. The plot shows historical and forecasted traffic.")
