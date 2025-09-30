# Ex.No: 6               HOLT WINTERS METHOD
### Date: 30-09-2025
### Name: K MADHAVA REDDY
### Register No: 212223240064


### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# 1. Load the avocado dataset
file_path = "/content/avocado.csv"
data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
data.sort_index(inplace=True)

# 2. Preprocess the data
# Aggregate data to get a single average price per day
daily_data = data.groupby('Date')['AveragePrice'].mean()
# Resample to weekly frequency to create a clean time series
weekly_data = daily_data.resample('W').mean()
weekly_data = weekly_data.dropna() # Drop any weeks with no data

# 3. Split data into training and testing sets (90% train, 10% test)
train_data = weekly_data[:int(0.9 * len(weekly_data))]
test_data = weekly_data[int(0.9 * len(weekly_data)):]

# 4. Fit the Holt-Winters model
# Using seasonal_periods=52 for weekly data to capture yearly seasonality
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=52
).fit()

# 5. Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# 6. Plot the training, testing, and predicted data
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title('Avocado Prices: Train, Test, and Predicted (Holt-Winters)')
plt.ylabel('Average Price (USD)')
plt.show()

# 7. Evaluate the model's performance on the test set
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error (MAE) = {mae:.4f}")
print(f"Mean Squared Error (MSE) = {mse:.4f}")

# 8. Fit the model to the ENTIRE dataset and forecast the future
final_model = ExponentialSmoothing(
    weekly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=52
).fit()

# Forecast 26 weeks (about 6 months) into the future
forecast_predictions = final_model.forecast(steps=26)

# 9. Plot the original data and the future forecast
plt.figure(figsize=(12, 8))
weekly_data.plot(legend=True, label='Original Data')
forecast_predictions.plot(legend=True, label='Forecasted Data', color='red')
plt.title('Avocado Prices: Original and Forecasted (Holt-Winters)')
plt.ylabel('Average Price (USD)')
plt.show()
```
### OUTPUT:


### TEST_PREDICTION
<img width="955" height="659" alt="image" src="https://github.com/user-attachments/assets/40b8d606-85bf-4580-8055-4219e648e09a" />



### FINAL_PREDICTION
<img width="989" height="697" alt="image" src="https://github.com/user-attachments/assets/1b28bab2-2272-44c5-973d-198edfca969d" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
