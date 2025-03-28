import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Data
try:
    data = pd.read_csv('sales_data.csv', parse_dates=['Date'])
except FileNotFoundError:
    # If the CSV file doesn't exist, create a synthetic dataset for demonstration.
    print("CSV file not found. Creating synthetic data...")
    dates = pd.date_range(start="2022-01-01", periods=100, freq='W')
    # Generate synthetic revenue data with an upward trend and some noise.
    revenue = np.linspace(1000, 5000, num=100) + np.random.normal(0, 300, 100)
    data = pd.DataFrame({"Date": dates, "Revenue": revenue})

# 2. Data Preprocessing
data.sort_values('Date', inplace=True)

data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

# Display the first few rows to verify preprocessing steps.
print("Data preview:")
print(data.head())

# 3. Feature Selection and Train-Test Split
X = data['Date_ordinal'].values.reshape(-1, 1)
y = data['Revenue'].values

# Split the dataset into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the revenue for test set
y_pred = model.predict(X_test)

# Calculate model performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], y, color='blue', label='Actual Revenue', alpha=0.6)

# For a smooth line, predict across the full date range
all_dates = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
all_dates_ordinal = all_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
all_predictions = model.predict(all_dates_ordinal)

plt.plot(all_dates, all_predictions, color='red', linewidth=2, label='Predicted Revenue Trend')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Sales and Revenue Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Future Revenue Prediction
future_date = data['Date'].max() + pd.Timedelta(days=30)
future_date_ordinal = np.array([[future_date.toordinal()]])
future_prediction = model.predict(future_date_ordinal)
print(f"\nPredicted Revenue for {future_date.date()}: ${future_prediction[0]:.2f}")
