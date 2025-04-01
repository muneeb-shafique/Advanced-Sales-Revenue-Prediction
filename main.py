#!/usr/bin/env python3
"""
Advanced Sales and Revenue Prediction Project

This script demonstrates an advanced approach to forecasting revenue
using two models:
  1. A multivariate Linear Regression model with feature engineering.
  2. A univariate ARIMA model for time series forecasting.

The script includes data preprocessing, cross validation, model evaluation,
and detailed visualizations with added hyperparameter tuning for ARIMA.

Author: Muneeb
Date: 2025-04-01
"""

import logging
import sys
import os
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore")

class SalesRevenuePredictor:
    def __init__(self, csv_path='sales_data.csv'):
        self.csv_path = csv_path
        self.data = None
        self.linear_model = None
        self.arima_model = None

    def load_and_preprocess_data(self):
        """
        Load data from CSV and perform preprocessing:
          - Parse dates and sort data.
          - Handle missing values.
          - Create new date-based features.
        """
        try:
            self.data = pd.read_csv(self.csv_path, parse_dates=['Date'])
            logging.info(f"Data loaded from {self.csv_path}")
        except FileNotFoundError:
            logging.warning("CSV file not found. Creating synthetic dataset...")
            dates = pd.date_range(start="2022-01-01", periods=150, freq='W')
            revenue = np.linspace(2000, 10000, num=150) + np.random.normal(0, 500, 150)
            self.data = pd.DataFrame({"Date": dates, "Revenue": revenue})
        
        # Sort and reset index
        self.data.sort_values('Date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # Handle missing values if any
        self.data['Revenue'].fillna(method='ffill', inplace=True)

        # Feature engineering: extract date features
        self.data['Date_ordinal'] = self.data['Date'].map(pd.Timestamp.toordinal)
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Weekday'] = self.data['Date'].dt.weekday

        logging.info("Data preprocessing complete. Data preview:")
        logging.info(self.data.head())
    
    def train_linear_regression(self):
        """
        Train a multivariate linear regression model using date features.
        Uses Date_ordinal, Month, and Weekday as features.
        """
        features = ['Date_ordinal', 'Month', 'Weekday']
        X = self.data[features].values
        y = self.data['Revenue'].values

        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        lr = LinearRegression()

        cv_scores = cross_val_score(lr, X, y, cv=tscv, scoring='neg_mean_squared_error')
        logging.info(f"Linear Regression CV MSE: {-np.mean(cv_scores):.2f} (avg over 5 folds)")

        # Fit model on entire dataset
        lr.fit(X, y)
        self.linear_model = lr
        self.data['LR_Predicted'] = lr.predict(X)

        # Evaluation metrics on training set
        mse = mean_squared_error(y, self.data['LR_Predicted'])
        mae = mean_absolute_error(y, self.data['LR_Predicted'])
        r2 = r2_score(y, self.data['LR_Predicted'])
        logging.info(f"Linear Regression Metrics: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
    
    def train_arima(self, order=(2,1,2)):
        """
        Train an ARIMA model on the revenue time series.
        The order can be adjusted based on analysis.
        """
        ts_data = self.data.set_index('Date')['Revenue']
        # Fit the ARIMA model
        self.arima_model = ARIMA(ts_data, order=order).fit()
        logging.info(f"ARIMA model trained with order {order}.")
        # Add in-sample predictions and residuals for evaluation
        self.data['ARIMA_Fitted'] = self.arima_model.fittedvalues
        residuals = ts_data - self.arima_model.fittedvalues
        mse = np.mean(np.square(residuals))
        logging.info(f"ARIMA Model In-sample MSE: {mse:.2f}")
    
    def tune_arima(self):
        """
        Tune ARIMA parameters using grid search.
        Searches over a range of p, d, q values to find the optimal model.
        """
        p = d = q = range(0, 3)
        best_aic = float('inf')
        best_order = None
        for param in product(p, d, q):
            try:
                model = ARIMA(self.data.set_index('Date')['Revenue'], order=param).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = param
            except:
                continue
        logging.info(f"Best ARIMA order found: {best_order} with AIC: {best_aic}")
        return best_order
    
    def visualize_models(self):
        """
        Visualize actual revenue vs. predictions from Linear Regression and ARIMA.
        Also plot residuals and forecast future revenue.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(self.data['Date'], self.data['Revenue'], label='Actual Revenue', marker='o', color='black')

        # Plot Linear Regression predictions
        plt.plot(self.data['Date'], self.data['LR_Predicted'], label='Linear Regression Prediction', color='blue', linewidth=2)

        # Plot ARIMA fitted values
        plt.plot(self.data['Date'], self.data['ARIMA_Fitted'], label='ARIMA Fitted', color='red', linestyle='--', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.title('Sales and Revenue Prediction: Actual vs. Model Predictions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Residual Analysis for Linear Regression
        plt.figure(figsize=(12, 4))
        residuals_lr = self.data['Revenue'] - self.data['LR_Predicted']
        sns.histplot(residuals_lr, kde=True, color='blue')
        plt.title('Residuals Distribution - Linear Regression')
        plt.xlabel('Residual')
        plt.show()

        # Residual Analysis for ARIMA
        plt.figure(figsize=(12, 4))
        ts_data = self.data.set_index('Date')['Revenue']
        residuals_arima = ts_data - self.arima_model.fittedvalues
        sns.histplot(residuals_arima, kde=True, color='red')
        plt.title('Residuals Distribution - ARIMA Model')
        plt.xlabel('Residual')
        plt.show()

    def forecast_future(self, days=60):
        """
        Forecast future revenue using the ARIMA model.
        Plots the forecast along with confidence intervals.
        """
        if self.arima_model is None:
            logging.error("ARIMA model is not trained yet.")
            return
        
        last_date = self.data['Date'].max()
        forecast = self.arima_model.get_forecast(steps=days)
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        plt.figure(figsize=(14, 6))
        plt.plot(self.data['Date'], self.data['Revenue'], label='Historical Revenue', color='black')
        plt.plot(forecast_index, forecast_mean, label='Forecasted Revenue (ARIMA)', color='green', linewidth=2)
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.3,
                         label='Confidence Interval')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.title('Future Revenue Forecast')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        future_date = forecast_index[-1]
        logging.info(f"Forecast complete. Revenue prediction for {future_date.date()}: {forecast_mean.iloc[-1]:.2f}")

    def run_all(self):
        """
        Run the full pipeline: data loading, training both models, visualization, and forecasting.
        """
        self.load_and_preprocess_data()
        self.train_linear_regression()

        best_arima_order = self.tune_arima()
        self.train_arima(order=best_arima_order)

        self.visualize_models()
        self.forecast_future(days=60)

if __name__ == "__main__":
    predictor = SalesRevenuePredictor(csv_path='sales_data.csv')
    predictor.run_all()
