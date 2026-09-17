# Motor Vehicle Fuel Price Regression with Deep Learning

This is our Deep Learning course project about predicting fuel prices using time-series data.

## Project Overview

In this project, we collected data from 2003 to 2025 and used deep learning models to predict crude oil prices. The dataset includes oil prices and economic variables from EIA and FRED.

We tested five different models:

- RNN
- LSTM
- GRU
- LSTM + GRU Hybrid
- Attention-based model

## Data Preprocessing

Before training the models, we did some basic preprocessing:

- Filled missing values using linear interpolation
- Normalized the data using Min-Max Scaling
- Selected useful economic and financial features
- Used previous time steps to predict future oil prices

## Results

The models were able to learn the main trends in oil prices.

Some results from our experiments:

| Model | Result |
| --- | --- |
| LSTM | Test RMSE: 0.0643, Test MAE: 0.0541, Test R²: 0.8199 |
| GRU | MAE: 0.0180, RMSE: 0.0242, R²: 0.9819 |
| Attention-based Model | Test RMSE: 0.0233 |

The GRU model achieved very good results after feature selection and hyperparameter tuning. The attention-based model also followed the main price trends well.

The LSTM + GRU hybrid model worked well on the training data, but its predictions became less accurate in some later parts of the test data.

