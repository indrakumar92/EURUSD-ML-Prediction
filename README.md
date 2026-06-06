# EURUSD Machine Learning Prediction

## Project Overview

This project uses Machine Learning to predict EURUSD price direction using 1-hour historical forex data from 2020 to 2024.

## Dataset

* Instrument: EURUSD
* Timeframe: 1 Hour
* Period: 2020–2024
* Records: 29,000+

## Features Used

* Return
* RSI
* Tick Volume
* Volatility
* Moving Averages (10,20,50)
* Momentum
* Hour of Day
* Range Percentage

## Model

Random Forest Classifier

Parameters:

* n_estimators = 300
* max_depth = 12
* min_samples_split = 10

## Results

Accuracy: 52.06%

Win Rate: 51.69%

Total Trades: 5800

Total Return: 0.02536

## Top Features

1. Hour of Day
2. Return
3. Tick Volume
4. Momentum
5. RSI

## Technologies

* Python
* Pandas
* Scikit-Learn
* Matplotlib

## Future Improvements

* XGBoost
* ATR Features
* Walk-Forward Testing
* Power BI Dashboard
