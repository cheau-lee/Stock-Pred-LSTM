# Stock Prediction Using LSTM
Click the badge to run this project in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cheau-lee/Stock-Pred-LSTM/HEAD)
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements and References](#acknowledgements-and-references)

## Introduction

Welcome to the Stock Price Prediction project! This repository showcases an advanced approach to forecasting Dow Jones Industrial Average (DJIA) stock prices utilising Long Short-Term Memory (LSTM) neural networks. Predicting stock prices is an inherently complex and challenging task, primarily due to the volatility and intricate nature of financial markets. This project leverages on LSTM networks, a specialised form of recurrent neural networks (RNN), to forecast future stock prices based on historical data. LSTM models are particularly well-suited for time series prediction because they can effectively capture and learn from long-term dependencies and patterns within the data, making them a powerful tool for financial forecasting.


## Features
- Data loading and preprocessing
- Min-Max scaling for normalisation
- Sequence creation for time series prediction
- LSTM model building and training
- Prediction on test data
- Visualisation of results
- Future price prediction
  
## Libraries

- **NumPy**: For numerical operations and efficient array handling.
- **Pandas**: For data manipulation and analysis.
- **OS Module**: For file handling.
- **Matplotlib**: For basic plotting and visualisation.
- **Plotly**: For interactive and advanced visualisations.
- **Scikit-Learn**: For data scaling using `MinMaxScaler`.
- **TensorFlow and Keras**: For building and training the LSTM model with `Sequential`, `LSTM`, `Dense`, and `Dropout` layers.
## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/cheau-lee/Stock-Pred-LSTM.git
    ```
2. Change the directory:
    ```bash
    cd Stock-Pred-LSTM
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the `upload_DJIA_table.csv` file in the project directory.
2. Run the main script to start the training and prediction pipeline:
    ```bash
    python stock_prediction.ipynb
    ```

## Model Architecture

The LSTM model consists of:
- Three LSTM layers with 50 units each
- Dropout layers with a dropout rate of 20% to prevent overfitting
- A dense layer to produce the final output

The model is compiled using the Adam optimiser and mean squared error (MSE) loss function.

## Results

The model's performance is evaluated by comparing the predicted stock prices against the actual prices from the test set. The results are visualised using Matplotlib, showing both training and validation losses over epochs.


## Future Work

Future improvements to this project could include:
- Experimenting with different model architectures and hyperparameters
- Using additional features (e.g., trading volume, technical indicators) for prediction
- Implementing real-time prediction with live stock data
- Exploring more advanced techniques like attention mechanisms

## Acknowledgements and References 

I would like to express a special thanks to Sun, J. for providing the invaluable dataset on Kaggle. This has been instrumental in enhancing my data preprocessing, deep learning, and data visualisation skills!

- Sun, J. (2016) 'Daily News for Stock Market Prediction, Version 1', Kaggle. Available at: https://www.kaggle.com/aaron7sun/stocknews (Accessed: 26 May 2024).
