import time
start_time = time.time() 

import os
import sys
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import function as tf_function
tf.config.run_functions_eagerly(True)

import yfinance as yf
from pytickersymbols import PyTickerSymbols

# Get tickers
INDEX = 'S&P 500' # choose from 'NASDAQ 100' 'S&P 500'

def get_tickers(index):
    stock_data = PyTickerSymbols()
    symbols = stock_data.get_stocks_by_index(index)
    symbols = pd.DataFrame(symbols)
    symbols = symbols[['symbol', 'name']]
    symbols = symbols.sort_values(by='symbol').reset_index(drop=True)
    symbols.to_csv('symbols.csv', index=False)
    tickers = symbols['symbol'].tolist()
    return tickers

tickers = get_tickers(INDEX)
print(f"Number of tickers in {INDEX}: {len(tickers)}")

# Prepare the dataset for training
data_dir = 'data'
REQUIRED_COMMON_FEATURES = 2
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
TIME_WINDOW = 3
TEST_SIZE = 0.2
SEED = 42

os.makedirs(data_dir, exist_ok=True)

def read_financial_statement(ticker, statement_type):
    filename = f"{ticker}_{statement_type}.csv"
    filepath = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(filepath, index_col=0)

        try:
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        except:
            print(f"The index of the {ticker} {statement_type} table could not be converted to a date format. Attempting to infer the format automatically.")
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(axis=0)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

financial_data = {}
for ticker in tickers:
    balance_sheet = read_financial_statement(ticker, 'balance_sheet')
    income_stmt = read_financial_statement(ticker, 'income_stmt')
    cash_flow = read_financial_statement(ticker, 'cash_flow')

    if balance_sheet is not None and cash_flow is not None and income_stmt is not None:
        financial_data[ticker] = {
            'balance_sheet': balance_sheet,
            'income_stmt': income_stmt,
            'cash_flow': cash_flow
        }
        print(f"Successfully read all financial statements of {ticker}.")
    else:
        missing_statements = []
        if balance_sheet is None:
            missing_statements.append('balance_sheet')
        if cash_flow is None:
            missing_statements.append('cash_flow')
        if income_stmt is None:
            missing_statements.append('fincome_stmt')
        print(f"Skip {ticker} (missing financial statements: {', '.join(missing_statements)})")

def preprocess_financial_data(financial_data, ticker):
    df_bs = financial_data[ticker]['balance_sheet']
    df_is = financial_data[ticker]['income_stmt']
    df_cf = financial_data[ticker]['cash_flow']

    df = df_bs.join(df_is, how='outer', lsuffix='_bs', rsuffix='_is')
    df = df.join(df_cf, how='outer', rsuffix='_cf')

    df = df.sort_index()

    df = df.fillna(method='ffill').fillna(0)

    return df

processed_data = {}
for ticker in financial_data:
    try:
        processed_data[ticker] = preprocess_financial_data(financial_data, ticker)
        print(f"Successfully preprocessed the data for {ticker}.")
    except Exception as e:
        print(f"Error occurred while preprocessing the data for {ticker}: {e}.")

###################### IMPORTANT **********************
features = [
    # 'Cash Cash Equivalents And Short Term Investments',
    # 'Accounts Receivable',
    # 'Inventory',
    # 'Other Current Assets',
    # 'Net PPE',
    # 'Accounts Payable',
    # 'Current Deferred Revenue',
    # 'Current Debt And Capital Lease Obligation',
    # 'Long Term Debt And Capital Lease Obligation',
    # 'Retained Earnings',
    # 'Net Income',
    # 'Repurchase Of Capital Stock'
    # 'Total Revenue',
    # 'Cost Of Revenue',
    # 'Gross Profit',
    # 'Operating Expense',
    # 'Depreciation',
    # 'Operating Cash Flow',
    'Net Income',
    'Operating Cash Flow'

]


targets = [
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
]
###################### IMPORTANT **********************

valid_tickers = []
missing_features = {}
missing_targets = {}

for ticker in processed_data:
    df = processed_data[ticker]
    missing_f = [feature for feature in features if feature not in df.columns]
    missing_t = [target for target in targets if target not in df.columns]
    if not missing_f and not missing_t:
        valid_tickers.append(ticker)
    else:
        missing_features[ticker] = missing_f
        missing_targets[ticker] = missing_t
        print(f"{ticker} is missing necessary features or targets, skip.")

print(f"Valid tickers: {valid_tickers}")

for ticker in missing_features:
    print(f"{ticker} missing features: {missing_features[ticker]}")
for ticker in missing_targets:
    print(f"{ticker} missing targets: {missing_targets[ticker]}")

common_features = set(features)
common_targets = set(targets)

for ticker in valid_tickers:
    df = processed_data[ticker]
    common_features &= set(df.columns)
    common_targets &= set(df.columns)

common_features = list(common_features)
common_targets = list(common_targets)

print(f"Number of common features: {len(common_features)}")
print(f"Number of common targets: {len(common_targets)}")

if len(common_features) >= REQUIRED_COMMON_FEATURES:
    final_features = common_features
else:
    raise ValueError(f"Fewer than {REQUIRED_COMMON_FEATURES} common features found, please check the data.")

print(f"Final selected features: {final_features}")

final_valid_tickers = []
for ticker in valid_tickers:
    df = processed_data[ticker]
    available_features = [feature for feature in final_features if feature in df.columns]
    
    # Check if there are enough features
    if len(available_features) < REQUIRED_COMMON_FEATURES:
        print(f"{ticker} has insufficient features after final selection, removed.")
        continue
    
    # Check if there is enough data within the specified date range considering the time window
    date_range = pd.date_range(start=START_DATE, end=END_DATE)
    df = df[df.index.isin(date_range)]
    if len(df) < TIME_WINDOW + 1:
        print(f"{ticker} does not have enough data within the date range {START_DATE} to {END_DATE}, removed.")
        continue
    
    final_valid_tickers.append(ticker)

print(f"Number of valid companies: {len(final_valid_tickers)}")

def create_dataset(df, features, targets, time_window):
    X, y = [], []
    for i in range(len(df) - time_window):
        # Include both features and lagged target values in X
        X.append(df[features + targets].iloc[i:i + time_window].values)
        y.append(df[targets].iloc[i + time_window].values)
    return np.array(X), np.array(y)

combined_X = []
combined_y = []

for ticker in final_valid_tickers:
    df = processed_data[ticker]
    date_range = pd.date_range(start=START_DATE, end=END_DATE)
    df = df[df.index.isin(date_range)]
    X, y = create_dataset(df, final_features, targets, TIME_WINDOW)
    combined_X.append(X)
    combined_y.append(y)
    print(f"Dimesions of the dataset for {ticker}: X={X.shape}, y={y.shape}")

if combined_X:
    X = np.vstack(combined_X)
    y = np.vstack(combined_y)
    print(f"Shape of the combined dataset: X={X.shape}, y={y.shape}")
else:
    print("No valid company data available for training.")
    X, y = np.array([]), np.array([])

scaler_X = StandardScaler()
scaler_y = StandardScaler()

if X.size > 0 and y.size > 0:
    X_shape = X.shape
    X = X.reshape(-1, X_shape[-1])
    X_scaled = scaler_X.fit_transform(X)
    X_scaled = X_scaled.reshape(X_shape)

    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, random_state=SEED
    )

    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
else:
    print("The dataset is empty, unable to perform scaling and splitting.")


# Train the LSTM model
MODEL_NAME = 'LSTM'
LAMBDA = 8.0
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 1
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 32


@tf.function

def custom_loss(y_true, y_pred, lambda_constraint=LAMBDA):
    mse = MeanSquaredError()
    mse_value = mse(y_true, y_pred)

    total_assets = y_pred[:, 0]
    total_liabilities = y_pred[:, 1]
    total_equity = y_pred[:, 2]

    accounting_loss = tf.square(total_assets - (total_liabilities + total_equity))

    total_loss = mse_value + lambda_constraint * accounting_loss
    return total_loss

def build_financial_forecast_model(input_shape, output_dim):
    model = models.Sequential()
    model.add(layers.LSTM(LSTM_UNITS_1, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(LSTM_UNITS_2, activation='relu'))
    model.add(layers.Dense(DENSE_UNITS, activation='relu'))
    model.add(layers.Dense(output_dim))

    model.compile(optimizer='adam', loss=custom_loss)
    return model


if X.size > 0 and y.size > 0:
    input_shape = (TIME_WINDOW, len(final_features) + len(targets))
    output_dim = len(targets)

    model = build_financial_forecast_model(input_shape, output_dim)
    model.summary()
else:
    print("The dataset is empty, unable to build the model.")

###################### IMPORTANT **********************
if X.size > 0 and y.size > 0:
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=VERBOSE
    )
else:
    print("The dataset is empty, unable to build the model.")
###################### IMPORTANT **********************

if X.size > 0 and y.size > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save the figure to a file in the current directory
    loss_curve_path = os.path.join(os.getcwd(), f'Loss_curve_{MODEL_NAME}.png')
    plt.savefig(loss_curve_path)
    print(f"Loss curve is plotted successfully and saved to '{loss_curve_path}'.")
else:
    print("No training history data available for plotting.")



# Random Walk Model
def random_walk_model(X):
    return X[:, -1, :]



# Evaluate model performance
def evaluate_model_performance(y_true, y_pred, scaler_y):
    # Inverse transform (denormalize)
    y_true_inv = scaler_y.inverse_transform(y_true)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Calculate MSE, MAE, RMSE, and R-squared
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_inv, y_pred_inv)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")

    return mse, mae, rmse, r2

if X.size > 0 and y.size > 0:
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse, mae, rmse, r2 = evaluate_model_performance(y_test, y_pred, scaler_y)
else:
    print("The dataset is empty, unable to perform evaluation.")

# Adjust predictions to satisfy the accounting equation
def adjust_predictions(y_pred):
    """
    Adjust predictions to ensure Total Assets = Total Liabilities + Shareholders' Equity
    Assume the first three columns of y_pred are Total Assets, Total Liabilities Net Minority Interest, and Total Equity Gross Minority Interest
    """
    adjusted_y_pred = y_pred.copy()
    total_liabilities = y_pred[:, 1]
    total_equity = y_pred[:, 2]

    # Recalculate Total Assets
    adjusted_y_pred[:, 0] = total_liabilities + total_equity

    return adjusted_y_pred

if X.size > 0 and y.size > 0:
    # Adjust predictions
    y_pred_adjusted = adjust_predictions(y_pred)

    # Re-evaluate the updated predictions
    mse_updated, mae_updated, rmse_updated, r2_updated = evaluate_model_performance(y_test, y_pred_adjusted, scaler_y)
    print(f"Updated MSE: {mse_updated}")
    print(f"Updated MAE: {mae_updated}")
    print(f"Updated RMSE: {rmse_updated}")
    print(f"Updated R-squared: {r2_updated}")

else:
    print("The dataset is empty, unable to make adjustments.")

# Evaluate the satisfaction of the accounting equation
def evaluate_accounting_constraint(y_true, y_pred_adjusted, scaler_y):
    """
    Evaluate the satisfaction of the accounting equation
    """
    # Inverse transform (denormalize)
    # y_true_inv = scaler_y.inverse_transform(y_true)
    y_pred_inv = scaler_y.inverse_transform(y_pred_adjusted)

    # Calculate the difference between assets and liabilities + equity
    asset_diff = y_pred_inv[:, 0] - (y_pred_inv[:, 1] + y_pred_inv[:, 2])
    mse_constraint = np.mean(np.square(asset_diff))
    mae_constraint = np.mean(np.abs(asset_diff))

    print(f"MSE of the accounting constraint: {mse_constraint}")
    print(f"MAE of the accounting constraint: {mae_constraint}")

    return mse_constraint, mae_constraint

if X.size > 0 and y.size > 0:
    # Evaluate the accounting equation
    mse_constraint, mae_constraint = evaluate_accounting_constraint(y_test, y_pred_adjusted, scaler_y)
else:
    print("The dataset is empty, unable to evaluate the accounting equation.")

if X.size > 0 and y.size > 0:
    # Create a DataFrame for the metrics
    metrics_data = {
        'Metric': ['MSE', 'MAE', 'RMSE', 'R-squared', 'MSE Constraint', 'MAE Constraint'],
        'LSTM': [mse_updated, mae_updated, rmse_updated, r2_updated, mse_constraint, mae_constraint]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Export the DataFrame to a CSV file
    csv_path = os.path.join(os.getcwd(), f'Evaluation_{MODEL_NAME}.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics table is exported successfully to '{csv_path}'.")
else:
    print("The dataset is empty, unable to evaluate the accounting equation.")


### make the table for comparison (not finished)

# Forecast and save results

def forecast_financials(model, scaler_X, scaler_y, df, features, targets, time_window):
    """
    Use the trained model to predict new financial data
    """
    # Construct input data
    X_new, _ = create_dataset(df, features, targets, time_window)
    if len(X_new) == 0:
        print("Insufficient data, unable to make predictions.")
        return None
    X_new_scaled = scaler_X.transform(X_new.reshape(-1, X_new.shape[-1])).reshape(X_new.shape)

    # Predict
    y_new_pred_scaled = model.predict(X_new_scaled)
    # y_new_pred = scaler_y.inverse_transform(y_new_pred_scaled)

    # Adjust predictions
    y_new_pred_adjusted = adjust_predictions(y_new_pred_scaled)
    y_new_pred_adjusted_inv = scaler_y.inverse_transform(y_new_pred_adjusted)

    return y_new_pred_adjusted_inv

forecast_results = {}
for ticker in final_valid_tickers:
    df = processed_data[ticker]
    # Only keep the data within the defined date range
    df = df.loc[START_DATE:END_DATE]
    forecasted_financials = forecast_financials(model, scaler_X, scaler_y, df, final_features, targets,
                                                time_window=TIME_WINDOW)
    if forecasted_financials is not None:
        # Get the latest forecasted results
        forecast_results[ticker] = forecasted_financials[-1]
        print(f"Forecasted Financials for {ticker}:\n{forecasted_financials[-1]}\n")
    else:
        print(f"Unable to predict financial data for {ticker}.\n")

# Save the results as a DataFrame
forecast_df = pd.DataFrame(forecast_results).T
forecast_df.columns = targets
forecast_df.to_csv('forecasted_financials.csv')
print("Forecast results have been saved to 'forecasted_financials.csv'.")




end_time = time.time()
total_time = end_time - start_time
print(f"The program has run successfully. Total time taken: {total_time:.2f} seconds.")