import time
start_time = time.time() 

import os
import sys
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import function as tf_function


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
TEST_SIZE = 0.20
SEED = 97

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
        # print(f"Successfully read all financial statements of {ticker}.")
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

    # Convert all values to billions of USD
    df = df / 1e9

    return df

processed_data = {}
for ticker in financial_data:
    try:
        processed_data[ticker] = preprocess_financial_data(financial_data, ticker)
        # print(f"Successfully preprocessed the data for {ticker}.")
    except Exception as e:
        print(f"Error occurred while preprocessing the data for {ticker}: {e}.")

###################### IMPORTANT **********************
features = [
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Working Capital',
    'Retained Earnings',
    'Net PPE',
    'Cash Cash Equivalents And Short Term Investments',
    'Inventory',
    'Accounts Receivable',
    'Operating Cash Flow',
    'Free Cash Flow',
    'Operating Cash Flow',
    'Capital Expenditure',
    'Changes In Cash',
    'Cash Flow From Continuing Operating Activities'
]


targets = [
    'Net Income',
    'Total Revenue',
    'Total Expenses',
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
    # print(f"Dimesions of the dataset for {ticker}: X={X.shape}, y={y.shape}")

if combined_X:
    X = np.vstack(combined_X)
    y = np.vstack(combined_y)
    print(f"Shape of the combined dataset: X={X.shape}, y={y.shape}")
else:
    print("No valid company data available for training.")
    X, y = np.array([]), np.array([])

# Print the mean for y
if y.size > 0:
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    print(f"Mean of y in billion USD: {y_mean}")
    print(f"Standard deviation of y in billion USD: {y_std}")
else:
    print("The dataset is empty, unable to calculate the mean for y.")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

if X.size > 0 and y.size > 0:
    # Split the data before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Fit the scaler on the training data and transform both training and test data
    X_train_shape = X_train.shape
    X_train = X_train.reshape(-1, X_train_shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_train_scaled = X_train_scaled.reshape(X_train_shape)
    
    X_test_shape = X_test.shape
    X_test = X_test.reshape(-1, X_test_shape[-1])
    X_test_scaled = scaler_X.fit_transform(X_test)
    X_test_scaled = X_test_scaled.reshape(X_test_shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.fit_transform(y_test)
    
    y_test_mean = scaler_y.mean_
    y_test_std = scaler_y.scale_
    print(f"Training set shape: X_train={X_train_scaled.shape}, y_train={y_train_scaled.shape}")
    print(f"Testing set shape: X_test={X_test_scaled.shape}, y_test={y_test_scaled.shape}")
else:
    print("The dataset is empty, unable to perform scaling and splitting.")


# Train the LSTM model
MODEL_NAME = 'LSTM'
LAMBDA = 8.0
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 1
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64


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
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Fit the model with early stopping
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stopping],
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
    loss_curve_path = os.path.join(os.getcwd(), f'Loss_curve_is_{MODEL_NAME}.png')
    plt.savefig(loss_curve_path)
    print(f"Loss curve is plotted successfully and saved to '{loss_curve_path}'.")
else:
    print("No training history data available for plotting.")

if X.size > 0 and y.size > 0:
    # Predict
    y_pred_lstm = model.predict(X_test_scaled)
    y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
else:
    print("The dataset is empty, unable to perform evaluation.")


###################### IMPORTANT **********************
# Adjust predictions to satisfy the accounting equation
def adjust_predictions(y_pred_level):
    """
    Adjust predictions to ensure Net Income = Total Revenue - Total Expenses
    Assume the first three columns of y_pred are Net Income, Total Revenue, and Total Expenses
    """
    
    # Adjust predictions
    adjusted_y_pred_level = y_pred_level.copy()
    total_revenue = y_pred_level[:, 1]
    total_expenses = y_pred_level[:, 2]

    # Recalculate Total Assets
    adjusted_y_pred_level[:, 0] = total_revenue - total_expenses

    return adjusted_y_pred_level

if X.size > 0 and y.size > 0:
    # Adjust predictions
    adjusted_y_pred_lstm_level = adjust_predictions(y_pred_lstm_inv)
else:
    print("The dataset is empty, unable to make adjustments.")
###################### IMPORTANT **********************


# Random Walk Model
def random_walk_model(X):
    return X[:, -1, -len(targets):]

y_pred_rw = random_walk_model(X_test_scaled)
y_pred_rw_inv = scaler_y.inverse_transform(y_pred_rw)


# Evaluate model performance
NUM_CONSTRAINTS = 1

def evaluate_model_performance(y_test, y_pred_level):
    # Initialize lists to store metrics for each dimension
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []

    # Calculate metrics for each dimension
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:, i], y_pred_level[:, i])
        rmse = root_mean_squared_error(y_test[:, i], y_pred_level[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred_level[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], y_pred_level[:, i])
        r2 = r2_score(y_test[:, i], y_pred_level[:, i])

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)

        print(f"Dimension {i+1} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R-squared: {r2}")

    # Aggregate metrics (mean of each metric across dimensions)
    mse_agg = np.mean(mse_list)
    rmse_agg= np.mean(rmse_list)
    mae_agg = np.mean(mae_list)
    mape_agg = np.mean(mape_list)
    r2_agg = np.mean(r2_list)

    print(f"Aggregated MSE: {mse_agg}")
    print(f"Aggregated RMSE: {rmse_agg}")
    print(f"Aggregated MAE: {mae_agg}")
    print(f"Aggregated MAPE: {mape_agg}")
    print(f"Aggregated R-squared: {r2_agg}")

    return mse_agg, rmse_agg, mae_agg, mape_agg, r2_agg


# Evaluate the satisfaction of the accounting equation
def evaluate_accounting_constraint(y_pred_level):
    """
    Evaluate the satisfaction of the accounting equation
    """

    # Calculate the difference between assets and liabilities + equity
    check = y_pred_level[:, 0] - (y_pred_level[:, 1] - y_pred_level[:, 2])
    mse_constraint = np.mean(np.square(check))
    mae_constraint = np.mean(np.abs(check))

    print(f"MSE of the accounting constraint: {mse_constraint}")
    print(f"MAE of the accounting constraint: {mae_constraint}")

    return mse_constraint, mae_constraint


# Evaluate LSTM model
if X.size > 0 and y.size > 0:
    # Evaluate the model performance
    mse_updated, rmse_updated, mae_updated, mape_updated, r2_updated = evaluate_model_performance(y_test, adjusted_y_pred_lstm_level)
    # Evaluate the accounting equation
    mse_constraint, mae_constraint = evaluate_accounting_constraint(adjusted_y_pred_lstm_level)
else:
    print("The dataset is empty, unable to evaluate the accounting equation.")

# Evaluate random walk model
if X.size > 0 and y.size > 0:
    # Evaluate the model performance
    mse_rw, rmse_rw, mae_rw, mape_rw, r2_rw = evaluate_model_performance(y_test, y_pred_rw_inv)
    # Evaluate the accounting equation
    mse_constraint_rw, mae_constraint_rw = evaluate_accounting_constraint(y_pred_rw_inv)
else:
    print("The dataset is empty, unable to evaluate the accounting equation.")


if X.size > 0 and y.size > 0:
    # Create a DataFrame for the metrics
    metrics_data = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared', 'MSE Constraint', 'MAE Constraint'],
        'LSTM': [mse_updated, rmse_updated, mae_updated, mape_updated, r2_updated, mse_constraint, mae_constraint],
        'Random Walk': [mse_rw, rmse_rw, mae_rw, mape_rw, r2_rw, mse_constraint_rw, mae_constraint_rw]
    }
    metrics_df = pd.DataFrame(metrics_data).round(2)

    # Export the DataFrame to a CSV file
    csv_path = os.path.join(os.getcwd(), f'Evaluation_is_all_models.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics table is exported successfully to '{csv_path}'.")
else:
    print("The dataset is empty, unable to evaluate the accounting equation.")



end_time = time.time()
total_time = end_time - start_time
print(f"The program has run successfully. Total time taken: {total_time:.2f} seconds.")