import time
start_time = time.time()

import os
import sys
import pandas as pd
pd.set_option('display.max_columns', None)

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

# Financial data
if not os.path.exists('data'):
    os.makedirs('data')

def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet.T
        income_stmt = stock.income_stmt.T
        cash_flow = stock.cashflow.T
        return balance_sheet, income_stmt, cash_flow
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None

financial_data = {}
for ticker in tickers:
    balance_sheet, income_stmt, cash_flow = get_financial_data(ticker)
    if balance_sheet is not None:
        financial_data[ticker] = {
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }

        income_stmt.to_csv(f'data/{ticker}_income_stmt.csv')
        balance_sheet.to_csv(f'data/{ticker}_balance_sheet.csv')
        cash_flow.to_csv(f'data/{ticker}_cash_flow.csv')
        print(f"Successfully fetched and saved data for {ticker}")
    else:
        print(f"Failed to fetch data for {ticker}")


end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"The program has run successfully in {int(minutes)} minutes and {seconds:.2f} seconds.")