import yfinance as yf
import pandas as pd

def get_stock_data(tickers: list[str], start_date: str, end_date: str):
    # Fetch historical adjusting closing price (auto_adjust=True) for a list of tickers.
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data.dropna()