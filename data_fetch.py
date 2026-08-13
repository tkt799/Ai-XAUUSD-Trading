import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Ticker candidates in priority order:
#   GC=F   - COMEX gold futures (closest to XAUUSD)
#   GLD    - SPDR Gold Shares ETF (fallback proxy, tracks spot gold)
#   IAU    - iShares Gold Trust ETF (second fallback proxy)
TICKER_CANDIDATES = ['GC=F', 'GLD', 'IAU']


def _normalize_download(data, ticker):
    """Normalize a yfinance download to a clean OHLCV frame.

    Handles the MultiIndex columns returned by newer yfinance versions
    (>= 0.2.4x / 1.x) as well as the classic flat layout.
    """
    if data is None or data.empty:
        return None

    # Newer yfinance: columns are (field, ticker) MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=-1)
        except (KeyError, TypeError):
            data.columns = data.columns.get_level_values(0)

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(how='all')
    if data.empty:
        return None

    data = data.reset_index()
    # Index column is 'Date' (daily) or 'Datetime' (intraday), case may vary
    first_col = data.columns[0]
    data = data.rename(columns={first_col: 'date'})
    return data


def fetch_xauusd_data(start_date, end_date, tickers=None):
    """
    Fetch gold daily data via yfinance with a ticker fallback chain.

    Returns a clean OHLCV DataFrame (columns: date/Open/High/Low/Close/Volume),
    or None if every candidate fails.
    """
    for ticker in (tickers or TICKER_CANDIDATES):
        try:
            raw = yf.download(ticker, start=start_date, end=end_date,
                              interval='1d', auto_adjust=False, progress=False)
            data = _normalize_download(raw, ticker)
            if data is not None and len(data) > 0:
                print(f"Data source: {ticker}  rows: {len(data)}")
                return data
        except Exception as e:
            print(f"{ticker}: download failed ({e})")

    print("No data fetched from any ticker candidate")
    return None


if __name__ == "__main__":
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    data = fetch_xauusd_data(start_date, end_date)
    if data is not None:
        data.to_csv('xauusd_data.csv', index=False)
        print("Data saved to xauusd_data.csv")
        print(f"Data shape: {data.shape}")
    else:
        print("Failed to fetch data")
