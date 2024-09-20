import pandas as pd
import sqlite3
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    db_directory = 'DataBase'
    db_file = 'intraday_stock_prices.db'
    db_path = os.path.join(db_directory, db_file)
    os.makedirs(db_directory, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA datetime_format = 'YYYY-MM-DD HH:MM:SS'")
    return conn


def fetch_intraday_data(ticker,interval="1min"):
    # Please check https://www.alphavantage.co/documentation/ for the details on the data

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    API_key = "Demo"
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&apikey={API_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    time_series = data.get(f'Time Series ({interval})', {})
    if not time_series:
        print(f"No data found for {ticker}")
        return None
    df = pd.DataFrame(time_series).T
    df.index.name = "timestamp"
    colums_str = ["open", "high", "low", "close", "volume"]
    df.columns = colums_str
    df=df.sort_index()
    df.reset_index(inplace = True)
    for S in colums_str:
        df[S] = df[S].astype(float)
    df["ticker"] = ticker
    return df

def Get_latest_timestamp(ticker):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = '''
        SELECT MAX(timestamp) FROM intraday_prices WHERE ticker = ?
        '''
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()[0]
    return result

def store_data_to_db(data, latest_timestamp):
    if data is not None and not data.empty:
        if latest_timestamp:
            data = data[data['timestamp'] > latest_timestamp]
        if not data.empty:
            with get_db_connection() as conn:
                data.to_sql('intraday_prices', conn, if_exists='append', index=False)
            print(f'Data for {data["ticker"].iloc[0]} stored successfully.')
        else:
            print(f'No new data to store for {data["ticker"].iloc[0]}.')
    else:
        print(f'No data to store.')

def query_data(ticker, start_time, end_time):
    query = """
    SELECT * FROM intraday_prices
    WHERE ticker = ? AND timestamp BETWEEN ? AND ?
    ORDER BY timestamp ASC
    """
    with get_db_connection() as conn:
        df = pd.read_sql(query, conn, params=(ticker, start_time, end_time))
    return df

def main():
    with get_db_connection() as conn:
        conn.execute("PRAGMA datetime_format = 'YYYY-MM-DD HH:MM:SS'")
        conn.execute('''
        CREATE TABLE IF NOT EXISTS intraday_prices (
            ticker TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
        ''')
    top_us_stocks = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK-A", "LLY", "TSM", "AVGO",
    "TSLA", "JPM", "WMT", "SONY", "UNH", "V", "XOM", "NVO", "MA", "PG", "JNJ", "COST",
    "ORCL", "HD", "ASML", "ABBV", "BAC", "KO", "MRK", "NFLX", "AZN", "CVX", "SMFG",
    "SAP", "CRM", "ADBE", "TM", "NVS", "PEP", "AMD", "TMUS", "TMO", "LIN", "ACN",
    "MCD", "CSCO", "ABT", "WFC", "BABA", "INTC", "IBM", "QCOM", "NVS", "UPS", "HON",
    "AMGN", "SBUX", "PFE", "INTU", "FDX", "MDT", "CHTR", "TGT", "LMT", "AMT", "WBA",
    "TFC", "TXN", "BLK", "CVS", "NOW", "SBAC", "MU", "ISRG", "GM", "DHR", "TJX",
    "ADI", "PYPL", "ZTS", "DUK", "CAT", "LOW", "PLD", "MS", "EL", "FIS", "ADP",
    "SO", "BDX", "MDLZ", "FDX", "GS", "SCHW", "KEP", "NEE", "CL", "BKNG", "CSX",
    "ADI", "ICE", "MCO", "EA", "GD", "TFC", "TD", "AON", "WDAY", "APD", "EQIX",
    "KMB", "CCI", "ETN", "SYK", "SPGI", "HUM", "EMR", "INTC", "NSC", "EXC", "WM",
    "BMY", "FISV", "LRCX", "VZ", "ROST", "EW", "PGR", "STZ", "AEP", "KLAC", "BSX",
    "APTV", "MU", "AIG", "ORLY", "NSC", "AFL" ]
    
    tickers = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    for ticker in tickers:
        latest_timestamp = Get_latest_timestamp(ticker)
        data = fetch_intraday_data(ticker)
        store_data_to_db(data,latest_timestamp)

if __name__ == "__main__":
    main()