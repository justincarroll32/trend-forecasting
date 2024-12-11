import utilities as ut
import requests # type: ignore
import json
import log_config
import process_ochlv as p_ochlv
import process_aroon as aroon
import process_bbands as bbands
import process_ema as ema

logger = log_config.logging_configure("stock_historical_data.py")

def get_time_series_daily(key: str, symbol: str, write_to_json: bool, filename: str, output_file_dir: str) -> dict:
    """Returns all daily historical data for all OCHLV data for one company."""

    logger.debug(f"Accessing TIME-SERIES-DAILY API for: {symbol.upper()}")

    data = []
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={key}'
    r = requests.get(url)
    data = r.json()

    if write_to_json:
        logger.debug(f'Writing {symbol.upper()} TIME_SERIES_DAILY data to {output_file_dir}/{filename}... ')
        ut.write_to_json_file(filename, output_file_dir, data)
        logger.info(f"Done writing {symbol.upper()} TIME_SERIES_DAILY data to json file...")

    print("-" * 100)

def get_time_series_intraday(key: str, symbol: str, write_to_json: bool, filename: str, output_file_dir: str, minutes: int) -> dict:
    """Returns all data for intraday (i.e. every 1,5,15,30,60 min intervals) for one company."""

    logger.debug(f"Accessing TIME-SERIES-INTRADAY API for: {symbol.upper()}")

    data = []
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={minutes}min&apikey={key}'
    r = requests.get(url)
    data = r.json()

    if write_to_json:
        logger.debug(f'Writing {symbol.upper()} TIME_SERIES_INTRADAY data to {output_file_dir}/{filename}... ')
        ut.write_to_json_file(filename, output_file_dir, data)
        logger.info(f"Done writing {symbol.upper()} TIME_SERIES_INTRADAY data to json file...")

    print("-" * 100)

def get_gdp_quarterly(key: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns gdp of US."""
  logger.debug(f"Accessing REAL_GDP API")

  data = []
  url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing REAL_GDP data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing REAL_GDP data to json file...")

  print("-" * 100)

def get_treasury_yield(key: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns treasury yield for US."""
  logger.debug(f"Accessing TREASURY_YIELD API")

  data = []
  url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing TREASURY_YIELD data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing TREASURY_YIELD data to json file...")

  print("-" * 100)

def get_unemployment(key: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns unemployment rate for US."""
  logger.debug(f"Accessing UNEMPLOYMENT API")

  data = []
  url = f'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing UNEMPLOYMENT data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing UNEMPLOYMENT data to json file...")

  print("-" * 100)


def get_ema(key: str, symbol: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns EMA for one company."""
  logger.debug(f"Accessing EMA API for: {symbol.upper()}")

  data = []
  url = f'https://www.alphavantage.co/query?function=EMA&symbol={symbol}&interval=daily&time_period=50&series_type=open&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing {symbol.upper()} EMA data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing {symbol.upper()} EMA data to json file...")

  print("-" * 100)

def get_aroon(key: str, symbol: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns AROON for one company."""
  logger.debug(f"Accessing AROON API for: {symbol.upper()}")

  data = []
  url = f'https://www.alphavantage.co/query?function=AROON&symbol={symbol}&interval=daily&time_period=20&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing {symbol.upper()} AROON data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing {symbol.upper()} AROON data to json file...")

  print("-" * 100)

def get_bbands(key: str, symbol: str, write_to_json: bool, filename: str, output_file_dir: str):
  """Returns BBANDS for one company."""
  logger.debug(f"Accessing BBANDS API for: {symbol.upper()}")

  data = []
  url = f'https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval=daily&time_period=10&series_type=close&nbdevup=2&nbdevdn=2&apikey={key}'
  r = requests.get(url)
  data = r.json()

  if write_to_json:
      logger.debug(f'Writing {symbol.upper()} BBANDS data to {output_file_dir}/{filename}... ')
      ut.write_to_json_file(filename, output_file_dir, data)
      logger.info(f"Done writing {symbol.upper()} BBANDS data to json file...")

  print("-" * 100)

def main():
    # api key insert here
    api_key = ""

    # insert list of tickers to get data for
    tickers = ['pins']

    for ticker in tickers:
      output_file_dir = ""
      output_filename = f'{ticker}-ochlv.json'
      # output_file_dir = "technical-factors-data/OCHLV-data/"
      get_time_series_daily(api_key, ticker, True, output_filename, output_file_dir)

      output_filename = f'{ticker}-ema.json'
      # output_file_dir = "technical-factors-data/ema-data/"
      get_ema(api_key, ticker, True, output_filename, output_file_dir)

      output_filename = f'{ticker}-aroon.json'
      # output_file_dir = "technical-factors-data/aroon-data/"
      get_aroon(api_key, ticker, True, output_filename, output_file_dir)

      output_filename = f'{ticker}-bbands.json'
      # output_file_dir = "technical-factors-data/bbands-data/"
      get_bbands(api_key, ticker, True, output_filename, output_file_dir)


    get_gdp_quarterly(api_key, True, "gdp-us.json", "technical-factors-data/gdp/")
    get_treasury_yield(api_key, True, "treasury-yield-quarterly.json", "technical-factors-data/treasury-yield/")
    get_unemployment(api_key, True, "unemployment.json", "technical-factors-data/unemployment/")

    print("PROCESSING DATA NOW ************************************************************************")
    p_ochlv.main(tickers)
    aroon.main(tickers)
    bbands.main(tickers)
    ema.main(tickers)


if __name__ == '__main__':
    main()