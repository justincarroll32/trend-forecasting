import json
from datetime import datetime

# writes aroon data to json file and filters it to start at jan 2 2002
def process_json_file(filename):
    date_threshold = "2002-01-01"

    with open(filename, 'r') as file:
        data = json.load(file)

    filtered_time_series = {
        date: info for date, info in data["Technical Analysis: AROON"].items()
        if date >= date_threshold
    }

    transformed_data = {
        "name": data["Meta Data"]["1: Symbol"],
        "Technical Analysis: AROON": filtered_time_series
    }

    with open(filename, 'w') as file:
        json.dump(transformed_data, file, indent=4)

    print(f"Processed and saved filtered data to {filename}")


def main(tickers: list):
    # insert tickers to process here
    tickers = []
    tickers = [(ticker + "-aroon.json") for ticker in tickers]

    keep = []
    correct_date_status = []
    for ticker in tickers:
      process_json_file(ticker)
      data = []
      with open(ticker, "r") as file:
          data = json.load(file)
      data = data["Technical Analysis: AROON"]
      data = list(data.keys())[-1]
      print(f'Last date for {ticker}:', data)
      keep.append(ticker)
      if data != "2002-01-02":
          correct_date_status.append((ticker, data))

    print("Number of files process: ", len(keep))

    if correct_date_status == []:
      print("All files have correct date.")
    else:
      for file in correct_date_status:
        print(f'Files that are not in correct date:')
        print(f'{file[0]} | {file[1]}')

if __name__ == "__main__":
    main()
