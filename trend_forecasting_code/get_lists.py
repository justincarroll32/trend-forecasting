import json

def get_aroon_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["Technical Analysis: AROON"]
  for day, value in data.items():
    gg = (day, (float(value["Aroon Up"]) - float(value["Aroon Down"])))
    final.append(gg)

  return final

def get_bbands_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["Technical Analysis: BBANDS"]
  for day, value in data.items():
    b_bandwidth = (float(value["Real Upper Band"]) - float(value["Real Lower Band"])) / float(value["Real Middle Band"])
    gg = (day, b_bandwidth)
    final.append(gg)

  return final

def get_ema_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["Technical Analysis: EMA"]
  for day, value in data.items():
    gg = (day, float(value["EMA"]))
    final.append(gg)

  return final

def get_closing_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["Time Series (Daily)"]
  for day, value in data.items():
    gg = (day, float(value["4. close"]))
    final.append(gg)

  return final

def get_gdp_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["data"]
  for day, value in data.items():
    gg = (day, float(value))
    final.append(gg)

  return final

def get_treasury_yield_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["data"]
  for day, value in data.items():
    gg = (day, float(value))
    final.append(gg)

  return final

def get_unemployment_list(filename: str) -> list:
  data = []
  final = []
  with open(f"all-data/{filename}", 'r') as file:
    data = json.load(file)

  data = data["data"]
  for day, value in data.items():
    gg = (day, float(value))
    final.append(gg)

  return final

def main(stock: str):
  # print(len(get_aroon_list("dis-aroon.json")))
  # print(len(get_bbands_list("dis-bbands.json")))
  # print(len(get_ema_list("dis-ema.json")))
  # print(len(get_closing_list("dis-ochlv.json")))
  # print(len(get_gdp_list("gdp-us-daily.json")))
  # print(len(get_treasury_yield_list("treasury-yield-daily.json")))
  # print(len(get_unemployment_list("unemployment-daily.json")))

  aroon = get_aroon_list(f"{stock}-aroon.json")
  bbands = get_bbands_list(f"{stock}-bbands.json")
  ema = get_ema_list(f"{stock}-ema.json")
  closing = get_closing_list(f"{stock}-ochlv.json")
  gdp = get_gdp_list("gdp-us-daily.json")
  treasury = get_treasury_yield_list("treasury-yield-daily.json")
  unemployment = get_unemployment_list("unemployment-daily.json")

  data = {
    "aroon": aroon,
    "bbands": bbands,
    "ema": ema,
    "closing_price": closing,
    "gdp": gdp,
    "unemployment": unemployment,
    "treasury_yield": treasury  
}

  return data

if __name__ == "__main__":
  main()

