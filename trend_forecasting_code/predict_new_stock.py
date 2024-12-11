import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, ReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import get_lists as getter

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://www.kaggle.com/code/ritesh7355/develop-lstm-models-for-time-series-forecasting

# (56)
tickers = ["pg", "ko", "pep", "msft", "mcd", "hd", "unh", "wmt", "nee", "amgn", "wm", "cl", "low", "amt", "adp", "syk", "cat", "duk",
           "rok", "jci", "psa", "ecl", "cb", 'csco', 'intc', 'ibm', 'orcl', 'txn', 'dis',
           'jnj', 'pfe', 'mrk', 'abt', 'mdt', 'aapl', 'nvda', 'cvx', "gis", "mdlz", "lmt", "afl", "spg",
           "ed", "xel", "ato", "lnt", "sre", "aep", "mmc", "pgr",
           "stt", "pnc", "ntrs", "hsy", "avy", "cmi"]
companies_data = {ticker: getter.main(ticker) for ticker in tickers}

# creates df
def create_company_df(company_data):
    date_range = pd.date_range(start="2002-01-02", end="2024-10-04", freq="D")
    feature_dfs = {}
    for feature, data in company_data.items():
        df = pd.DataFrame(data, columns=["date", feature])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        feature_dfs[feature] = df
    company_df = pd.concat(feature_dfs.values(), axis=1)
    company_df = company_df.reindex(date_range)
    company_df.fillna(method='ffill', inplace=True)
    company_df.fillna(method='bfill', inplace=True)
    company_df.interpolate(method='linear', inplace=True)
    return company_df

company_dfs = {ticker: create_company_df(data) for ticker, data in companies_data.items()}

# combining
combined_train_data = pd.concat(company_dfs.values(), axis=0, ignore_index=True)

# scaling
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(combined_train_data)

# sequencing and targeting for closing price
sequence_length = 90
X_train, y_train = [], []
for i in range(sequence_length, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-sequence_length:i])
    y_train.append(scaled_train_data[i, 3])

X_train, y_train = np.array(X_train), np.array(y_train)

# model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),
    LSTM(units=50, return_sequences=True),
    Dropout(0.4),
    LSTM(units=30),
    Dense(units=1),
    ReLU()
])

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# for early-age stock to predict
def prepare_new_company_data(new_company_data):
    new_company_df = create_company_df(new_company_data)
    new_company_scaled = scaler.transform(new_company_df)

    X_new = []
    for i in range(sequence_length, len(new_company_scaled)):
        X_new.append(new_company_scaled[i-sequence_length:i])
    X_new = np.array(X_new)

    return X_new, new_company_scaled

# actually use model here
def predict_new_company_trend(new_company_ticker):

    new_company_data = getter.main(new_company_ticker)
    X_new, scaled_data = prepare_new_company_data(new_company_data)

    predictions_90 = model.predict(X_new[-90:])
    predictions_180 = []

    future_sequence = scaled_data[-sequence_length:]
    for _ in range(180):
        future_sequence = future_sequence.reshape((1, sequence_length, scaled_data.shape[1]))
        next_pred = model.predict(future_sequence)[0][0]
        predictions_180.append(next_pred)
        next_sequence = np.concatenate([future_sequence[0, 1:, :], np.array([[next_pred] + [0] * (scaled_data.shape[1] - 1)])], axis=0)
        future_sequence = next_sequence

    predictions_90_unscaled = scaler.inverse_transform(np.concatenate([predictions_90, np.zeros((90, scaled_data.shape[1] - 1))], axis=1))[:, 0]
    predictions_180_unscaled = scaler.inverse_transform(np.concatenate([np.array(predictions_180).reshape(-1, 1), np.zeros((180, scaled_data.shape[1] - 1))], axis=1))[:, 0]

    return predictions_90_unscaled, predictions_180_unscaled

new_company_ticker = "pins"
predictions_90, predictions_180 = predict_new_company_trend(new_company_ticker)

plt.figure(figsize=(12, 6))
plt.plot(range(90), predictions_90, label='90-day Predictions')
plt.xlabel('Days')
plt.ylabel('Predicted Closing Price')
plt.legend()
plt.title(f'90-Day Trend Predictions for {new_company_ticker}')
plt.savefig(f"{new_company_ticker}_90_days_predictions.png")

# 180 days is too far, doesn't work
plt.figure(figsize=(12, 6))
plt.plot(range(180), predictions_180, label='180-day Predictions')
plt.xlabel('Days')
plt.ylabel('Predicted Closing Price')
plt.legend()
plt.title(f'180-Day Trend Predictions for {new_company_ticker}')
plt.savefig(f"{new_company_ticker}_180_days_predictions.png")

print("Predictions complete.")
