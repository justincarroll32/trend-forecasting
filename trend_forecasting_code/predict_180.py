import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import get_lists as getter

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://www.kaggle.com/code/ritesh7355/develop-lstm-models-for-time-series-forecasting

# 56 total stocks
tickers = ["pg", "ko", "pep", "msft", "mcd", "hd", "unh", "wmt", "nee", "amgn", "wm", "cl", "mdt", "low", "amt", "adp", "syk", "cat", "duk",
               "rok", "jci", "psa", "ecl", "cb", 'csco', 'intc', 'ibm', 'orcl', 'txn', 'dis',
               'jnj', 'pfe', 'mrk', 'abt', 'aapl', 'nvda', 'cvx', "gis", "mdlz", "lmt", "afl", "spg",
               "ed", "xel", "ato", "lnt", "sre", "aep", "mmc", "pgr",
               "stt", "pnc", "ntrs", "hsy", "avy", "cmi"]

companies_data = {ticker: getter.main(ticker) for ticker in tickers}

# create the company dataframe
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

train_companies = [df for ticker, df in company_dfs.items() if ticker != 'txn']
combined_train_data = pd.concat(train_companies, axis=0, ignore_index=True)

# scalers
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(combined_train_data)

sequence_length = 60
X_train, y_train = [], []
for i in range(sequence_length, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-sequence_length:i])
    y_train.append(scaled_train_data[i, 3])

X_train, y_train = np.array(X_train), np.array(y_train)

# model configuration
model = Sequential([
    LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=30),
    Dropout(0.2),
    Dense(units=1)
])

# more configuration and where model is actually called
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# plot training loss statistics
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("training_validation_loss_txn_run.png")

txn_data = scaler.transform(company_dfs['txn'])

# last 60 days of data
X_test_txn, y_test_txn = [], []
for i in range(len(txn_data) - 30 - sequence_length, len(txn_data) - 30):
    X_test_txn.append(txn_data[i-sequence_length:i])
    y_test_txn.append(txn_data[i, 3])

# prepare txn data to be predicted (part of LOOCV)
X_test_txn, y_test_txn = np.array(X_test_txn), np.array(y_test_txn)

txn_predictions = model.predict(X_test_txn)

padding_txn_predictions = np.concatenate([txn_predictions, np.zeros((txn_predictions.shape[0], txn_data.shape[1] - 1))], axis=1)
padding_y_test_txn = np.concatenate([y_test_txn.reshape(-1, 1), np.zeros((y_test_txn.shape[0], txn_data.shape[1] - 1))], axis=1)

txn_predictions_unscaled = scaler.inverse_transform(padding_txn_predictions)[:, 0]
y_test_txn_unscaled = scaler.inverse_transform(padding_y_test_txn)[:, 0]

# plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_txn_unscaled, label='Actual txn')
plt.plot(txn_predictions_unscaled, label='Predicted txn')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.title('Actual vs Predicted Closing Prices for txn')
plt.savefig("txn_last_60_days_actual_vs_predicted.png")
