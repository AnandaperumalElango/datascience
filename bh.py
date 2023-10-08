import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import yfinance as yf
#import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

stock_symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2021-12-31'
data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data['Close']
# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(np.array(data).reshape(-1, 1))
# Create training and testing datasets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create sequences of data for training
def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

seq_length = 10  # Adjust this parameter as needed
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predicted_prices = model.predict(X_test)

# Inverse transform the predictions to get actual stock prices
predicted_prices = scaler.inverse_transform(predicted_prices)

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Prices', color='blue')
plt.plot(np.arange(train_size+seq_length, len(data)), predicted_prices, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.legend()
plt.show()