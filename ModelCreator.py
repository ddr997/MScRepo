import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class ModelCreator:

    def __init__(self, dataFrame: DataFrame):
        self.dataFrame = dataFrame

    # Split the time-series data into training seq X and output value Y
    def extract_seqX_outcomeY(self, data, N, offset):
        """
        Split time-series into training sequence X and outcome value Y
        Args:
            data - dataset
            N - window size, e.g., 50 for 50 days of historical stock prices
            offset - position to start the split
        """
        X, y = [], []

        for i in range(offset, len(data)):
            X.append(data[i - N:i])
            y.append(data[i])

        return np.array(X), np.array(y)


    def createLSTMPrediction(self, timestep):
        timestep = int(timestep)

        close_prices = self.dataFrame["Close"]
        values = close_prices.values
        training_data_len = math.ceil(len(values) * 0.8)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values.reshape(-1, 1))
        train_data = scaled_data[0: training_data_len, :]

        x_train = []
        y_train = []

        for i in range(timestep, len(train_data)):
            x_train.append(train_data[i - timestep:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        scaled_data = self.dataFrame["Close"].values.reshape(-1, 1)

        test_data = scaled_data[training_data_len - timestep:, :]
        x_test = []
        y_test = values[training_data_len:]

        for i in range(timestep, len(test_data)):
            x_test.append(test_data[i - timestep:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=3)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        rmse

        data = self.dataFrame.filter(['Close'])
        train = data[:training_data_len]
        validation = data[training_data_len:]
        validation['Predictions'] = predictions

        fig = px.line(
            validation,
            x=validation.index,
            y=["Close", "Predictions"],
            markers=True,
            title=f"closing price line chart")

        return fig


    #### Calculate the metrics RMSE and MAPE ####
    def calculate_rmse(y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def calculate_mape(y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

