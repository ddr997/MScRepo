import math

import numpy as np
import pandas as pd
import plotly.express as px
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


class ModelCreator:

    def __init__(self, dataFrame: DataFrame):
        self.dataFrame = dataFrame
        self.layers = None

    # Parse layers
    def parseLayers(self, text: str):
        self.layers = list(map(lambda l: l.split(" "),
                               text.strip().split("\n")
                               ))
        return self.layers

    def createLSTMPrediction(self, epochs: int, batchSize: int, timeWindow: int):

        df = self.dataFrame['Close'].values
        df = df.reshape(-1, 1)

        len = df.shape[0]
        splitIndex = int(len * 0.8)
        dataset_train = np.array(df[:splitIndex])
        dataset_test = np.array(df[splitIndex:])

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_train = scaler.fit_transform(dataset_train)
        dataset_test = scaler.fit_transform(dataset_test)

        print(np.ndarray.tolist(dataset_train))
        def create_dataset(df, window):
            x = []
            y = []
            for i in range(window, df.shape[0]):
                x.append(df[i - window:i, 0])
                y.append(df[i, 0])
            x = np.array(x)
            y = np.array(y)
            return x, y

        x_train, y_train = create_dataset(dataset_train, timeWindow)
        x_test, y_test = create_dataset(dataset_test, timeWindow)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=2, batch_size=32)
        predictions = model.predict(x_test)

        # results
        predictions = scaler.inverse_transform(predictions)
        # actual
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        dates = self.dataFrame.index[splitIndex:]
        dfPredictions = pd.DataFrame()
        # print(dates)
        predictions = np.ndarray.tolist(predictions)
        dfPredictions['Date'] = dates
        dfPredictions['Predictions'] = predictions
        dfActual = pd.DataFrame(dates, y_test_scaled)
        print(dfActual)
        # fig = px.line(
        #     dfPredictions,
        #     x="Date",
        #     y="Prediction",
        #     markers=True,
        #     title=f"closing price line chart")

        return 1

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
