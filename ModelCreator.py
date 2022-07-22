import math

import numpy as np
import pandas as pd
import plotly.express as px
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class ModelCreator:

    def __init__(self, dataFrame: DataFrame):
        self.dataFrame = dataFrame
        self.layers = list(list())

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
        dataset_test = np.array(df[splitIndex - timeWindow:])

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_train = scaler.fit_transform(dataset_train)
        dataset_test = scaler.transform(dataset_test)

        # For the features (x), we will always append the last 50 prices, and for the label (y), we will append the next price.
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

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # LSTM need 3D array
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        LSTMlayers = [layer[0] for layer in self.layers].count("LSTM")
        LSTMcounter = 0
        for index, layer in enumerate(self.layers):
            num = layer[1]
            if index == 0:
                model.add(
                    LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1))
                )
                LSTMcounter += 1
                continue
            if layer[0] == "LSTM":
                if LSTMlayers-LSTMcounter == 1:
                    model.add(
                        LSTM(units=int(num))
                    )
                else:
                    model.add(
                        LSTM(units=int(num), return_sequences=True)
                    )
                    LSTMcounter += 1
            elif layer[0] == "Dense":
                model.add(
                    Dense(units=int(num))
                )
            elif layer[0] == "Dropout":
                model.add(
                    Dropout(float(num))
                )
            else:
                Exception("No such layer in neural network API!")
        # model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=96, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=96))
        # model.add(Dropout(0.2))
        # model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        with st.spinner('Creating a prediction...'):
            model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize)
        st.write("Success!")
        predictions = model.predict(x_test)
        print(predictions)

        # results
        predictions = scaler.inverse_transform(predictions)
        # actual
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        dates = self.dataFrame.index[splitIndex:]

        dfPredictions = pd.DataFrame()
        dfPredictions['Date'] = dates
        dfPredictions['Predictions'] = predictions
        dfPredictions['Actual'] = y_test_scaled
        print(dfPredictions)
        fig1 = px.line(
            dfPredictions,
            x="Date",
            y="Predictions",
            markers=True,
            title=f"Predicted price")
        fig1.update_traces(line_color='#FF8C0A', line_width=2, marker_size=1)

        fig2 = px.line(
            dfPredictions,
            x="Date",
            y="Actual",
            markers=True,
            title=f"Actual price")
        fig2.update_traces(line_color='#05F097', line_width=2, marker_size=1)
        ret = go.Figure(data=fig1.data + fig2.data)
        ret.update_layout(
            title=f"Stock price prediction in selected time window of {timeWindow} days:",
            showlegend=True
        )
        rmse = self.calculate_rmse(y_test_scaled, predictions)
        mape = self.calculate_mape(y_test_scaled, predictions)
        return ret, rmse, mape

    def createLinearRegressionPrediction(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataFrame.index, self.dataFrame["Close"], test_size = 0.33, random_state = 42
        )

        model = LinearRegression()
        model.fit(x_train, y_train)

        prediction = model.predict(x_test)
        print(y_test, prediction)


    #### Calculate the metrics RMSE and MAPE ####
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def calculate_mape(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape
