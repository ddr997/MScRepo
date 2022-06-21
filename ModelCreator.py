import math
import numpy as np
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class ModelCreator:

    def __init__(self, dataFrame: DataFrame):
        self.dataFrame = dataFrame


    def createLSTMPrediction(self):

        scaled_data = self.dataFrame["Close"].values.reshape(-1, 1)

        time_step = 20 # window required for LSTM training
        x_train = []
        y_train = []

        # for each day we create target function of X days
        for x in range(time_step, len(scaled_data)):
            x_train.append(scaled_data[x - time_step:x, 0]) # od 0 do maxa
            y_train.append(scaled_data[x, 0])# od timestep do maxa (1 dzien w przyszlosc)
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))

        model.summary()
        model.compile(optimizer='adam',
                      loss='mean_squared_error')

        model.fit(x_train,
                  y_train,
                  epochs=25,
                  batch_size=32)

        test_data = scaled_data[len(self.dataFrame["Close"]) - time_step:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        # 2.  Convert the values into arrays for easier computation
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # 3. Making predictions on the testing data
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        return y_train

