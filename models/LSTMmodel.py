import numpy as np
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten
from keras.models import Sequential
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVR
from Ticker import Ticker
from models.DataProcessing import DataProcessing as dp
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import keras.layers
from sklearn.preprocessing import StandardScaler

class LSTMmodel:

    def __init__(self):
        self.currentModel = None

        # remembering dataFrames
        self.trainingData = None
        self.targetData = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.testPredictedValues = None

        # index for dataframes (forecast requires shifting index to compare actual vs predicted)
        self.indexForTestData = None

        # metrics
        self.RMSE = None
        self.MAPE = None

        # fig
        self.fig = None


    # prepare model bedzie musial wylapac parametry w srodku createprediction
    def prepareModel(self):
        return

    def createPrediction(self, df: DataFrame, daysToShift: int = 0):
        # generate data with shift, by default training happens on k-1, and it predicts k, and then we use all k data to predict k+1
        self.trainingData, self.targetData = dp.generateXY_withDaysShift(df, daysToShift)

        # split the data for training and testing
        # x_train, x_test, y_train, y_test = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.8)
        self.trainX, self.testX, self.trainY, self.testY = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.75)

        # normalize training data after split (data leaking if all data normalisation), where each k has target function k+1
        # data has to be as arrays
        X = dp.meanNormalizeDataframe(self.trainX)
        Y = self.trainY.values

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM need 3D array

        # feed model with train data
        model = Sequential()
        model.add(
            LSTM(units=8, input_shape=(X.shape[1], 1), activation="sigmoid")
        )
        model.add(Dense(units=1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, Y, epochs=1000, batch_size=64)


        # append new date for future DF output
        self.indexForTestData = self.testX.index[1:]
        ndate = pd.DatetimeIndex([self.testX.index[-1] + pd.tseries.offsets.BDay()])
        self.indexForTestData = self.indexForTestData.append(ndate)


        # test model with test data
        testX = dp.meanNormalizeDataframe(self.testX)
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # LSTM need 3D array
        testPredictedValues_array = model.predict(testX)
        # dp.scaler.inverse_transform(testPredictedValues_array)
        self.testPredictedValues = dp.convertTo_Dataframe(testPredictedValues_array, index=self.indexForTestData, columnLabels=["LSTM predicted C(k)"])



        # evaluate metrics
        self.RMSE = dp.calculate_rmse(self.testY.values, self.testPredictedValues.values[:-1])
        self.MAPE = dp.calculate_mape(self.testY.values, self.testPredictedValues.values[:-1])
        print(self.RMSE, self.MAPE)

        predictionDataFrame = pd.concat([df['Close'], self.testPredictedValues], axis=1)
        # dp.plotBasicComparisonGraph(predictionDataFrame).show()
        return predictionDataFrame



if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(500).iloc[:,0:5]
    lstm = LSTMmodel()
    lstm.prepareModel()
    predicted = lstm.createPrediction(df, 1)
    print(predicted)
    print(lstm.RMSE)
    print(lstm.MAPE)