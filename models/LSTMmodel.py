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

        #in this model it needs to be like this
        self.hp = dict(
                units=20,
                epochs=100,
                batch_size=32,
                activation="sigmoid",
                recurrent_activation="sigmoid"
            )


    # prepare model bedzie musial wylapac parametry w srodku createprediction
    def prepareModel(self, **kwargs):
        self.hp = kwargs
        return

    def createPrediction(self, df: DataFrame, daysToShift: int = 0):
        # save real close prices
        realClose = df["Close"]
        # create diffrentiation
        df['Close'] = df['Close'].diff()
        # generate data with shift, by default training happens on k-1, and it predicts k, and then we use all k data to predict k+1
        self.trainingData, self.targetData = dp.generateXY_withDaysShift(df, daysToShift)

        # split the data for training and testing
        self.trainX, self.testX, self.trainY, self.testY = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.75)

        # normalize training data after split (data leaking if all data normalisation), where each k has target function k+1
        # data has to be as arrays
        X = dp.meanNormalizeDataframe(self.trainX)
        Y = self.trainY.values

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM need 3D array

        # feed model with train data
        model = Sequential()
        model.add(
            LSTM(
                units=self.hp.get("units"),
                input_shape=(X.shape[1], 1),
                activation=self.hp.get("activation"),
                recurrent_activation=self.hp.get("recurrent_activation"))
        )
        model.add(Dense(units=1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, Y, epochs=self.hp.get("epochs"), batch_size=self.hp.get("batch_size"))


        # append new date for future DF output
        self.indexForTestData = self.testX.index[1:]
        ndate = pd.DatetimeIndex([self.testX.index[-1] + pd.tseries.offsets.BDay()])
        self.indexForTestData = self.indexForTestData.append(ndate)


        # test model with test data
        testX = dp.meanNormalizeDataframe(self.testX)
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # LSTM need 3D array
        testPredictedValues_array = model.predict(testX)
        self.testPredictedValues = dp.convertTo_Dataframe(testPredictedValues_array, index=self.indexForTestData, columnLabels=["LSTM predicted C(k)"])

        # evaluate metrics
        self.RMSE = dp.calculate_mse(self.testY.values, self.testPredictedValues.values[:-1])
        self.MAPE = dp.calculate_mape(self.testY.values, self.testPredictedValues.values[:-1])
        print(self.RMSE, self.MAPE)

        predictionDataFrame = pd.concat([realClose, self.testPredictedValues], axis=1)

        #convert output to Close prices
        for i in range(0,len(predictionDataFrame.values)):
            if predictionDataFrame.iloc[i,1] != np.nan:
                predictionDataFrame.iloc[i,1] += predictionDataFrame.iloc[i-1, 0]

        # dp.plotBasicComparisonGraph(predictionDataFrame).show()
        return predictionDataFrame



if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(500).iloc[:,0:5]
    lstm = LSTMmodel()
    predicted = lstm.createPrediction(df, 1)
    print(predicted)
    print(lstm.RMSE)
    print(lstm.MAPE)