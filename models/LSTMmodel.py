import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import pandas as pd
from pandas import DataFrame
from tensorflow.python.keras.activations import sigmoid

from Ticker import Ticker
from models.DataProcessing import DataProcessing as dp
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
from scikeras.wrappers import KerasRegressor
class LSTMmodel:

    parameters = dict(
        units=randint(14,50),
        epochs=randint(14,42),
        batch_size=[4,8,16,32,64],
    )

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
        self.MSE = None
        self.MAPE = None
        self.MDA = None
        self.metricsDict = {}

        # fig
        self.fig = None

        #in this model it needs to be like this
        self.hp = dict(
                units=14,
                epochs=42,
                batch_size=32,
                activation="sigmoid",
                recurrent_activation="sigmoid"
            )


    # prepare model bedzie musial wylapac parametry w srodku createprediction
    def prepareModel(self, **kwargs):
        self.hp = kwargs
        return

    def estimateHyperparameters(self, model, X, Y):
        tscv = TimeSeriesSplit(n_splits=5).split(X)
        nmodel = KerasRegressor(model=model, activation=sigmoid, recurrent_activation=sigmoid, units=46)
        gsearch = RandomizedSearchCV(estimator=nmodel, cv=tscv, param_distributions=LSTMmodel.parameters, n_iter=1000, scoring="neg_mean_squared_error")
        gsearch.fit(X,Y)
        self.bestHyperparameters = gsearch.best_params_
        print(self.bestHyperparameters)

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
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM need 3D array
        Y = self.trainY.values


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

        #estiamte hyperparameters
        # self.estimateHyperparameters(model, X, Y)


        # append new date for future DF output
        self.indexForTestData = self.testX.index[1:]
        ndate = pd.DatetimeIndex([self.testX.index[-1] + pd.tseries.offsets.BDay()])
        self.indexForTestData = self.indexForTestData.append(ndate)


        # test model with test data
        testX = dp.meanNormalizeDataframe(self.testX)
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # LSTM need 3D array
        testPredictedValues_array = model.predict(testX)
        self.testPredictedValues = dp.convertTo_Dataframe(testPredictedValues_array, index=self.indexForTestData, columnLabels=["LSTM predicted C(k)"])


        predictionDataFrame = pd.concat([realClose, self.testPredictedValues], axis=1)
        #convert output to Close prices
        for i in range(0,len(predictionDataFrame.values)):
            if predictionDataFrame.iloc[i,1] != np.nan:
                predictionDataFrame.iloc[i,1] += predictionDataFrame.iloc[i-1, 0]


        # evaluate metrics
        self.MSE = dp.calculate_mse(self.testY.values, self.testPredictedValues.values[:-1])
        self.MAPE = dp.calculate_mape(self.testY.values, self.testPredictedValues.values[:-1])
        self.MDA = dp.calculate_mda(predictionDataFrame)
        print(f"MSE:{self.MSE}\n" +
              f"MAPE:{self.MAPE}\n" +
              f"MDA:{self.MDA}")
        self.metricsDict = {"LSTM": [self.MSE, self.MAPE, self.MDA]}


        # dp.plotBasicComparisonGraph(predictionDataFrame).show()
        return predictionDataFrame



if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(500).iloc[:,0:5]
    lstm = LSTMmodel()
    predicted = lstm.createPrediction(df, 1)
    print(predicted)