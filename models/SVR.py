import random

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVR
from Ticker import Ticker
from models.DataProcessing import DataProcessing as dp
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
class SVRmodel:


    # parameters = {
    #     "kernel": ["linear", "rbf", "poly"],
    #     "C": [1e-08, 1e-06, 1e-2, 0.1, 1, 10],
    #     "epsilon": [1, 0.1, 1e-2, 1e-3, 1e-4]
    # }
    parameters = {
        "kernel": ["linear", "rbf", "sigmoid"],
        "C": uniform(0.1, 9.9),
        "epsilon": uniform(1e-3, 0.1)
    }

    def __init__(self):
        self.currentModel = None

        #remembering dataFrames
        self.trainingData = None
        self.targetData = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY= None
        self.testPredictedValues = None

        #index for dataframes (forecast requires shifting index to compare actual vs predicted)
        self.indexForTestData = None

        #metrics
        self.MSE = None
        self.MAPE = None
        self.MDA = None
        self.metricsDict = {}

        #fig
        self.fig = None

        #computed best hyperparameters
        self.bestHyperparameters = None

    def prepareModel(self, kernel="rbf", C=10, epsilon=0.01):
        self.currentModel = SVR(kernel=kernel, C=C, epsilon=epsilon)
        return self.currentModel

    def estimateHyperparameters(self, X, Y):
        model = SVR()
        tscv = TimeSeriesSplit(n_splits=5).split(X)
        # gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=SVRmodel.parameters, scoring="neg_mean_squared_error")
        gsearch = RandomizedSearchCV(estimator=model, cv=tscv, param_distributions=SVRmodel.parameters, n_iter=1000)
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
        # x_train, x_test, y_train, y_test = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.8)
        self.trainX, self.testX, self.trainY, self.testY = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.75)


        # normalize training data after split (data leaking if all data normalisation), where each k has target function k+1
        # data has to be as arrays
        X = dp.meanNormalizeDataframe(self.trainX)
        Y = self.trainY.values


        # feed model with train data
        self.currentModel.fit(X, Y)


        #see the best parameters
        # self.estimateHyperparameters(X, Y)


        # append new date
        self.indexForTestData = self.testX.index[1:]
        ndate = pd.DatetimeIndex([self.testX.index[-1] + pd.tseries.offsets.BDay()])
        self.indexForTestData = self.indexForTestData.append(ndate)


        # test model with test data
        testX = dp.meanNormalizeDataframe(self.testX)
        testPredictedValues_array = self.currentModel.predict(testX)
        self.testPredictedValues = dp.convertTo_Dataframe(testPredictedValues_array, index=self.indexForTestData, columnLabels=["SVR predicted C(k)"])


        # concatenate prediction and real
        predictionDataFrame = pd.concat([realClose, self.testPredictedValues], axis=1)
        # predictionDataFrame = pd.concat([df['Close'], self.testPredictedValues], axis=1)
        #convert output to Close prices
        for i in range(0,len(predictionDataFrame.values)):
            if predictionDataFrame.iloc[i,1] != np.nan:
                predictionDataFrame.iloc[i,1] += predictionDataFrame.iloc[i-1, 0]


        self.MSE = dp.calculate_mse(self.testY.values, self.testPredictedValues.values[:-1])
        self.MAPE = dp.calculate_mape(self.testY.values, self.testPredictedValues.values[:-1])
        self.MDA = dp.calculate_mda(predictionDataFrame)
        print(f"MSE:{self.MSE}\n" +
              f"MAPE:{self.MAPE}\n" +
              f"MDA:{self.MDA}")
        self.metricsDict = {"SVR": [self.MSE, self.MAPE, self.MDA]}

        # dp.plotBasicComparisonGraph(predictionDataFrame).show()
        return predictionDataFrame


if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(500).iloc[:,0:5]
    svr = SVRmodel()
    svr.prepareModel()
    predicted = svr.createPrediction(df, 1)
