import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVR
from Ticker import Ticker
from models.DataProcessing import DataProcessing as dp
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
class SVRmodel:


    parameters = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.00001, 0.01, 0.1, 1, 10],
        "epsilon": [0.1, 0.01, 0.001]
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
        self.indexForWholeData = None

        #Forecast day values
        self.predictionDayInput = None
        self.predictionDayActualValue = None
        self.predictionForTheDay = None

        #Whole dataset prediction
        self.wholeDatasetPrediction = None

        #metrics
        self.RMSE = None
        self.MAPE = None

    def prepareModel(self, kernel="rbf", C=10, epsilon=0.01):
        self.currentModel = SVR(kernel=kernel, C=C, epsilon=epsilon)
        return self.currentModel

    def estimateHyperparameters(self, X, Y):
        model = SVR()
        tscv = TimeSeriesSplit(n_splits=5).split(X)
        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=SVRmodel.parameters, scoring=mean_squared_error)
        gsearch.fit(X,Y)
        print(gsearch.best_params_)

    def createPrediction(self, df: DataFrame, daysToShift: int = 0):
        # daysToShift - 0 means we will use most recent day from data k to predict k+1 (next day in the future, data we dont know) If we input 1, we will use k-1 to predict k, and we can compare with actual k that we have. So, if we have 1, we will have k-1 as a most recent target in target column, and after training we will use k-1 data to predict k

        # generate data with shift, by default training happens on k-1, and it predicts k, and then we use all k data to predict k+1
        self.trainingData, self.targetData = dp.generateXY_withDaysShift(df, daysToShift)

        # split the data for training and testing
        # x_train, x_test, y_train, y_test = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.8)
        self.trainX, self.testX, self.trainY, self.testY = dp.splitData_Dataframe(self.trainingData, self.targetData, trainRatio=0.8)

        # normalize training data after split (data leaking if all data normalisation), where each k has target function k+1
        # data has to be as arrays
        X = dp.meanNormalizeDataframe(self.trainX)
        Y = self.trainY.values

        # feed model with train data
        self.currentModel.fit(X, Y)

        #see the best parameters
        self.estimateHyperparameters(X, Y)

        #fetch real index for test data (actual vs predicted on certain day)
        # if daysToShift != 0:
        #     self.indexForTestData = df.index[1 + len(self.trainX.values) : len(df.values) - daysToShift + 1] #dla 0 wyjebie blad
        # newDate = df.index[-1] + pd.DateOffset(days=1)
        self.indexForTestData = self.testX.index[1:]
        ndate = pd.DatetimeIndex([self.testX.index[-1] + pd.tseries.offsets.BDay()])
        self.indexForTestData = self.indexForTestData.append(ndate)

        # test model with test data
        testX = dp.meanNormalizeDataframe(self.testX)
        testPredictedValues_array = self.currentModel.predict(testX)
        self.testPredictedValues = dp.convertTo_Dataframe(testPredictedValues_array, index=self.indexForTestData, columnLabels=["Test Predicted"])

        # evaluate metrics
        self.RMSE = dp.calculate_rmse(self.testY.values, self.testPredictedValues.values)
        self.MAPE = dp.calculate_mape(self.testY.values, self.testPredictedValues.values)
        print(self.RMSE, self.MAPE)

        # # generate input for prediction day, it means k-shift, so we add actual k data to predict k+1
        # predictionDayIndex = len(df.values) - daysToShift
        # self.predictionDayInput = df.drop("Close", axis=1).iloc[[predictionDayIndex]]
        # predictionDayInput_scaled = dp.meanNormalizeDataframe(self.predictionDayInput, resetScaler=False)
        # self.predictionForTheDay = self.currentModel.predict(predictionDayInput_scaled)

        # generated expected prices dataframe
        predictionIndex = self.trainY.index

        # # generate whole prediction to see how model performs overall
        # # prediction is k+1, so if we want to compare with actual price (what actually happened), we need to shift this data index forward so it matches the day it was desired to be predicted
        # wholeDatasetNormalized = dp.meanNormalizeDataframe(self.trainingData, resetScaler=False)
        # wholeDatasetPrediction_array = self.currentModel.predict(wholeDatasetNormalized)
        # self.indexForWholeData = df.index[1:len(df.values) - daysToShift]
        # self.wholeDatasetPrediction = dp.convertTo_Dataframe(
        #     wholeDatasetPrediction_array,
        #     index = self.indexForWholeData,
        #     columnLabels=["Close Predicted over whole Dataset"]
        # )

        # if we try to predict past day, we can see the actual value for that day (shift != 0, if equal we predict next day from today)
        # if daysToShift != 0:
        #     self.predictionDayActualValue = df["Close"].iloc[-daysToShift]
        #     #add forecast to whole dataset
        #     predictionRow = pd.DataFrame(self.predictionForTheDay, index = [df.index[len(df.values)-daysToShift]], columns=self.wholeDatasetPrediction.columns)
        #     self.wholeDatasetPrediction = self.wholeDatasetPrediction.append(predictionRow)

        #add forecast to whole dataset
        # self.wholeDatasetPrediction.append([self.predictionForTheDay[0].item()], ignore_index=False)

        plotDataframe = pd.concat([df['Close'], self.testPredictedValues], axis=1)
        dp.plotBasicComparisonGraph(plotDataframe)

        return self.wholeDatasetPrediction


if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(500).iloc[:,0:5]
    svr = SVRmodel()
    svr.prepareModel()
    predicted = svr.createPrediction(df, 1)
    print(predicted)
    print(svr.RMSE)
    print(svr.MAPE)