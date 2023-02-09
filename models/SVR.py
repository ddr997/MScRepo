import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVR
from Ticker import Ticker
from models.DataProcessing import DataProcessing as dp
import plotly.express as px
class SVRmodel:
    def __init__(self):
        self.currentModel = None

        self.trainingData = None
        self.targetData = None

        self.testingDayInput = None
        self.testingDayTarget = None

        self.predictionForTheDay = None

    def prepareModel(self, kernel="rbf"):
        self.currentModel = SVR(kernel=kernel)
        return self.currentModel

    def createPrediction(self, df: DataFrame, daysToShift: int = 0):
        # daysToShift - 0 means we will use most recent day k to predict k+1 (next day in the future, data we dont know)
        # if we input 1, we will use k-1 to predict k, and we can compare with actual k that we have
        # so if we have 1, we will have k-1 as a most recent target, and after training we will use k-1 all data to predict k

        #generate data with shift, by default training happens on k-1, and it predicts k, and then we use all k data to predict k+1
        self.trainingData, self.targetData = dp.generateXY_withShift(df, daysToShift)
        #normalize training data, where each k has target function k+1
        self.trainingData_scaled = dp.meanNormalizeDataframe(self.trainingData)
        X = self.trainingData_scaled
        Y = self.targetData.values
        self.prepareModel()
        self.currentModel.fit(X, Y)

        #generate input for test day, it means k-shift, so we add actual k data to predict k+1
        self.testingDayInput = df.drop("Close", axis=1).iloc[[len(df.values) - daysToShift - 1]]
        self.testingDayInput_scaled = dp.meanNormalizeDataframe(self.testingDayInput, resetScaler=False)
        self.predictionForTheDay = self.currentModel.predict(self.testingDayInput_scaled)

        #generate whole prediction to see how model performs (equivalent of test data, I will have to confirm that)
        # prediction is k+1, so if we want to compare with actual price (what actually happened), we need to shift this
        # data forward so it matches the day it was desired to be predicted
        self.outputForWholeTrainingPeriod = self.currentModel.predict(X)
        self.outputForWholeTrainingPeriod = dp.convertTo_dataframe(
            self.outputForWholeTrainingPeriod,
            index = df.index[1:len(df.values)-daysToShift],
            columnLabels=["Close Predicted"]
        )
        #if we try to predict past day, we can see the actual value for that day (shift != 0, if equal we predict next day from today)
        if daysToShift != 0:
            self.testingDayTarget = df["Close"].iloc[-daysToShift]

        p = pd.concat([df['Close'], self.outputForWholeTrainingPeriod], axis=1)

        dp.plotBasicComparisonGraph(p)

        return self.outputForWholeTrainingPeriod

if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(365).iloc[:,0:5]
    svr = SVRmodel()
    predicted = svr.createPrediction(df, 1)
    print(predicted)

