from pandas import DataFrame
from sklearn.svm import SVR
from Ticker import Ticker
from models.DataProcessing import DataProcessing
import plotly.express as px
class SVRmodel:
    def __init__(self):
        self.currentModel = None
        self.trainingData = None
        self.targetData = None

        self.testInput = None
        self.testExpectedOutput = None

    def createPrediction(self, df: DataFrame, daysToPredict: int):
        self.targetData = df["Close"].shift(-daysToPredict).iloc[:-daysToPredict]
        self.trainingData = DataProcessing.meanNormalizeDataframe(df).iloc[:-daysToPredict]
        testingDay = df['Close'].iloc[-1]

        # X = df[['Open', 'Close', 'High', 'Low', 'Volume']].values
        X = self.trainingData.values
        Y = self.targetData.values
        svr_rbf = SVR(kernel='rbf')
        svr_rbf.fit(X, Y)

        testInput = self.trainingData.iloc[-1].values.reshape(1,-1)
        prediction = svr_rbf.predict(testInput)
        all = svr_rbf.predict(X)

        return prediction

if __name__ == '__main__':
    ticker = Ticker("ALE.WA")
    df = ticker.getData(365).iloc[:,0:5]
    svr = SVRmodel()
    predicted = svr.createPrediction(df, 1)
    print(predicted)

