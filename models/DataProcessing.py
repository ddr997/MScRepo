import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from Ticker import Ticker


class DataProcessing():

    # def meanNormalizeDataframe(df: DataFrame, columnsToOmit: list = []):
    #     df = df.copy(deep=True).drop(columnsToOmit, axis=1)
    #     normalized_df = (df - df.mean()) / df.std()
    #     return normalized_df
    scaler = StandardScaler()

    def meanNormalizeDataframe(df: DataFrame, columnsToOmit: list = [], resetScaler = True):
        df = df.copy(deep=True).drop(columnsToOmit, axis=1)
        if resetScaler:
            DataProcessing.scaler.fit(df.values)
        normalized_array = DataProcessing.scaler.transform(df.values)
        normalized_dataframe = pd.DataFrame(normalized_array, index = df.index, columns = df.columns)
        return normalized_array

    # def minMaxNormalizeDataframe(df: DataFrame, columnsToOmit: list = []):
    #     df = df.copy(deep=True).drop(columnsToOmit, axis=1)
    #     normalized_df = (df - df.min()) / (df.max() - df.min())
    #     return normalized_df

    def splitData(x: np.ndarray, y: np.ndarray, trainProportion: float = 0.8):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=trainProportion, shuffle=False)
        return x_train, x_test, y_train, y_test

    def convertTo_ndarray(df: DataFrame):
        return df.values

    # def convertTo_dataframe(array: np.ndarray, index: pd.DatetimeIndex, columnLabels:list = []):
    #     return pd.DataFrame(array, columns = columnLabels, index = index)

    def convertTo_dataframe(*arrays: np.ndarray, index: pd.DatetimeIndex, columnLabels:list = []):
        df = pd.DataFrame(index = index)
        iterator = iter(columnLabels)
        for i in arrays:
            if iterator.__length_hint__() > 0:
                df[next(iterator)] = i
            else:
                df[len(df.columns)] = i
        return df

    def generateXY_withShift(df: DataFrame, daysShift:int):
        target = df['Close'].shift(-1).dropna()
        target = target.iloc[:len(target.values)-daysShift] # k in k-1 row
        df_copy = df.copy(deep=True)
        input = df_copy.drop('Close', axis=1).iloc[:-1-daysShift] #k-1
        return input, target


    def extendDataFrameWithLookBacksColumn(df: DataFrame, daysToLookBack: int):
        for k in range(1,daysToLookBack+1):
            colIndex = "Close_k-"+str(k)
            df[colIndex] = df['Close'].shift(k)
        return df

    def plotBasicComparisonGraph(df: DataFrame):
        fig = px.line(df, markers=True, title="Close price actual vs predicted")
        fig.update_layout(yaxis=dict(title="Close price"))
        fig.show()
        return


if __name__ == "__main__":
    ticker = Ticker("ALE.WA")
    df = ticker.getData(365)
    print(df)
    # print(DataProcessing.minMaxNormalizeDataframe(df, ["Volume"]))
    print(DataProcessing.meanNormalizeDataframe(df))

    # a = df['Close'].values
    # b = df['Open'].values
    # output = DataProcessing.convertTo_dataframe(a,b, index=df.index, columnLabels = ["L1"])
    # print(output)
    # # DataProcessing.plotBasicComparisonGraph(output)
    #
    # lookbacks = DataProcessing.extendDataFrameWithLookBacksColumn(df, 5)
    # print(lookbacks)