import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from Ticker import Ticker


class DataProcessing():

    scaler = StandardScaler()

    def meanNormalizeDataframe(df: DataFrame, columnsToOmit: list = [], resetScaler = True) -> np.ndarray:
        df = df.copy(deep=True).drop(columnsToOmit, axis=1)
        if resetScaler:
            DataProcessing.scaler.fit(df.values)
        normalized_array = DataProcessing.scaler.transform(df.values)
        normalized_dataframe = pd.DataFrame(normalized_array, index = df.index, columns = df.columns)
        return normalized_array

    def meanNormalizeArray(array: np.ndarray, resetScaler = True)  -> np.ndarray:
        if resetScaler:
            DataProcessing.scaler.fit(df.values)
        normalized_array = DataProcessing.scaler.transform(df.values)
        return normalized_array

    def splitData_array(x: np.ndarray, y: np.ndarray, trainRatio: float = 0.8):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=trainRatio, shuffle=False)
        return x_train, x_test, y_train, y_test

    def splitData_Dataframe(trainDF: DataFrame, testDF: DataFrame, trainRatio: float = 0.8):
        l = len(trainDF.values)
        splitIndex = round(l*trainRatio)
        x_train = trainDF.iloc[0:splitIndex]
        x_test = trainDF.iloc[splitIndex:]

        y_train = testDF.iloc[0:splitIndex]
        y_test = testDF.iloc[splitIndex:]
        return x_train, x_test, y_train, y_test

    def convertTo_ndarray(df: DataFrame):
        return df.values

    # def convertTo_dataframe(array: np.ndarray, index: pd.DatetimeIndex, columnLabels:list = []):
    #     return pd.DataFrame(array, columns = columnLabels, index = index)

    def convertTo_Dataframe(*arrays: np.ndarray, index: pd.DatetimeIndex, columnLabels:list = []):
        df = pd.DataFrame(index = index)
        iterator = iter(columnLabels)
        for i in arrays:
            if iterator.__length_hint__() > 0:
                df[next(iterator)] = i
            else:
                df[len(df.columns)] = i
        return df

    def generateXY_withDaysShift(df: DataFrame, daysShift: int):
        target = df['Close'].shift(-1).dropna().iloc[:len(df.index)-daysShift-1]
        # target = target.iloc[:len(target.values)-daysShift] # k target in k-1 row training

        df_copy = df.copy(deep=True)
        input = df_copy.drop('Close', axis=1).iloc[:len(df.index)-daysShift] # k-1 row
        return input, target


    def extendDataFrameWithLookbacksColumn(df: DataFrame, daysToLookBack: int):
        for k in range(0, daysToLookBack+1):
            colIndex = "Close_k-"+str(k)
            df[colIndex] = df['Close'].shift(k)
        # df["C_k"] = pd.Series(dtype=object)
        # for k in range(daysToLookBack, len(df.values)):
        #     df["C_k"].iloc[k] = np.asarray(df["Close"].iloc[k-daysToLookBack:k+1].values).reshape(-1,1)
        return df

    def plotBasicComparisonGraph(df: DataFrame):
        fig = px.line(df, markers=True, title="Close price actual vs predicted")
        fig.update_layout(yaxis=dict(title="Close price"), hovermode="x unified")
        return fig

    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Squared Error (RMSE)
        """
        # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse = mean_squared_error(y_true, y_pred)
        return rmse

    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return mape

    def calculate_mda(df: DataFrame):
        """
        Calculate Mean Directional Accuracy
        """
        signs = np.sign(df.diff())
        signs.dropna(inplace=True)
        AND = signs.iloc[:,0] * signs.iloc[:,1]
        return 100*(AND[AND > 0].sum()/len(AND.values))

    def createSlidingWindow(array: np.ndarray, daysToLookBack):
        # df["C_k"] = pd.Series(dtype=object)
        # for k in range(daysToLookBack, len(df.values)):
        #     df["C_k"].iloc[k] = np.asarray(df["Close"].iloc[k-daysToLookBack:k+1].values) #.reshape(-1,1)
        temp = []
        for k in range(daysToLookBack, len(array)):
            temp.append(array[k-daysToLookBack:k+1])
        return

if __name__ == "__main__":
    ticker = Ticker("ALE.WA")
    df = ticker.getData(365)
    print(df)
    # print(DataProcessing.minMaxNormalizeDataframe(df, ["Volume"]))
    print(DataProcessing.createSlidingWindow(df, 1))
    # DataProcessing.meanNormalizeDataframe(df)


    a = df['Close'].values
    # b = df['Open'].values
    # output = DataProcessing.convertTo_dataframe(a,b, index=df.index, columnLabels = ["L1"])
    # print(output)
    # # DataProcessing.plotBasicComparisonGraph(output)
    #
    # lookbacks = DataProcessing.extendDataFrameWithLookBacksColumn(df, 5)
    # print(lookbacks)