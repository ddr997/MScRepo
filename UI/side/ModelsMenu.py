from pandas import DataFrame
from ModelCreator import ModelCreator
from StateData import StateData
import streamlit as st
import numpy as np
from models.DataProcessing import DataProcessing
from models.SVR import SVRmodel
from models.GBoost import GBoost
from models.LSTMmodel import LSTMmodel
import pandas as pd

class ModelsMenu:
    def __init__(self, stateData: StateData):
        self.availableModels = ["SVR", "GradientBoost", "XGBoost", "LSTM"]
        self.shifts = 0
        self.includeDays = 0

        with st.sidebar:
            stateData.choosenModel = st.selectbox("Choose prediction algorithm:", self.availableModels)
            self.shifts = st.number_input("Shift days from last session:", value=1, step=1, help="0 means predicting day we don't have data about yet")
            self.includeDays = st.number_input("Include past close prices:", value=-1, step=1, help="-1 means no close prices; 0 means only C(k)")

            if(stateData.choosenModel == "SVR"):
                self.SVR(stateData)
                None

            if(stateData.choosenModel == "GradientBoost"):
                self.GradientBoost(stateData)
                None

            if(stateData.choosenModel == "XGBoost"):
                self.XGBoost(stateData)
                None

            if(stateData.choosenModel == "LSTM"):
                self.LSTM(stateData)
                None

        return

    def SVR(self, stateData):
        with st.form("SVR options"):
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            C = st.number_input("C", value=10.0)
            epsilon = st.number_input("Epsilon", value=0.1)
            submit = st.form_submit_button(label="Make prediction")
            arguments = dict(kernel=kernel, C=C, epsilon=epsilon)
            if(submit):
                self.runPrediction(stateData, SVRmodel(), **arguments)
        return

    def GradientBoost(self, stateData):
        with st.form("GB options"):
            learning_rate = st.number_input("Learning rate", value=0.1)
            loss= st.selectbox("Loss function", ['squared_error'])
            n_estimators= st.number_input("Number of estimators", value=100, step = 1)
            min_samples_split= st.number_input("Min samples split", value=2, step = 1)
            min_samples_leaf= st.number_input("Min samples leaf", value=1, step = 1)
            max_depth= st.number_input("Max Depth", value=3, step = 1)
            submit = st.form_submit_button(label="Make prediction")
            arguments = dict(learning_rate=learning_rate,
                             loss=loss,
                             n_estimators=n_estimators,
                             min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf,
                             max_depth=max_depth)
            if(submit):
                self.runPrediction(stateData, GBoost(), **arguments)
        return

    def XGBoost(self, stateData):
        return

    def LSTM(self, stateData):
        with st.form("LSTM options"):
            submit = st.form_submit_button(label="Make prediction")
            if(submit):
                self.runPrediction(stateData, LSTMmodel())
        return

    def prepareGlobalSettings(self, df : DataFrame):
        df = df.drop(["Open", "High", "Volume", "Low"], axis=1)
        DataProcessing.extendDataFrameWithLookbacksColumn(df, self.includeDays)
        df.replace(np.nan, 0, inplace=True)
        return df

    def runPrediction(self, stateData: StateData, model, **kwargs):
        df = self.prepareGlobalSettings(stateData.dataFrame)
        model.prepareModel(**kwargs)
        predDF = model.createPrediction(df, self.shifts)
        stateData.predictionDataFrame = predDF
        st.experimental_rerun()