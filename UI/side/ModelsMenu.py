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
from models.XGBoost import XGBoostModel

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
            C = st.number_input("C", value=1.0, format="%.5f")
            epsilon = st.number_input("Epsilon", value=1e-3, format="%.5f")
            submit = st.form_submit_button(label="Make prediction")
            arguments = dict(kernel=kernel, C=C, epsilon=epsilon)
            if(submit):
                self.runPrediction(stateData, SVRmodel(), **arguments)
        return

    def LSTM(self, stateData):
        with st.form("LSTM options"):
            units = st.number_input("Units in hidden layer", value=28, step = 1)
            activation = st.selectbox("Activation function", ["sigmoid", 'tanh'])
            epochs = st.number_input("Epochs", value=21, step = 1)
            batch_size = st.number_input("Batch size", value=32, step = 1)
            submit = st.form_submit_button(label="Make prediction")
            arguments = dict(
                units=units,
                epochs=epochs,
                batch_size=batch_size,
                activation=activation,
            )
            if(submit):
                self.runPrediction(stateData, LSTMmodel(), **arguments)
        return

    def GradientBoost(self, stateData):
        with st.form("GBoost options"):
            learning_rate = st.number_input("Learning rate", value=0.2, step=0.001, format="%.5f")
            min_samples_split = st.number_input("Min samples split", value=2, step = 1)
            min_samples_leaf = st.number_input("Min samples leaf", value=1, step = 1)
            max_depth = st.number_input("Max Depth", value=14, step = 1)
            n_estimators = st.number_input("Number of estimators", value=100, step = 1)
            loss = st.selectbox("Loss function", ['squared_error'])
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
        with st.form("XGBoost options"):

            learning_rate = st.number_input("Learning rate", value=0.2, step=0.001, format="%.5f")
            n_estimators= st.number_input("Number of estimators", value=100, step = 1)
            max_depth= st.number_input("Max Depth", value=14, step = 1)
            max_leaves = st.number_input("Max Leaves", value=0, step = 1, help="0 means no limit")
            booster = st.selectbox("Loss function", ['gbtree', "gblinear"])
            gamma = st.number_input("Gamma", value=0.0, step=0.001, format="%.5f")
            reg_alpha = st.number_input("L1 regularization", value=0.0, step=0.001, format="%.5f")
            reg_lambda = st.number_input("L2 regularization", value=1.0, step=0.001, format="%.5f")
            submit = st.form_submit_button(label="Make prediction")
            arguments = dict(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_leaves=max_leaves,
                booster=booster,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda
            )
            if(submit):
                self.runPrediction(stateData, XGBoostModel(), **arguments)
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
        stateData.predictionDataFrame[predDF.columns[1]] = predDF.iloc[:, 1]
        stateData.predictionDataFrame.dropna(inplace=True)

        stateData.metrics.update(model.metricsDict)

        st.experimental_rerun()