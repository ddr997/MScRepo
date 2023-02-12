from ModelCreator import ModelCreator
from StateData import StateData
import streamlit as st
import numpy as np
from models.DataProcessing import DataProcessing
from models.SVR import SVRmodel
class ModelsMenu:
    def __init__(self, stateData: StateData):
        self.availableModels = ["SVR", "LSTM", "GradientBoost", "XGBoost"]
        with st.sidebar:
            stateData.choosenModel = st.selectbox("Choose prediction algorithm:", self.availableModels)

            if(stateData.choosenModel == "LSTM"):
                self.LSTM(stateData)
                None

            if(stateData.choosenModel == "SVR"):
                self.SVR(stateData)
                None

        return

    def SVR(self, stateData):
        submit = False
        with st.form("SVR options"):
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            C = st.number_input("C", value=10.0)
            epsilon = st.number_input("Epsilon", value=0.1)
            submit = st.form_submit_button(label="Make prediction")
            if(submit):
                df = stateData.dataFrame.drop(["Open", "High", "Volume", "Low"], axis=1)
                # df = DataProcessing.extendDataFrameWithLookbacksColumn(df, 5)
                df = df.replace(np.nan, 0)
                model = SVRmodel()
                model.prepareModel(kernel=kernel, C=C, epsilon=epsilon)
                stateData.predictionFigure = model.createPrediction(df, 1)
                st.experimental_rerun()
        return

    def GradientBoost(self, stateData):
        return

    def XGBoost(self, stateData):
        return

    def LSTM(self, stateData):
        return