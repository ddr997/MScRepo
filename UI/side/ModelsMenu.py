from ModelCreator import ModelCreator
from StateData import StateData
import streamlit as st
class ModelsMenu:
    def __init__(self, stateData: StateData):
        self.availableModels = ["SVR", "LSTM", "GradientBoost", "XGBoost"]
        st.write("Make prediction")
        with st.sidebar:
            stateData.choosenModel = st.selectbox("Choose prediction algorithm:", self.availableModels)

            if(stateData.choosenModel == "LSTM"):
                self.LSTM(stateData)
                None
        return

    def SVR(self, stateData):
        return

    def GradientBoost(self, stateData):
        return

    def XGBoost(self, stateData):
        return

    def LSTM(self, stateData):
        return