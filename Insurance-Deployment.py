import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained machine learning model
model = joblib.load("RFC_model.h5")
preprocessing = joblib.load("Pre_pipeline_.h5")
#Vehicle_Age_list = joblib.load(r"C:\Users\Dell\Desktop\Data science projects\Final project\insurance\Dump\Vehicle_Age_list.h5")

from sklearn.preprocessing import OneHotEncoder


def predict(Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage):
    input_df = pd.DataFrame(columns=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'])
    input_df.at[0,'Gender']= Gender
    input_df.at[0,'Age']= Age
    input_df.at[0,'Driving_License']= Driving_License
    input_df.at[0,'Region_Code']= Region_Code
    input_df.at[0,'Previously_Insured']= Previously_Insured
    input_df.at[0,'Vehicle_Age']= Vehicle_Age
    input_df.at[0,'Vehicle_Damage']= Vehicle_Damage
    input_df.at[0,'Annual_Premium']= Annual_Premium
    input_df.at[0,'Policy_Sales_Channel']= Policy_Sales_Channel
    input_df.at[0,'Vintage']= Vintage


    pre_processing = preprocessing.transform(input_df)

    if model.predict(pre_processing)[0] == 1:
        result = "Expected to respond !"
    elif model.predict(pre_processing)[0] == 0:
        result = "Not expected to respond !"

    return result


st.title("Vechial Insurance Predictions")

def main():
    Gender= st.radio("Gender:", ['Male', 'Female'])
    Age = st.number_input("Age:")
    Driving_License = st.radio("Driving License", [1,0])
    Region_Code = st.number_input("Region Code (1-100)")
    select_Previously_Insured=st.radio("Previously Insured", ['Yes','No'])
    if select_Previously_Insured == 'yes':
        Previously_Insured=1
    elif select_Previously_Insured == 'No':
        Previously_Insured=0
    #Previously_Insured = st.radio("Previously Insured", [1,0])
    Vehicle_Age = st.selectbox("Vehicle Age", ['< 1 Year','1-2 Year','> 2 Years'])
    Vehicle_Damage = st.radio("Vehicle Damaged Before :", ['Yes', 'No'])
    Annual_Premium = st.number_input("Annual Premium",)
    Policy_Sales_Channel = st.number_input("Policy Sales Channel (1-100)")
    Vintage = st.number_input("Vintage",)

    if st.button('predict'):
        result=predict(Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage)
        st.write(result)
if __name__ =='__main__':
    main()

