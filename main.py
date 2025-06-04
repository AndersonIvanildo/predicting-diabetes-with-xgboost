import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

""" READING MODELS FILE AND LOAD"""
path = os.path.join('data', 'models', 'diabetes_classifier3Classes.json')

model_xgb = xgb.XGBClassifier()
model_xgb.load_model(path)



# 2. Função para traduzir a saída
def traduzir_classe(valor):
    return {0: "Não diabético", 1: "Pré-diabético", 2: "Diabético"}.get(valor, "Desconhecido")

# 3. Título
st.title("Predição de Diabetes com XGBoost")

# 4. Inputs do usuário
age = st.number_input("Idade", min_value=0, max_value=120, step=1)
urea = st.number_input("Ureia")
cr = st.number_input("Creatinina")
hba1c = st.number_input("HbA1c")
chol = st.number_input("Colesterol Total")
tg = st.number_input("Triglicerídeos")
hdl = st.number_input("HDL")
ldl = st.number_input("LDL")
vldl = st.number_input("VLDL")
bmi = st.number_input("IMC (BMI)")

# Ajust gender to view
gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
gender_f = 1 if gender == "Feminino" else 0
gender_m = 1 if gender == "Masculino" else 0


if st.button("Prever"):
    # Create a Dataframe to apply in model
    input_data = pd.DataFrame([[
        age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi,
        gender_f, gender_m
    ]], columns=[
        'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL',
        'BMI', 'Gender_F', 'Gender_M'
    ])

    prediction = model_xgb.predict(input_data)[0]
    classe = traduzir_classe(prediction)

    st.success(f"Resultado: {classe}")