
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import xgboost as xgb

# ===============================
# Dicionário com textos PT/EN
# ===============================
texts = {
    'pt': {
        'title': "Predição de Diabetes com XGBoost",
        'intro': (
            "Bem-vindo ao preditor de diabetes!\n\n"
            "Este aplicativo utiliza um modelo de aprendizado de máquina (XGBoost) para classificar o risco de diabetes "
            "em três categorias: Não diabético, Pré-diabético e Diabético.\n\n"
            "Para isso, ele analisa variáveis clínicas e bioquímicas importantes, como idade, níveis de glicose média (HbA1c), "
            "colesterol, triglicerídeos e indicadores de função renal.\n\n"
            "Ao preencher os dados, o modelo avalia o risco e apresenta uma previsão junto com a probabilidade associada, "
            "ajudando a entender melhor seu estado de saúde.\n\n"
            "Importante: Este é um suporte à decisão, não substitui avaliação médica."
        ),
        'age': "Idade",
        'age_help': "Idade do paciente em anos.",
        'urea': "Ureia",
        'urea_help': "Nível de ureia no sangue, indicador de função renal.",
        'cr': "Creatinina",
        'cr_help': "Nível de creatinina no sangue, indicador de função renal.",
        'hba1c': "HbA1c",
        'hba1c_help': "Hemoglobina glicada, média dos níveis de açúcar no sangue nos últimos meses.",
        'chol': "Colesterol Total",
        'chol_help': "Nível total de colesterol no sangue.",
        'tg': "Triglicerídeos",
        'tg_help': "Nível de triglicerídeos no sangue.",
        'hdl': "HDL",
        'hdl_help': "Colesterol 'bom' (lipoproteína de alta densidade).",
        'ldl': "LDL",
        'ldl_help': "Colesterol 'ruim' (lipoproteína de baixa densidade).",
        'vldl': "VLDL",
        'vldl_help': "Colesterol de densidade muito baixa.",
        'bmi': "IMC (BMI)",
        'bmi_help': "Índice de Massa Corporal (peso em kg / altura² em m).",
        'gender': "Gênero",
        'gender_options': ["Masculino", "Feminino"],
        'gender_help': "O gênero do paciente (F para Feminino, M para Masculino).",
        'predict_btn': "Prever",
        'result_not': "Não diabético",
        'result_pre': "Pré-diabético",
        'result_diab': "Diabético",
        'result_unknown': "Desconhecido",
        'result_text': "Resultado: ",
        'prob_title': "Probabilidades das classes:",
        'select_language': "Selecionar idioma / Select Language"
    },
    'en': {
        'title': "Diabetes Prediction with XGBoost",
        'intro': (
            "Welcome to the diabetes predictor!\n\n"
            "This application uses a machine learning model (XGBoost) to classify diabetes risk into three categories: "
            "Non-diabetic, Pre-diabetic, and Diabetic.\n\n"
            "It analyzes important clinical and biochemical variables such as age, average glucose levels (HbA1c), cholesterol, "
            "triglycerides, and kidney function indicators.\n\n"
            "By entering the data, the model evaluates the risk and presents a prediction along with the associated probability, "
            "helping you better understand your health status.\n\n"
            "Important: This is a decision support tool and does not replace medical evaluation."
        ),
        'age': "Age",
        'age_help': "Patient's age in years.",
        'urea': "Urea",
        'urea_help': "Blood urea level, indicator of kidney function.",
        'cr': "Creatinine",
        'cr_help': "Blood creatinine level, indicator of kidney function.",
        'hba1c': "HbA1c",
        'hba1c_help': "Glycated hemoglobin, average blood sugar levels over recent months.",
        'chol': "Total Cholesterol",
        'chol_help': "Total blood cholesterol level.",
        'tg': "Triglycerides",
        'tg_help': "Blood triglycerides level.",
        'hdl': "HDL",
        'hdl_help': "Good cholesterol (high-density lipoprotein).",
        'ldl': "LDL",
        'ldl_help': "Bad cholesterol (low-density lipoprotein).",
        'vldl': "VLDL",
        'vldl_help': "Very low-density lipoprotein cholesterol.",
        'bmi': "BMI",
        'bmi_help': "Body Mass Index (weight in kg / height² in meters).",
        'gender': "Gender",
        'gender_options': ["Male", "Female"],
        'gender_help': "Patient's gender (F for Female, M for Male).",
        'predict_btn': "Predict",
        'result_not': "Non-diabetic",
        'result_pre': "Pre-diabetic",
        'result_diab': "Diabetic",
        'result_unknown': "Unknown",
        'result_text': "Result: ",
        'prob_title': "Class Probabilities:",
        'select_language': "Select Language / Selecionar idioma"
    }
}

# ===============================
# Carregar modelo
# ===============================
# READING MODELS FILE AND LOAD
path = os.path.join('data', 'models', 'diabetes_classifier3Classes.json')
model_xgb = xgb.XGBClassifier()
model_xgb.load_model(path)

# ===============================
# Função para traduzir a saída
# ===============================
def traduzir_classe(valor, lang='pt'):
    map_pt = {
        0: texts['pt']['result_not'],
        1: texts['pt']['result_pre'],
        2: texts['pt']['result_diab']
    }
    map_en = {
        0: texts['en']['result_not'],
        1: texts['en']['result_pre'],
        2: texts['en']['result_diab']
    }
    if lang == 'en':
        return map_en.get(valor, texts['en']['result_unknown'])
    else:
        return map_pt.get(valor, texts['pt']['result_unknown'])

# ===============================
# Interface Streamlit
# ===============================

# Seleção de idioma
language = st.selectbox(texts['pt']['select_language'], ['Português', 'English'])
lang_code = 'pt' if language == 'Português' else 'en'

st.title(texts[lang_code]['title'])
st.write(texts[lang_code]['intro'])

# Layout em 2 colunas
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(texts[lang_code]['age'], min_value=0, max_value=120, step=1, help=texts[lang_code]['age_help'])
    urea = st.number_input(texts[lang_code]['urea'], help=texts[lang_code]['urea_help'])
    cr = st.number_input(texts[lang_code]['cr'], help=texts[lang_code]['cr_help'])
    hba1c = st.number_input(texts[lang_code]['hba1c'], help=texts[lang_code]['hba1c_help'])
    chol = st.number_input(texts[lang_code]['chol'], help=texts[lang_code]['chol_help'])
    tg = st.number_input(texts[lang_code]['tg'], help=texts[lang_code]['tg_help'])

with col2:
    hdl = st.number_input(texts[lang_code]['hdl'], help=texts[lang_code]['hdl_help'])
    ldl = st.number_input(texts[lang_code]['ldl'], help=texts[lang_code]['ldl_help'])
    vldl = st.number_input(texts[lang_code]['vldl'], help=texts[lang_code]['vldl_help'])
    bmi = st.number_input(texts[lang_code]['bmi'], help=texts[lang_code]['bmi_help'])
    gender = st.selectbox(texts[lang_code]['gender'], texts[lang_code]['gender_options'], help=texts[lang_code]['gender_help'])

# Transformação do gênero para hot encoding
gender_f = 1 if gender in ['Feminino', 'Female'] else 0
gender_m = 1 if gender in ['Masculino', 'Male'] else 0

if st.button(texts[lang_code]['predict_btn']):
    # Montar DataFrame
    input_data = pd.DataFrame([[
        age, urea, cr, hba1c, chol, tg,
        hdl, ldl, vldl, bmi,
        gender_f, gender_m
    ]], columns=[
        'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG',
        'HDL', 'LDL', 'VLDL', 'BMI',
        'Gender_F', 'Gender_M'
    ])

    # Previsão
    prediction = model_xgb.predict(input_data)[0]
    proba = model_xgb.predict_proba(input_data)[0]

    classe = traduzir_classe(prediction, lang=lang_code)

    st.success(texts[lang_code]['result_text'] + classe)

    st.write(texts[lang_code]['prob_title'])
    # Barra para cada classe
    probs_df = pd.DataFrame({
        texts[lang_code]['result_not']: [proba[0]],
        texts[lang_code]['result_pre']: [proba[1]],
        texts[lang_code]['result_diab']: [proba[2]],
    })
    st.bar_chart(probs_df.T, y_label="Prob")

