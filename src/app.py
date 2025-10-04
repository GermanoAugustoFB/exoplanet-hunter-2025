# src/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_habitability
from src.visualization import plot_confusion_matrix_plotly, plot_feature_importance_plotly, plot_label_distribution_plotly
import numpy as np
import streamlit as st

# Caminhos para os arquivos salvos
BASE_PATH = os.path.join('..')
Y_TRUE_PATH = os.path.join(BASE_PATH, 'y_true.npy')
Y_PRED_PATH = os.path.join(BASE_PATH, 'y_pred.npy')
Y_LABELS_PATH = os.path.join(BASE_PATH, 'y_labels.npy')
FEATURE_IMPORTANCES_PATH = os.path.join(BASE_PATH, 'feature_importances.npy')

st.title("🌌 Exoplanet Hunter 2025")
st.header("Previsão de Habitabilidade de Exoplanetas")

# Formulário de previsão
period = st.number_input("Período Orbital (dias)", min_value=0.0, value=365.25)
prad = st.number_input("Raio Planetário (raios terrestres)", min_value=0.0, value=1.0)
teq = st.number_input("Temperatura de Equilíbrio (Kelvin)", min_value=0.0, value=288.0)
insol = st.number_input("Insolação (fluxo terrestre)", min_value=0.0, value=1.0)

if st.button("Prever"):
    status, prob = predict_habitability(period, prad, teq, insol)
    st.write(f"**Resultado**: {status}")
    st.write(f"**Probabilidade de ser habitável**: {prob:.2%}")

# Carregar dados do treinamento (com fallback para dados fictícios)
st.header("Resultados do Modelo")
try:
    y_true = np.load(Y_TRUE_PATH)
    y_pred = np.load(Y_PRED_PATH)
    y = np.load(Y_LABELS_PATH)
    importances = np.load(FEATURE_IMPORTANCES_PATH)
except FileNotFoundError:
    st.warning("Arquivos de dados do treinamento não encontrados. Usando dados fictícios.")
    y_true = np.array([0] * 548 + [1] * 2)  # Do treinamento anterior
    y_pred = np.array([0] * 548 + [0] * 2)  # Do treinamento anterior
    y = np.array([0] * 2736 + [1] * 10)  # Distribuição de labels
    importances = [0.090789, 0.290536, 0.321802, 0.296872]

features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']

st.subheader("Matriz de Confusão")
fig_cm = plot_confusion_matrix_plotly(y_true, y_pred)
if fig_cm:
    st.plotly_chart(fig_cm)

st.subheader("Importância das Features")
fig_fi = plot_feature_importance_plotly(features, importances)
st.plotly_chart(fig_fi)

st.subheader("Distribuição de Labels")
fig_ld = plot_label_distribution_plotly(y)
st.plotly_chart(fig_ld)