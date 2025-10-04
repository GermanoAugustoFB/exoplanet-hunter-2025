# src/app.py
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st

# Adicionar o diretório raiz ao sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.predict import predict_habitability
from src.visualization import plot_confusion_matrix_plotly, plot_feature_importance_plotly, plot_label_distribution_plotly

# Caminhos absolutos
Y_TRUE_PATH = os.path.join(ROOT_DIR, 'y_true.npy')
Y_PRED_PATH = os.path.join(ROOT_DIR, 'y_pred.npy')
Y_LABELS_PATH = os.path.join(ROOT_DIR, 'y_labels.npy')
FEATURE_IMPORTANCES_PATH = os.path.join(ROOT_DIR, 'feature_importances.npy')
PREDICTIONS_PATH = os.path.join(ROOT_DIR, 'predictions.csv')

st.title("🌌 Exoplanet Hunter 2025")
st.header("Previsão de Habitabilidade de Exoplanetas")

# Formulário de previsão
period = st.number_input("Período Orbital (dias)", min_value=0.0, value=365.25)
prad = st.number_input("Raio Planetário (raios terrestres)", min_value=0.0, value=1.0)
teq = st.number_input("Temperatura de Equilíbrio (Kelvin)", min_value=0.0, value=288.0)
insol = st.number_input("Insolação (fluxo terrestre)", min_value=0.0, value=1.0)

if st.button("Prever"):
    data = [period, prad, teq, insol]  # Só 4 features!
    status, prob = predict_habitability(data, features=['koi_period', 'koi_prad', 'koi_teq', 'koi_insol'])
    st.write(f"**Resultado**: {status[0]}")
    st.write(f"**Probabilidade de ser habitável**: {prob[0]:.2%}")

# Carregar dados do treinamento
st.header("Resultados do Modelo")
try:
    y_true = np.load(Y_TRUE_PATH)
    y_pred = np.load(Y_PRED_PATH)
    y = np.load(Y_LABELS_PATH)
    importances = np.load(FEATURE_IMPORTANCES_PATH)
    st.success("Dados reais do treinamento carregados com sucesso!")
except FileNotFoundError:
    st.warning("Arquivos de dados do treinamento não encontrados. Usando dados fictícios.")
    y_true = np.array([0] * 548 + [1] * 2)
    y_pred = np.array([0] * 548 + [0] * 2)
    y = np.array([0] * 2736 + [1] * 10)
    importances = [0.08, 0.27, 0.30, 0.28, 0.07]

features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']  # Remova 'koi_steff'

st.subheader("Matriz de Confusão")
fig_cm = plot_confusion_matrix_plotly(y_true, y_pred)
if fig_cm:
    st.plotly_chart(fig_cm)
else:
    st.error("Erro ao gerar a matriz de confusão.")

st.subheader("Importância das Features")
fig_fi = plot_feature_importance_plotly(features, importances)
if fig_fi:
    st.plotly_chart(fig_fi)
else:
    st.error("Erro ao gerar o gráfico de importância das features.")

st.subheader("Distribuição de Labels")
fig_ld = plot_label_distribution_plotly(y)
if fig_ld:
    st.plotly_chart(fig_ld)
else:
    st.error("Erro ao gerar o gráfico de distribuição de labels.")

# Exibir predições do CSV
st.header("Predições para Exoplanetas do CSV")
try:
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    st.write(f"Total de planetas: {len(predictions_df)}")
    st.write(f"Planetas habitáveis: {len(predictions_df[predictions_df['status'] == 'Habitável'])}")
    
    # Filtro para mostrar apenas habitáveis
    show_habitable = st.checkbox("Mostrar apenas planetas habitáveis", value=False)
    if show_habitable:
        filtered_df = predictions_df[predictions_df['status'] == 'Habitável']
    else:
        filtered_df = predictions_df
    
    st.write("Predições salvas no CSV:")
    st.dataframe(filtered_df[['kepoi_name', 'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'status', 'prob_habitavel']])
except FileNotFoundError:
    st.warning("Arquivo predictions.csv não encontrado. Rode predict.py para gerar.")