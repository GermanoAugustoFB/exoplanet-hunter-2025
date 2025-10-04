# src/predict.py
import pandas as pd
import numpy as np
import joblib
import os

# Caminhos relativos
MODEL_PATH = os.path.join('..', 'modelo_habitabilidade.pkl')
IMPUTER_PATH = os.path.join('..', 'imputer.pkl')

# Função para previsão (para uso no Streamlit)
def predict_habitability(period, prad, teq, insol):
    """
    Prever habitabilidade de um exoplaneta.
    Entradas: period (dias), prad (raios terrestres), teq (Kelvin), insol (fluxo terrestre).
    Retorna: (status, probabilidade de habitável).
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMPUTER_PATH):
        return "Erro: Modelo ou imputer não encontrados. Execute model_training.py primeiro.", 0.0

    # Carregar modelo e imputer
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    # Criar DataFrame com os dados de entrada
    features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']
    data = pd.DataFrame([[period, prad, teq, insol]], columns=features)

    # Pré-processar
    data_imputed = pd.DataFrame(imputer.transform(data), columns=features)

    # Prever
    prediction = model.predict(data_imputed)[0]
    probability = model.predict_proba(data_imputed)[0, 1]
    status = 'Habitável' if prediction == 1 else 'Não Habitável'

    return status, probability

# Exemplo de uso standalone
if __name__ == "__main__":
    print("=" * 60)
    print("🌍 TESTE DE PREVISÃO DE HABITABILIDADE")
    print("=" * 60)

    # Dados fictícios para teste
    test_planets = [
        (365.25, 1.0, 288, 1.0),   # Terra-like
        (88, 0.38, 440, 7.0),      # Mercúrio-like
        (1000, 2.5, 150, 0.2),     # Distante/grande
    ]

    for i, (period, prad, teq, insol) in enumerate(test_planets, 1):
        status, prob = predict_habitability(period, prad, teq, insol)
        print(f"Planeta {i}: {status} (probabilidade de habitável: {prob:.2%})")