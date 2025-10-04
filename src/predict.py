# src/predict.py
import numpy as np
import pandas as pd
import joblib
import os

# Caminhos absolutos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'modelo_habitabilidade.pkl')
IMPUTER_PATH = os.path.join(ROOT_DIR, 'imputer.pkl')
DATA_PATH = os.path.join(ROOT_DIR, 'PythonNasaAppChallenge_VoyIAger', 'cumulative_2025.10.03_20.29.09.csv')
PREDICTIONS_PATH = os.path.join(ROOT_DIR, 'predictions.csv')

def predict_habitability(data, features=['koi_period', 'koi_prad', 'koi_teq', 'koi_insol'], threshold=0.2):
    """
    Predicts habitability of exoplanets based on input features.
    Args:
        data: DataFrame with columns [koi_period, koi_prad, koi_teq, koi_insol, koi_steff] or list of values
        features: List of feature names
        threshold: Probability threshold for classifying as 'Habitable'
    Returns:
        statuses: List of 'Habitable' or 'Non-Habitable'
        probs: List of probabilities of being habitable
    """
    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load(IMPUTER_PATH)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos {MODEL_PATH} ou {IMPUTER_PATH} não encontrados.")
        return ["Erro"], [0.0]

    # Converter entrada para DataFrame, se necessário
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame([data], columns=features)
    elif not isinstance(data, pd.DataFrame):
        print("❌ Erro: Dados de entrada devem ser DataFrame ou lista.")
        return ["Erro"], [0.0]

    # Garantir que as colunas estão corretas
    if not all(col in data.columns for col in features):
        print(f"❌ Erro: Colunas {features} não encontradas nos dados.")
        return ["Erro"], [0.0]

    # Imputar valores nulos
    X_imputed = pd.DataFrame(imputer.transform(data[features]), columns=features)

    # Prever
    probs = model.predict_proba(X_imputed)[:, 1]  # Probabilidade da classe "Habitável"
    statuses = ["Habitável" if prob >= threshold else "Não Habitável" for prob in probs]
    return statuses, probs

def predict_from_csv():
    """
    Loads the CSV, filters confirmed exoplanets, makes predictions, and saves to predictions.csv.
    """
    print("=" * 60)
    print("🔮 PREVISÃO DE HABITABILIDADE PARA TODOS OS EXOPLANETAS CONFIRMADOS NO CSV")
    print("=" * 60)

    # Carregar dados do CSV
    print(f"📂 Carregando arquivo: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH, comment='#', skiprows=52, low_memory=False)
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return

    # Filtrar planetas confirmados
    confirmed_planets = df[df['koi_disposition'] == 'CONFIRMED'].copy()
    print(f"🪐 Planetas confirmados no CSV: {len(confirmed_planets)}")

    # Selecionar features
    features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff']
    if not all(col in df.columns for col in features):
        print(f"❌ Erro: Colunas {features} não encontradas no CSV.")
        return

    # Fazer predições
    statuses, probs = predict_habitability(confirmed_planets[features])

    # Adicionar colunas ao DataFrame
    confirmed_planets['status'] = statuses
    confirmed_planets['prob_habitavel'] = probs

    # Contar planetas habitáveis
    num_habitable = sum(status == "Habitável" for status in statuses)
    print(f"🌍 Planetas classificados como Habitáveis: {num_habitable} ({num_habitable/len(statuses):.2%})")

    # Top 10 planetas mais habitáveis
    top_habitable = confirmed_planets[confirmed_planets['status'] == 'Habitável'].nlargest(10, 'prob_habitavel')
    print("\n🏆 Top 10 Planetas Mais Habitáveis:")
    print(top_habitable[['kepoi_name', 'prob_habitavel']])

    # Selecionar colunas relevantes para salvar
    output_columns = ['kepoi_name', 'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'status', 'prob_habitavel']
    confirmed_planets[output_columns].to_csv(PREDICTIONS_PATH, index=False)
    print(f"💾 Predições salvas em: {PREDICTIONS_PATH}")

if __name__ == "__main__":
    # Previsões para todos os exoplanetas do CSV
    predict_from_csv()

    # Previsões para casos de teste fixos
    print("\n============================================================")
    print("🌍 TESTE DE PREVISÃO PARA CASOS FIXOS")
    print("============================================================")

    test_planets = [
        {"name": "Planeta 1 (Terra-like)", "data": [365.25, 1.0, 288.0, 1.0]},           # 4 valores
        {"name": "Planeta 2 (Mercúrio-like)", "data": [88.0, 0.38, 440.0, 6.5]},         # 4 valores
        {"name": "Planeta 3 (Distante/Grande)", "data": [500.0, 5.0, 150.0, 0.1]}        # 4 valores
    ]

    for planet in test_planets:
        status, prob = predict_habitability(planet["data"], features=['koi_period', 'koi_prad', 'koi_teq', 'koi_insol'])
        print(f"{planet['name']}: {status[0]} (Probabilidade: {prob[0]:.2%})")