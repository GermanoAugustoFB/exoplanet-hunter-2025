# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Caminhos relativos (corrigido)
DATA_PATH = os.path.join('PythonNasaAppChallenge_VoyIAger', 'cumulative_2025.10.03_20.29.09.csv')
MODEL_PATH = os.path.join('..', 'modelo_habitabilidade.pkl')
IMPUTER_PATH = os.path.join('..', 'imputer.pkl')

print("=" * 60)
print("🚀 TREINAMENTO DE IA PARA EXOPLANETAS: HABITABILIDADE")
print("=" * 60)

# Verificar se o arquivo existe
if not os.path.exists(DATA_PATH):
    print(f"❌ Erro: Arquivo não encontrado em {DATA_PATH}")
    print("Dica: Liste os arquivos com 'ls PythonNasaAppChallenge_VoyIAger/'")
    print("Ou procure o arquivo com 'find ~/Documentos -name \"cumulative_2025*.csv\"'")
    exit(1)

# Carregar dados
print(f"📂 Carregando arquivo: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, comment='#', skiprows=52, low_memory=False)

# Filtrar planetas confirmados
confirmed_planets = df[df['koi_disposition'] == 'CONFIRMED'].copy()
print(f"🪐 Planetas confirmados: {len(confirmed_planets)}")

# Selecionar features
features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']
X = confirmed_planets[features].copy()

# Tratar valores nulos
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

# Criar label de habitabilidade
y = np.where(
    (confirmed_planets['koi_insol'].fillna(0) > 0.38) &
    (confirmed_planets['koi_insol'].fillna(0) < 1.1) &
    (X_imputed['koi_prad'] < 1.8), 1, 0
)
print(f"🎯 Distribuição de labels: {np.bincount(y)} (0: Não Habitável, 1: Habitável)")

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Acurácia do modelo: {accuracy:.2%}")
print("\n📈 Relatório de classificação:")
print(classification_report(y_test, y_pred, target_names=['Não Habitável', 'Habitável']))

# Matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Habitável', 'Habitável'],
            yticklabels=['Não Habitável', 'Habitável'])
plt.title('Matriz de Confusão: Previsão de Habitabilidade')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig(os.path.join('..', 'matriz_confusao.png'))
plt.show()

# Importância das features
importances = pd.DataFrame({
    'Feature': features,
    'Importância': model.feature_importances_
}).sort_values('Importância', ascending=False)
print(f"\n🔍 Importância das features:")
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', x='Feature', y='Importância', color='skyblue')
plt.title('Importância das Features no Modelo')
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join('..', 'importancia_features.png'))
plt.show()

# Salvar modelo e imputer
joblib.dump(model, MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)
print(f"\n💾 Modelo salvo em: {MODEL_PATH}")
print(f"💾 Imputer salvo em: {IMPUTER_PATH}")