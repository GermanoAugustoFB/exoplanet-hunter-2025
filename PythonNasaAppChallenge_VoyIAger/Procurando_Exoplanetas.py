import pandas as pd
import numpy as np

# Carregar dados corretamente
df = pd.read_csv('cumulative_2025.10.03_20.29.09.csv',
                 comment='#',
                 skiprows=52,
                 low_memory=False)

print("=" * 60)
print("🌌 NASA EXOPLANET ARCHIVE - ANÁLISE INICIAL")
print("=" * 60)

print(f"📊 Total de registros: {len(df):,}")
print(f"🔤 Total de colunas: {len(df.columns)}")
print(f"📏 Dimensões: {df.shape}")

print(f"\n🎯 Primeiros 5 exoplanetas:")
print(df.head())

print(f"\n❌ Valores nulos por coluna (top 10):")
null_counts = df.isnull().sum().sort_values(ascending=False)
print(null_counts.head(10))

print(f"\n📈 Estatísticas básicas:")
print(df.describe())

print(f"\n🔍 Tipos de dados:")
print(df.dtypes.head(15))

# Filtrar apenas planetas confirmados
planetas_confirmados = df[df['koi_disposition'] == 'CONFIRMED']

print(f"🪐 Planetas confirmados: {len(planetas_confirmados)}")
print(f"📊 Distribuição de status:")
print(df['koi_disposition'].value_counts())

# Filtrar apenas planetas confirmados para análise focada
confirmed_planets = df[df['koi_disposition'] == 'CONFIRMED'].copy()

print(f"🌌 ANÁLISE DOS PLANETAS CONFIRMADOS")
print("=" * 50)
print(f"🪐 Total de planetas confirmados: {len(confirmed_planets)}")
print(f"📊 Colunas disponíveis: {len(confirmed_planets.columns)}")

# Estatísticas dos planetas confirmados
print(f"\n📈 Estatísticas dos planetas confirmados:")
print(confirmed_planets[['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']].describe())

# Filtrar planetas na zona habitável conservadora
# (insolação similar à Terra: entre 0.38 e 1.1 vezes a insolação terrestre)

habitable_zone = confirmed_planets[
    (confirmed_planets['koi_insol'] > 0.38) &
    (confirmed_planets['koi_insol'] < 1.1) &
    (confirmed_planets['koi_prad'] < 1.8)  # Planetas rochosos
]

print(f"\n🌍 PLANETAS NA ZONA HABITÁVEL:")
print(f"📍 Planetas potencialmente habitáveis: {len(habitable_zone)}")

if len(habitable_zone) > 0:
    print(f"\n🔍 Planetas mais promissores:")
    print(habitable_zone[['kepoi_name', 'kepler_name', 'koi_prad', 'koi_insol', 'koi_teq']].head(10))

# Classificar planetas por tamanho
def classify_planet_size(radius):
    if pd.isna(radius):
        return 'Unknown'
    elif radius < 1.25:
        return 'Earth-like'
    elif radius < 2:
        return 'Super-Earth'
    elif radius < 6:
        return 'Neptune-like'
    else:
        return 'Jupiter-like'

confirmed_planets['size_category'] = confirmed_planets['koi_prad'].apply(classify_planet_size)

print(f"\n📏 DISTRIBUIÇÃO POR TAMANHO:")
size_distribution = confirmed_planets['size_category'].value_counts()
print(size_distribution)

import matplotlib.pyplot as plt

# Gráfico de distribuição de tamanhos
plt.figure(figsize=(10, 6))
size_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribuição de Planetas Confirmados por Tamanho')
plt.xlabel('Categoria de Tamanho')
plt.ylabel('Número de Planetas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de período orbital vs tamanho
plt.figure(figsize=(10, 6))
plt.scatter(confirmed_planets['koi_period'], confirmed_planets['koi_prad'],
            alpha=0.5, color='red')
plt.xlabel('Período Orbital (dias)')
plt.ylabel('Raio Planetário (Terra = 1)')
plt.title('Relação: Período Orbital vs Tamanho do Planeta')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()

# Verificar se há informação sobre método de descoberta
if 'koi_disposition' in df.columns:
    print(f"\n🔭 MÉTODOS DE DESCOBERTA:")
    print(df['koi_disposition'].value_counts())