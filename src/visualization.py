# src/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import os

# Caminhos para salvar gráficos
BASE_PATH = os.path.join('..', 'graficos')  # Pasta para salvar gráficos
os.makedirs(BASE_PATH, exist_ok=True)  # Criar pasta 'graficos' se não existir

def plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(BASE_PATH, 'matriz_confusao.png')):
    """
    Gera e salva a matriz de confusão.
    Args:
        y_true: Rótulos verdadeiros (array numpy).
        y_pred: Rótulos previstos (array numpy).
        save_path: Caminho para salvar o gráfico.
    Returns:
        fig: Objeto matplotlib para uso opcional.
    """
    cm = np.array([[0, 0], [0, 0]])  # Placeholder
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Erro ao calcular matriz de confusão: {e}")
        return None

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Habitável', 'Habitável'],
                yticklabels=['Não Habitável', 'Habitável'])
    plt.title('Matriz de Confusão: Previsão de Habitabilidade')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 Gráfico salvo em: {save_path}")
    # plt.show()  # Comentar para evitar warning em terminais sem GUI
    return plt.gcf()

def plot_feature_importance(features, importances, save_path=os.path.join(BASE_PATH, 'importancia_features.png')):
    """
    Gera e salva o gráfico de importância das features.
    Args:
        features: Lista de nomes das features.
        importances: Array de importâncias (ex.: model.feature_importances_).
        save_path: Caminho para salvar o gráfico.
    Returns:
        fig: Objeto matplotlib para uso opcional.
    """
    df = pd.DataFrame({'Feature': features, 'Importância': importances}).sort_values('Importância', ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importância', y='Feature', data=df, color='skyblue')
    plt.title('Importância das Features no Modelo')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 Gráfico salvo em: {save_path}")
    # plt.show()  # Comentar para evitar warning
    return plt.gcf()

def plot_label_distribution(y, save_path=os.path.join(BASE_PATH, 'distribuicao_labels.png')):
    """
    Gera e salva o gráfico de distribuição de labels.
    Args:
        y: Rótulos (array numpy, 0 para Não Habitável, 1 para Habitável).
        save_path: Caminho para salvar o gráfico.
    Returns:
        fig: Objeto matplotlib para uso opcional.
    """
    counts = np.bincount(y)
    labels = ['Não Habitável', 'Habitável']
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Distribuição de Planetas: Habitáveis vs. Não Habitáveis')
    plt.ylabel('Número de Planetas')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 Gráfico salvo em: {save_path}")
    # plt.show()  # Comentar para evitar warning
    return plt.gcf()

def plot_confusion_matrix_plotly(y_true, y_pred):
    """
    Gera matriz de confusão interativa para Streamlit com Plotly.
    Args:
        y_true: Rótulos verdadeiros (array numpy).
        y_pred: Rótulos previstos (array numpy).
    Returns:
        fig: Objeto plotly para uso no Streamlit.
    """
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Não Habitável', 'Habitável'],
            y=['Não Habitável', 'Habitável'],
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(title='Matriz de Confusão: Previsão de Habitabilidade',
                         xaxis_title='Previsto', yaxis_title='Verdadeiro')
        return fig
    except Exception as e:
        print(f"Erro ao criar matriz de confusão com Plotly: {e}")
        return None

def plot_feature_importance_plotly(features, importances):
    """
    Gera gráfico de importância das features interativo para Streamlit com Plotly.
    Args:
        features: Lista de nomes das features.
        importances: Array de importâncias.
    Returns:
        fig: Objeto plotly para uso no Streamlit.
    """
    df = pd.DataFrame({'Feature': features, 'Importância': importances}).sort_values('Importância', ascending=False)
    fig = px.bar(df, x='Importância', y='Feature', orientation='h', title='Importância das Features no Modelo',
                 color_discrete_sequence=['skyblue'])
    fig.update_layout(xaxis_title='Importância', yaxis_title='')
    return fig

def plot_label_distribution_plotly(y):
    """
    Gera gráfico de distribuição de labels interativo para Streamlit com Plotly.
    Args:
        y: Rótulos (array numpy).
    Returns:
        fig: Objeto plotly para uso no Streamlit.
    """
    counts = np.bincount(y)
    labels = ['Não Habitável', 'Habitável']
    df = pd.DataFrame({'Label': labels, 'Contagem': counts})
    fig = px.bar(df, x='Label', y='Contagem', title='Distribuição de Planetas: Habitáveis vs. Não Habitáveis',
                 color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig.update_layout(yaxis_title='Número de Planetas')
    return fig

if __name__ == "__main__":
    # Exemplo de uso standalone com dados fictícios (substitua pelos dados reais)
    y_true = np.array([0] * 548 + [1] * 2)  # Do treinamento anterior
    y_pred = np.array([0] * 548 + [0] * 2)  # Do treinamento anterior
    features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']
    importances = [0.090789, 0.290536, 0.321802, 0.296872]  # Do treinamento anterior
    y = np.array([0] * 2736 + [1] * 10)  # Distribuição de labels

    # Gerar gráficos com matplotlib/seaborn
    plot_confusion_matrix(y_true, y_pred)
    plot_feature_importance(features, importances)
    plot_label_distribution(y)

    # Gerar gráficos com plotly (para teste)
    fig_cm = plot_confusion_matrix_plotly(y_true, y_pred)
    fig_fi = plot_feature_importance_plotly(features, importances)
    fig_ld = plot_label_distribution_plotly(y)