import matplotlib.pyplot as plt
import seaborn as sns

def graficar_matriz_correlacion(df):
    """Genera la matriz de correlación del conjunto de datos."""
    matriz_correlacion = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de correlación del conjunto de datos")
    plt.tight_layout()
    plt.show()

