import matplotlib.pyplot as plt
import seaborn as sns

def graficar_distribucion_tiempo_pantalla(df):
    """Grafica la distribución del tiempo diario frente a la pantalla con estilo pastel."""
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x='Daily_Usage_Hours',
        kde=True,
        bins=20,
        color=sns.color_palette('pastel')[0]
    )
    plt.title('Distribución del tiempo diario frente a la pantalla')
    plt.xlabel('Horas de uso diario')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()