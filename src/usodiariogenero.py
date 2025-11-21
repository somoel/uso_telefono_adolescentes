import matplotlib.pyplot as plt
import seaborn as sns



def graficar_uso_diario_por_genero(df):
    """Grafica las horas promedio de uso diario por género con paleta pastel."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x='Gender',
        y='Daily_Usage_Hours',
        hue='Gender',
        palette='pastel',
        errorbar='sd',
        legend=False
    )
    plt.title('Horas promedio de uso diario por género')
    plt.xlabel('Género')
    plt.ylabel('Uso diario (horas)')
    plt.tight_layout()
    plt.show()