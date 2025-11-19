import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
def grafico_distribucion(df, variable):
    plt.figure(figsize=(8, 6))

    media = df[variable].mean()
    mediana = df[variable].median()
    moda = float(mode(df[variable], keepdims=False).mode)

    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
    sns.boxplot(x=df[variable], ax=ax1, color="skyblue")
    ax1.set(xlabel='')
    ax1.set_title(f'Distribución de {variable}')
    ax1.grid(True)

    ax2 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
    sns.histplot(df[variable], kde=True, bins=30, color="steelblue", edgecolor="black", ax=ax2)

    ax2.axvline(media, color='orange', linestyle='--', label='Media')
    ax2.axvline(mediana, color='green', linestyle='--', label='Mediana')
    ax2.axvline(moda, color='red', linestyle='--', label='Moda')

    ax2.set_xlabel(variable)
    ax2.set_ylabel("Frecuencia")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
