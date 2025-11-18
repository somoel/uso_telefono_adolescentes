import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Matriz de dispersión para una columna objetivo
def plot_matriz_dispersion(df, target_col="Addiction_Level", excluir_cols=None, n_cols=3):
    if excluir_cols is None:
        excluir_cols = [target_col, "AddiccionBinaria"]

    variables_numericas = df.select_dtypes(include="number").columns
    variables_numericas = [col for col in variables_numericas if col not in excluir_cols]

    n_vars = len(variables_numericas)
    if n_vars == 0:
        return

    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    unique_levels = sorted(df[target_col].unique())
    unique_levels_int = sorted({int(round(x)) for x in unique_levels})

    for idx, col in enumerate(variables_numericas):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].scatter(df[col], df[target_col], alpha=0.5)
        axes[r][c].set_title(col)
        axes[r][c].set_xlabel(col)
        axes[r][c].set_ylabel(target_col)
        axes[r][c].set_yticks(unique_levels_int)

    for idx in range(n_vars, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle(f"Matriz de gráficos de dispersión ({target_col} en eje Y)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#matrix de confusion
def entrenar_y_evaluar_modelo_adiccion(df: pd.DataFrame):

    # 1. Crear variable binaria de adicción (1 = alto, 0 = bajo)
    # Se considera 'alto' si el nivel es mayor a 5.
    df["AddiccionBinaria"] = (df["Addiction_Level"] > 5).astype(int)

    # 2. Variables predictoras y objetivo
    # 'Daily_Usage_Hours' (Horas de uso) es la variable predictora (X)
    X = df[["Daily_Usage_Hours"]]
    # 'AddiccionBinaria' es la etiqueta u objetivo (y)
    y = df["AddiccionBinaria"]

    # 3. Separar datos entrenamiento/prueba (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42  # random_state para reproducibilidad
    )

    # 4. Modelo: Inicializar y entrenar la Regresión Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Modelo de Regresión Logística entrenado.")

    # 5. Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # 6. Resultados: Calcular y mostrar la Matriz de Confusión
    matriz_conf = confusion_matrix(y_test, y_pred)
    print("\n RESULTADOS")
    print("MATRIZ DE CONFUSIÓN (Real vs Predicción):")
    print(matriz_conf)

    # Devuelve el modelo y la matriz para usarlos fuera de la función
    return model, matriz_conf

