import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def grafico_uso_por_genero(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Gender', y='Daily_Usage_Hours', hue='Gender',
                palette='pastel', errorbar='sd', legend=False)
    plt.title('Horas promedio de uso diario por género')
    plt.xlabel('Género')
    plt.ylabel('Horas de uso diario')
    plt.show()


def grafico_distribucion_uso_diario(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Daily_Usage_Hours'], kde=True, bins=20, color='skyblue')
    plt.title('Distribución del tiempo diario frente a la pantalla')
    plt.xlabel('Horas de uso diario')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()


def grafico_tiempo_redes_por_edad(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x='Age',
        y='Time_on_Social_Media',
        hue='Age',
        errorbar='sd',
        palette='muted',
        legend=False
    )
    plt.title('Tiempo promedio en redes sociales por edad')
    plt.xlabel('Edad')
    plt.ylabel('Tiempo en redes sociales (horas)')
    plt.show()


def grafico_autoestima_por_edad(df: pd.DataFrame) -> None:
    age_self_esteem = df.groupby('Age')['Self_Esteem'].mean()
    plt.figure(figsize=(8, 8))
    age_self_esteem.plot.pie(
        autopct='%1.1f%%',
        startangle=140,
        cmap='Set3'
    )
    plt.title('Autoestima promedio por edad')
    plt.ylabel('')
    plt.show()


def grafico_uso_vs_adiccion(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Daily_Usage_Hours')['Addiction_Level'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Daily_Usage_Hours', y='Addiction_Level', marker='o', color='blue')
    plt.title('Horas diarias vs nivel de adicción')
    plt.xlabel('Horas de uso diario')
    plt.ylabel('Nivel promedio de adicción')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_edad_vs_depresion(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Age')['Depression_Level'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Age', y='Depression_Level', marker='o', color='green')
    plt.title('Gráfico lineal: Edad vs. Nivel de depresión')
    plt.xlabel('Edad')
    plt.ylabel('Nivel promedio de depresión')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_edad_vs_comunicacion(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Age')['Family_Communication'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Age', y='Family_Communication', marker='o', color='purple')
    plt.title('Gráfico lineal: Edad vs. Comunicación familiar')
    plt.xlabel('Edad')
    plt.ylabel('Puntuación promedio de comunicación familiar')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_proposito_uso(df: pd.DataFrame, horas_uso: int = 5) -> None:
    filtered_df = df[df['Daily_Usage_Hours'] == horas_uso]
    purpose_counts = filtered_df['Phone_Usage_Purpose'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(purpose_counts, labels=purpose_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribución del propósito del uso del teléfono para {horas_uso} horas de uso diario')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def grafico_ejercicio_vs_autoestima(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Exercise_Hours')['Self_Esteem'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Exercise_Hours', y='Self_Esteem', marker='o', color='teal')
    plt.title('Gráfico lineal: Horas de ejercicio vs. Autoestima')
    plt.xlabel('Horas de ejercicio')
    plt.ylabel('Puntuación media de autoestima')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_ejercicio_vs_adiccion(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Exercise_Hours')['Addiction_Level'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Exercise_Hours', y='Addiction_Level', marker='o', color='darkblue')
    plt.title('Gráfico lineal: Horas de ejercicio vs. Nivel de adicción')
    plt.xlabel('Horas de ejercicio')
    plt.ylabel('Nivel promedio de adicción')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_actividades_por_edad(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Age')[['Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education']].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=grouped_df, x='Age', y='Time_on_Social_Media', marker='o', label='Redes Sociales')
    sns.lineplot(data=grouped_df, x='Age', y='Time_on_Gaming', marker='o', label='Gaming')
    sns.lineplot(data=grouped_df, x='Age', y='Time_on_Education', marker='o', label='Educación')

    plt.title('Gráfico lineal: Edad vs. Tiempo dedicado a las actividades')
    plt.xlabel('Edad')
    plt.ylabel('Tiempo promedio (horas)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_redes_vs_rendimiento(df: pd.DataFrame) -> None:
    grouped_df = df.groupby('Time_on_Social_Media')['Academic_Performance'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_df, x='Time_on_Social_Media', y='Academic_Performance', marker='o', color='darkgreen')
    plt.title('Gráfico lineal: Tiempo en redes sociales vs. Rendimiento académico')
    plt.xlabel('Tiempo en redes sociales (horas)')
    plt.ylabel('Rendimiento académico promedio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def grafico_matriz_correlacion(df: pd.DataFrame) -> None:
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de correlación del conjunto de datos sobre la adicción al teléfono en adolescentes")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def grafico_histogramas(df: pd.DataFrame) -> None:
    df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def grafico_matrices_confusion(models: dict, X_test, y_test) -> None:
    from sklearn.metrics import confusion_matrix

    num_models = len(models)
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(7 * num_models, 6))

    if num_models == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicción')
        axes[idx].set_ylabel('Real')

    plt.tight_layout()
    plt.show()


def grafico_comparacion_modelos(results: dict) -> None:
    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy",
                                                                                                ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Accuracy",
        y="Model",
        data=results_df,
        hue="Model",
        dodge=False,
        palette="coolwarm",
        legend=False
    )
    plt.title('Comparación de precisión de modelos')
    plt.xlabel('Precisión')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()
