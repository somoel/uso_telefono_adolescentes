import pandas as pd


def calcular_estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include='all')


def analizar_distribucion_por_genero(df: pd.DataFrame) -> pd.DataFrame:
    stats_genero = df.groupby('Gender')['Daily_Usage_Hours'].agg(['mean', 'std', 'min', 'max', 'count'])
    return stats_genero


def analizar_tiempo_redes_por_edad(df: pd.DataFrame) -> pd.DataFrame:
    stats_edad = df.groupby('Age')['Time_on_Social_Media'].mean().reset_index()
    return stats_edad


def analizar_autoestima_por_edad(df: pd.DataFrame) -> pd.DataFrame:
    stats_autoestima = df.groupby('Age')['Self_Esteem'].mean().reset_index()
    return stats_autoestima


def analizar_uso_vs_adiccion(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('Daily_Usage_Hours')['Addiction_Level'].mean().reset_index()
    return grouped


def analizar_edad_vs_depresion(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('Age')['Depression_Level'].mean().reset_index()
    return grouped


def analizar_edad_vs_comunicacion_familiar(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('Age')['Family_Communication'].mean().reset_index()
    return grouped


def analizar_proposito_uso_por_horas(df: pd.DataFrame, horas_uso: int = 5) -> pd.Series:
    filtered_df = df[df['Daily_Usage_Hours'] == horas_uso]
    purpose_counts = filtered_df['Phone_Usage_Purpose'].value_counts()
    return purpose_counts


def analizar_ejercicio_vs_autoestima(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('Exercise_Hours')['Self_Esteem'].mean().reset_index()
    return grouped

def analizar_ejercicio_vs_adiccion(df: pd.DataFrame) -> pd.DataFrame:

    grouped = df.groupby('Exercise_Hours')['Addiction_Level'].mean().reset_index()
    return grouped


def analizar_actividades_por_edad(df: pd.DataFrame) -> pd.DataFrame:

    grouped = df.groupby('Age')[['Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education']].mean().reset_index()
    return grouped


def analizar_redes_vs_rendimiento(df: pd.DataFrame) -> pd.DataFrame:

    grouped = df.groupby('Time_on_Social_Media')['Academic_Performance'].mean().reset_index()
    return grouped


def calcular_matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:

    correlation_matrix = df.corr(numeric_only=True)
    return correlation_matrix


def obtener_distribucion_adiccion(df: pd.DataFrame) -> pd.Series:
    return df['Addiction_Level'].value_counts().sort_index()