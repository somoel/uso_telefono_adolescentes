
import pandas as pd
import numpy as np


def cargar_datos(ruta_csv: str) -> pd.DataFrame:

    df = pd.read_csv(ruta_csv)
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:

    df_limpio = df.copy()

    # Eliminar columnas innecesarias
    columnas_eliminar = ['ID', 'Name']
    df_limpio.drop(columns=columnas_eliminar, inplace=True, errors='ignore')

    # Información sobre duplicados y nulos
    print(f"Valores duplicados encontrados: {df_limpio.duplicated().sum()}")
    print(f"\nValores nulos por columna:\n{df_limpio.isna().sum()}")

    return df_limpio


def crear_categorias_adiccion(df: pd.DataFrame) -> pd.DataFrame:

    df_categorizado = df.copy()

    df_categorizado['Addiction_Level_Category'] = pd.cut(
        df_categorizado['Addiction_Level'],
        bins=[-np.inf, 3.5, 7.5, np.inf],
        labels=[0, 1, 2]
    )

    df_categorizado.drop(columns=['Addiction_Level'], inplace=True)
    df_categorizado.rename(columns={'Addiction_Level_Category': 'Addiction_Level'}, inplace=True)

    return df_categorizado


def filtrar_clases_validas(df: pd.DataFrame, columna_target: str = 'Addiction_Level',
                           min_muestras: int = 2) -> pd.DataFrame:

    df_filtrado = df.copy()

    class_counts = df_filtrado[columna_target].value_counts()
    valid_classes = class_counts[class_counts >= min_muestras].index
    df_filtrado = df_filtrado[df_filtrado[columna_target].isin(valid_classes)]

    print(f"\nClases válidas después del filtrado: {valid_classes.tolist()}")
    print(f"Total de muestras después del filtrado: {len(df_filtrado)}")

    return df_filtrado


def obtener_informacion_dataset(df: pd.DataFrame) -> dict:
    info = {
        'num_filas': len(df),
        'num_columnas': len(df.columns),
        'columnas': df.columns.tolist(),
        'tipos_datos': df.dtypes.to_dict(),
        'valores_nulos': df.isna().sum().to_dict(),
        'valores_duplicados': df.duplicated().sum()
    }

    return info
