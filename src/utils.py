import pandas as pd
import numpy as np


def imprimir_separador(titulo: str = "", longitud: int = 60, caracter: str = "=") -> None:
    if titulo:
        print(f"\n{caracter * longitud}")
        print(f"{titulo.center(longitud)}")
        print(f"{caracter * longitud}")
    else:
        print(f"\n{caracter * longitud}")


def guardar_resultados_csv(df: pd.DataFrame, nombre_archivo: str, carpeta: str = "../data/") -> None:

    ruta_completa = f"{carpeta}{nombre_archivo}.csv"
    df.to_csv(ruta_completa, index=False)
    print(f" Resultados guardados en: {ruta_completa}")


def obtener_resumen_datos(df: pd.DataFrame) -> dict:
    resumen = {
        'dimensiones': df.shape,
        'total_filas': len(df),
        'total_columnas': len(df.columns),
        'columnas': df.columns.tolist(),
        'tipos_datos': df.dtypes.to_dict(),
        'valores_nulos': df.isna().sum().to_dict(),
        'valores_duplicados': df.duplicated().sum(),
        'memoria_uso_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
    }

    return resumen


def imprimir_resumen_datos(df: pd.DataFrame) -> None:
    resumen = obtener_resumen_datos(df)

    imprimir_separador("RESUMEN DEL DATASET")
    print(f" Dimensiones: {resumen['dimensiones']}")
    print(f" Total de filas: {resumen['total_filas']}")
    print(f" Total de columnas: {resumen['total_columnas']}")
    print(f" Uso de memoria: {resumen['memoria_uso_mb']:.2f} MB")
    print(f" Valores duplicados: {resumen['valores_duplicados']}")

    print(f"\n Columnas:")
    for col in resumen['columnas']:
        print(f"   - {col}")

    print(f"\n Valores nulos:")
    for col, nulos in resumen['valores_nulos'].items():
        if nulos > 0:
            print(f"   - {col}: {nulos}")

    if sum(resumen['valores_nulos'].values()) == 0:
        print("   No hay valores nulos")

    imprimir_separador()


def exportar_metricas_modelo(metricas: dict, nombre_archivo: str, carpeta: str = "../data/") -> None:
    # Convertir report a DataFrame
    if 'classification_report' in metricas:
        report_df = pd.DataFrame(metricas['classification_report']).transpose()
        ruta_completa = f"{carpeta}{nombre_archivo}_report.csv"
        report_df.to_csv(ruta_completa)
        print(f" Métricas guardadas en: {ruta_completa}")

    # Guardar confusion matrix
    if 'confusion_matrix' in metricas:
        cm_df = pd.DataFrame(metricas['confusion_matrix'])
        ruta_cm = f"{carpeta}{nombre_archivo}_confusion_matrix.csv"
        cm_df.to_csv(ruta_cm, index=False)
        print(f" Matriz de confusión guardada en: {ruta_cm}")


def calcular_metricas_adicionales(y_true, y_pred) -> dict:
    """
    Calcula métricas adicionales de evaluación.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Diccionario con métricas adicionales
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    metricas = {
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    return metricas
