import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def codificar_variables_categoricas(df: pd.DataFrame) -> tuple:
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    print(f"Variables codificadas: {list(label_encoders.keys())}")

    return df_encoded, label_encoders


def estandarizar_features(X: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Features estandarizadas correctamente")

    return X_scaled, scaler


def separar_features_target(df: pd.DataFrame, columna_target: str = 'Addiction_Level') -> tuple:
    X = df.drop(columna_target, axis=1)
    y = df[columna_target].astype(int)

    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")

    return X, y


def aplicar_smote(X_train, y_train):
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("Distribución antes de SMOTE:")
    print(y_train.value_counts())

    print("\nDistribución después de SMOTE:")
    print(y_train_res.value_counts())

    return X_train_res, y_train_res


def preprocesar_datos_completo(df: pd.DataFrame, columna_target: str = 'Addiction_Level',
                               aplicar_balance: bool = True) -> dict:
    from sklearn.model_selection import train_test_split

    # 1. Codificar variables categóricas
    df_encoded, label_encoders = codificar_variables_categoricas(df)

    # 2. Separar features y target
    X, y = separar_features_target(df_encoded, columna_target)

    # 3. Estandarizar features
    X_scaled, scaler = estandarizar_features(X)

    # 4. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDatos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")

    # 5. Aplicar SMOTE si es necesario
    if aplicar_balance:
        X_train, y_train = aplicar_smote(X_train, y_train)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': df.drop(columna_target, axis=1).columns.tolist()
    }
