from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def crear_modelos() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    return models


def entrenar_modelo(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluar_modelo(model, X_test, y_test, nombre_modelo: str = "Modelo") -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print(f"\n🔍 Evaluación de {nombre_modelo}")
    print(f" Accuracy: {acc:.4f}")
    print(" Confusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }


def entrenar_evaluar_multiples_modelos(models: dict, X_train, y_train, X_test, y_test) -> dict:
    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f" Entrenando {name}...")
        print(f"{'=' * 50}")

        # Entrenar
        model_trained = entrenar_modelo(model, X_train, y_train)

        # Evaluar
        metrics = evaluar_modelo(model_trained, X_test, y_test, name)

        results[name] = {
            'model': model_trained,
            'metrics': metrics
        }

    return results


def comparar_modelos(results: dict) -> pd.DataFrame:
    accuracies = {name: result['metrics']['accuracy'] for name, result in results.items()}

    comparison_df = pd.DataFrame(
        list(accuracies.items()),
        columns=["Model", "Accuracy"]
    ).sort_values(by="Accuracy", ascending=False)

    print("\n" + "=" * 50)
    print(" RESUMEN COMPARATIVO DE MODELOS")
    print("=" * 50)
    print(comparison_df.to_string(index=False))
    print("=" * 50)

    return comparison_df


def obtener_mejor_modelo(results: dict) -> tuple:
    best_name = max(results, key=lambda x: results[x]['metrics']['accuracy'])
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['metrics']['accuracy']

    print(f"\n🏆 Mejor modelo: {best_name} con accuracy de {best_accuracy:.4f}")

    return best_name, best_model, best_accuracy


def pipeline_modelado_completo(X_train, y_train, X_test, y_test) -> dict:
    # 1. Crear modelos
    models = crear_modelos()

    # 2. Entrenar y evaluar
    results = entrenar_evaluar_multiples_modelos(models, X_train, y_train, X_test, y_test)

    # 3. Comparar
    comparison = comparar_modelos(results)

    # 4. Obtener mejor modelo
    best_name, best_model, best_accuracy = obtener_mejor_modelo(results)

    return {
        'results': results,
        'comparison': comparison,
        'best_model_name': best_name,
        'best_model': best_model,
        'best_accuracy': best_accuracy
    }
