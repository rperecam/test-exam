import pandas as pd
import os
import cloudpickle
import argparse
import sys


def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde un CSV y devuelve un DataFrame."""
    print(f"Cargando datos desde {path}...")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Preprocesa los datos de forma similar a get_X_y pero sin crear el target."""
    print("Preprocesando datos de entrada...")
    df = df.copy()

    # Procesamiento de presión arterial si existe
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['BP_sys'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['BP_dia'] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(columns=['Blood Pressure'], inplace=True)

        # Features adicionales derivados de BP
        df['BP_mean'] = (df['BP_sys'] + df['BP_dia']) / 2
        df['high_bp'] = ((df['BP_sys'] >= 130) | (df['BP_dia'] >= 80)).astype(int)

    # Manejo de valores faltantes para Sleep Disorder
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    # Si hay columna de calidad de sueño, la preservamos para comparación
    has_quality = False
    if 'Quality of Sleep' in df.columns:
        has_quality = True
        quality = df['Quality of Sleep'].copy()
        actual_target = (quality >= 7).astype(int)
        df.drop(columns=['Quality of Sleep'], inplace=True)

    print(f"Datos preprocesados: {df.shape[0]} registros, {df.shape[1]} características")

    if has_quality:
        return df, actual_target, quality
    else:
        return df, None, None


def load_model(model_path: str):
    """Carga el modelo y el threshold óptimo."""
    print(f"Cargando modelo desde {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            package = cloudpickle.load(f)

        pipeline = package['pipeline']
        threshold = package['threshold']
        print(f"Modelo cargado correctamente con threshold {threshold:.4f}")
        return pipeline, threshold
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None


def make_predictions(pipeline, X, threshold):
    """Realiza predicciones usando el modelo y el threshold optimizado."""
    print("Realizando predicciones...")

    # Obtener probabilidades
    probas = pipeline.predict_proba(X)[:, 1]

    # Aplicar threshold
    predictions = (probas >= threshold).astype(int)

    return predictions, probas


def save_predictions(df, predictions, probabilities, actual_target, quality, output_path):
    """Guarda las predicciones en un CSV junto con probabilidades y datos originales."""
    print(f"Guardando predicciones en {output_path}...")

    # Crear un DataFrame para guardar resultados
    results = df.copy()
    results['predicted_proba'] = probabilities
    results['predicted_good_sleep'] = predictions

    # Si tenemos los valores reales, incluirlos para comparación
    if actual_target is not None:
        results['actual_quality'] = quality
        results['actual_good_sleep'] = actual_target

    # Guardar resultados
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predicciones guardadas exitosamente en {output_path}")


def main():
    """Función principal del flujo de trabajo de inferencia."""

    # Valores por defecto
    input_path = "data/inference.csv"
    model_path = "model/pipeline.cloudpkl"
    output_path = "data/output_predictions.csv"

    # Intentar procesar argumentos si no estamos en un entorno IDE
    if len(sys.argv) > 1 and not any("--host=" in arg for arg in sys.argv):
        # Configurar parser de argumentos
        parser = argparse.ArgumentParser(description="Realizar inferencia con el modelo de calidad de sueño")
        parser.add_argument("--input", type=str, default=input_path, help="Ruta al archivo CSV de entrada")
        parser.add_argument("--model", type=str, default=model_path, help="Ruta al modelo guardado")
        parser.add_argument("--output", type=str, default=output_path, help="Ruta para guardar las predicciones")

        # Parsear argumentos
        args = parser.parse_args()
        input_path = args.input
        model_path = args.model
        output_path = args.output

    print("Iniciando proceso de inferencia para calidad del sueño...")

    # 1. Cargar datos
    df = load_data(input_path)

    # 2. Preprocesar datos
    X, actual_target, quality = preprocess_data(df)

    # 3. Cargar modelo y threshold
    model, threshold = load_model(model_path)

    if model is None:
        print("No se pudo cargar el modelo. Abortando.")
        return

    # 4. Realizar predicciones
    predictions, probabilities = make_predictions(model, X, threshold)

    # 5. Guardar predicciones
    save_predictions(df, predictions, probabilities, actual_target, quality, output_path)

    # 6. Mostrar resumen de predicciones
    print("\nResumen de predicciones:")
    print(f"Total de registros procesados: {len(predictions)}")
    print(
        f"Predicciones positivas (buena calidad de sueño): {predictions.sum()} ({predictions.sum() / len(predictions) * 100:.1f}%)")
    print(
        f"Predicciones negativas (mala calidad de sueño): {len(predictions) - predictions.sum()} ({(len(predictions) - predictions.sum()) / len(predictions) * 100:.1f}%)")

    # 7. Si hay datos reales, mostrar algunas métricas básicas
    if actual_target is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

        accuracy = accuracy_score(actual_target, predictions)
        precision = precision_score(actual_target, predictions)
        recall = recall_score(actual_target, predictions)
        f1 = f1_score(actual_target, predictions)
        f2 = fbeta_score(actual_target, predictions, beta=2.0)

        print("\nMétricas de evaluación:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"F2-score: {f2:.4f}")

    print("\nProceso de inferencia completado.")


if __name__ == "__main__":
    main()