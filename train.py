import pandas as pd
import numpy as np
import os
import cloudpickle

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, fbeta_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde un CSV y devuelve un DataFrame."""
    print(f"Cargando datos desde {path}...")
    return pd.read_csv(path)


def get_X_y(df: pd.DataFrame):
    """Preprocesa los datos y construye X, y para el modelado."""
    print("Preparando features y variable objetivo...")
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
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    # Definición de variable objetivo: calidad de sueño buena (≥7)
    df['target'] = (df['Quality of Sleep'] >= 7).astype(int)

    # Eliminar columna original de calidad para evitar data leakage
    df.drop(columns=['Quality of Sleep'], inplace=True)

    # Dividir en características y target
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"Datos preparados: {X.shape[0]} registros, {X.shape[1]} características")

    return X, y


def get_pipeline(X_sample, pos_ratio):
    """Crea el pipeline de preprocesamiento para XGBoost."""

    # Identificar tipos de características
    numeric_features = X_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline para características numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinar pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    # Modelo base de XGBoost - usamos scale_pos_weight en lugar de SMOTE
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        scale_pos_weight=pos_ratio,  # Usamos este parámetro en lugar de SMOTE
        random_state=42
    )

    # Pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])

    return pipeline


def find_threshold_max_precision(y_true, y_proba, min_recall=0.6):
    """Encuentra el umbral que maximiza la precisión sin que el recall caiga por debajo de min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Añadir el último umbral (1.0) si falta
    thresholds = np.append(thresholds, 1.0)

    # Filtrar umbrales con suficiente recall
    valid_idx = np.where(recall >= min_recall)[0]

    if len(valid_idx) == 0:
        print(f"⚠️  Ningún umbral mantiene recall ≥ {min_recall}. Se tomará el mejor posible.")
        best_idx = np.argmax(precision)
    else:
        best_idx = valid_idx[np.argmax(precision[valid_idx])]

    return thresholds[best_idx], precision[best_idx], recall[best_idx]



def cross_validate_threshold_precision(X, y, pipeline, cv_splits=10, min_recall=0.6):
    """Realiza validación cruzada para encontrar el umbral que maximiza precisión sin perder recall."""
    print(f"Realizando validación cruzada para maximizar precisión con recall mínimo de {min_recall}...")

    all_probas = []
    all_true = []

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"  Fold {fold + 1}/{cv_splits}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_pipeline = pipeline.fit(X_train, y_train)
        y_val_proba = fold_pipeline.predict_proba(X_val)[:, 1]

        all_probas.extend(y_val_proba)
        all_true.extend(y_val)

    all_probas = np.array(all_probas)
    all_true = np.array(all_true)

    # Obtener mejor threshold maximizando precisión sin perder recall
    optimal_threshold, best_precision, corresponding_recall = find_threshold_max_precision(
        all_true, all_probas, min_recall=min_recall
    )

    print(f"\n🔍 Umbral óptimo: {optimal_threshold:.4f} | Precisión: {best_precision:.4f} | Recall: {corresponding_recall:.4f}")

    optimal_preds = (all_probas >= optimal_threshold).astype(int)
    print("\nReporte de clasificación con umbral óptimo:")
    print(classification_report(all_true, optimal_preds))

    return optimal_threshold



def optimize_hyperparameters(X, y, pipeline, pos_ratio, cv_splits=10, beta=0.5):
    """Realiza búsqueda aleatoria de hiperparámetros para XGBoost."""
    print("Optimizando hiperparámetros para XGBoost...")

    # Definir espacio de búsqueda
    param_dist = {
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__scale_pos_weight': [pos_ratio, pos_ratio*0.8, pos_ratio*1.2],  # Variaciones del ratio
    }

    # Crear scorer de F-beta
    f_beta_scorer = make_scorer(fbeta_score, beta=beta)

    # Configurar CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=100,
        scoring=f_beta_scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Realizar búsqueda
    random_search.fit(X, y)

    # Mostrar resultados
    print("\nMejores hiperparámetros encontrados:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Mejor F{beta}-score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def save_model_and_threshold(model, threshold, path='best_model.pkl'):
    """Guarda el modelo entrenado y el threshold óptimo."""
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Crear paquete para guardar
    package = {
        'pipeline': model,
        'threshold': threshold
    }

    # Guardar modelo
    with open(path, 'wb') as f:
        cloudpickle.dump(package, f)

    print(f"Modelo y threshold guardados en '{path}'")


def evaluate_model_with_threshold(model, X_test, y_test, threshold, beta=0.5):
    """Evalúa el modelo con el threshold optimizado."""
    # Obtener probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplicar threshold optimizado
    y_pred_threshold = (y_proba >= threshold).astype(int)

    # Evaluar
    print("\nEvaluación con threshold optimizado:")
    print(classification_report(y_test, y_pred_threshold))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred_threshold))

    # Calcular F-beta score
    fbeta = fbeta_score(y_test, y_pred_threshold, beta=beta)
    print(f"F{beta}-score: {fbeta:.4f}")


def main():
    """Función principal del flujo de trabajo."""
    print("Iniciando proceso de modelado para calidad del sueño...")

    # 1. Cargar datos
    df = load_data('data/train.csv')

    # 2. Preprocesar datos
    X, y = get_X_y(df)

    # 3. División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"División train-test: {X_train.shape[0]} registros de entrenamiento, {X_test.shape[0]} de test")

    # 4. Calcular ratio de desbalance para scale_pos_weight
    pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Ratio de desbalance (neg:pos): {pos_ratio:.2f}")

    # 5. Obtener pipeline base
    base_pipeline = get_pipeline(X_train, pos_ratio)

    # 6. Optimizar hiperparámetros con validación cruzada
    best_model, best_params = optimize_hyperparameters(X_train, y_train, base_pipeline, pos_ratio, cv_splits=10, beta=0.5)

    # 7. Encontrar threshold óptimo
    optimal_threshold = cross_validate_threshold_precision(X_train, y_train, best_model, cv_splits=10, min_recall=0.6)

    # 8. Evaluar modelo final en test set
    evaluate_model_with_threshold(best_model, X_test, y_test, optimal_threshold, beta=0.5)

    # 9. Guardar modelo final y threshold
    save_model_and_threshold(best_model, optimal_threshold, 'model/pipeline.cloudpkl')

    print("\nProceso de modelado completado.")


if __name__ == "__main__":
    main()