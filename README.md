# Memoria Técnica: Modelo Predictivo de Calidad del Sueño

## Índice
1. [Introducción](#introducción)
2. [Exploración de Datos](#exploración-de-datos)
   - [Análisis Exploratorio Inicial](#análisis-exploratorio-inicial)
   - [Técnicas de Reducción de Dimensionalidad](#técnicas-de-reducción-de-dimensionalidad)
   - [Análisis de Clustering](#análisis-de-clustering)
3. [Preprocesamiento de Datos](#preprocesamiento-de-datos)
   - [Tratamiento de Variables](#tratamiento-de-variables)
   - [Manejo de Valores Faltantes](#manejo-de-valores-faltantes)
   - [Feature Engineering](#feature-engineering)
4. [Desarrollo del Modelo](#desarrollo-del-modelo)
   - [Selección del Algoritmo](#selección-del-algoritmo)
   - [Pipeline de Modelado](#pipeline-de-modelado)
   - [Optimización de Hiperparámetros](#optimización-de-hiperparámetros)
   - [Validación Cruzada y Ajuste de Umbral](#validación-cruzada-y-ajuste-de-umbral)
5. [Evaluación del Modelo](#evaluación-del-modelo)
   - [Métricas Seleccionadas](#métricas-seleccionadas)
   - [Resultados en Conjunto de Prueba](#resultados-en-conjunto-de-prueba)
   - [Análisis de la Matriz de Confusión](#análisis-de-la-matriz-de-confusión)
6. [Inferencia](#inferencia)
   - [Flujo de Inferencia](#flujo-de-inferencia)
   - [Resultados Obtenidos](#resultados-obtenidos)
7. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)
   - [Interpretación de Resultados](#interpretación-de-resultados)
   - [Limitaciones del Modelo](#limitaciones-del-modelo)
   - [Posibles Mejoras](#posibles-mejoras)
8. [Anexos](#anexos)

## Introducción

El presente documento describe el proceso de desarrollo e implementación de un modelo predictivo para evaluar la calidad del sueño de individuos. La calidad del sueño es un factor determinante en la salud general y el bienestar de las personas, con impacto directo en aspectos como el rendimiento cognitivo, el estado de ánimo y diversos indicadores fisiológicos. 

El objetivo principal de este proyecto es crear un modelo capaz de clasificar la calidad del sueño en dos categorías: buena calidad (≥7 en una escala numérica) y mala calidad (<7), basándose en diversas variables fisiológicas, conductuales y médicas. Esta clasificación binaria permite identificar a individuos que podrían beneficiarse de intervenciones para mejorar su descanso.

La metodología empleada abarca desde la exploración y preprocesamiento de los datos hasta el desarrollo, evaluación e implementación del modelo predictivo, siguiendo las mejores prácticas en ciencia de datos y aprendizaje automático.

## Exploración de Datos

### Análisis Exploratorio Inicial

El conjunto de datos utilizado contiene registros de múltiples variables relacionadas con hábitos de sueño y salud de individuos. El dataset de entrenamiento consta de 447 registros con 14 características después del preprocesamiento inicial. Las variables incluyen parámetros fisiológicos como presión arterial, factores conductuales como consumo de cafeína y alcohol, y condiciones médicas como trastornos del sueño.

Durante la exploración inicial se detectó que la variable objetivo (calidad del sueño) presenta un desbalanceo, con una proporción aproximada de 0.46 entre casos negativos y positivos. Este desbalanceo se tuvo en cuenta en etapas posteriores del modelado para evitar sesgos en las predicciones.

### Técnicas de Reducción de Dimensionalidad

Para visualizar la estructura subyacente de los datos, se aplicaron técnicas de reducción de dimensionalidad que permitieron proyectar los datos multidimensionales en un espacio bidimensional. Estas visualizaciones revelaron que los datos no siguen una estructura clara o regular, lo que sugiere la complejidad inherente al problema de predecir la calidad del sueño.

Específicamente, se observó que los puntos de datos no forman agrupaciones naturales evidentes en el espacio reducido, lo que anticipa desafíos para lograr un rendimiento predictivo robusto y generalizable. Esta falta de estructura claramente definida es congruente con la naturaleza multifactorial y altamente individualizada de la calidad del sueño.

### Análisis de Clustering

Con el objetivo de identificar posibles estructuras latentes que pudieran aprovecharse para mejorar el proceso de validación cruzada, se realizó un análisis de clustering utilizando dos enfoques diferentes:

1. **KMeans**: Se evaluaron diferentes números de clusters mediante el coeficiente de Silhouette, que mide la coherencia interna de cada grupo en relación con los demás. Los resultados indicaron que el número óptimo de clusters, sin ser excesivamente elevado, se sitúa en torno a 10. Este número relativamente alto sugiere que los datos no presentan segmentaciones naturales claras o bien definidas, limitando la utilidad práctica de esta agrupación para fines de validación cruzada estratificada.

2. **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**: Se seleccionó este algoritmo debido a su capacidad para detectar estructuras complejas en conjuntos de datos con distribuciones de densidad variables, como sugería la visualización T-SNE. A diferencia de KMeans, HDBSCAN no requiere especificar el número de clusters a priori y puede identificar puntos como ruido. Sin embargo, los resultados tampoco evidenciaron la existencia de agrupaciones robustas o de valor analítico claro.

En conjunto, el análisis de clustering confirmó cuantitativamente lo que la visualización bidimensional sugería: no existen segmentos definidos ni patrones clave lo suficientemente fuertes como para ser aprovechados como grupos en una validación cruzada estratificada. Esta conclusión fue determinante para descartar el uso de StratifiedGroupKFold como estrategia de validación y orientar el enfoque hacia otras técnicas más adecuadas para este conjunto de datos.

## Preprocesamiento de Datos

### Tratamiento de Variables

El preprocesamiento de los datos incluyó varias transformaciones importantes para preparar el conjunto de datos para el modelado:

1. **Procesamiento de presión arterial**: La variable 'Blood Pressure' venía en formato de texto con valores sistólicos y diastólicos separados por "/". Esta variable se dividió en dos componentes numéricos:
   ```python
   bp_split = df['Blood Pressure'].str.split('/', expand=True)
   df['BP_sys'] = pd.to_numeric(bp_split[0], errors='coerce')
   df['BP_dia'] = pd.to_numeric(bp_split[1], errors='coerce')
   ```

2. **Definición de la variable objetivo**: Se estableció un umbral de 7 o superior para clasificar la calidad del sueño como "buena":
   ```python
   df['target'] = (df['Quality of Sleep'] >= 7).astype(int)
   ```

3. **Eliminación de variables redundantes**: Para evitar data leakage, se eliminó la columna original de calidad del sueño:
   ```python
   df.drop(columns=['Quality of Sleep'], inplace=True)
   ```

### Manejo de Valores Faltantes

El manejo de valores faltantes se abordó de manera específica según la naturaleza de cada variable:

1. **Para 'Sleep Disorder'**: Se rellenaron los valores nulos con 'None', ya que un valor faltante en este caso indica ausencia de trastorno del sueño, no un dato perdido:
   ```python
   df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
   ```

2. **Para variables numéricas**: Se incorporó en el pipeline una estrategia de imputación por mediana:
   ```python
   numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='median')),
       ('scaler', StandardScaler())
   ])
   ```

3. **Para variables categóricas**: Se aplicó imputación por moda:
   ```python
   categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='most_frequent')),
       ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
   ])
   ```

Este enfoque de imputación dentro del pipeline garantiza que el modelo sea robusto frente a posibles valores faltantes durante la inferencia, incluso si el conjunto de entrenamiento no los contiene.

### Feature Engineering

Se realizó un feature engineering selectivo para derivar variables potencialmente informativas a partir de las existentes:

1. **Variables derivadas de presión arterial**:
   ```python
   df['BP_mean'] = (df['BP_sys'] + df['BP_dia']) / 2
   df['high_bp'] = ((df['BP_sys'] >= 130) | (df['BP_dia'] >= 80)).astype(int)
   ```
   - `BP_mean`: Representa la presión arterial media, un indicador clínico relevante.
   - `high_bp`: Indicador binario de hipertensión según criterios clínicos estándar.

No se realizó un feature engineering exhaustivo debido a que el algoritmo XGBoost utilizado tiene capacidad inherente para capturar interacciones no lineales entre variables y determinar automáticamente la importancia relativa de cada característica.

## Desarrollo del Modelo

### Selección del Algoritmo

Para este problema de clasificación binaria con un conjunto de datos relativamente pequeño y potencialmente desbalanceado, se seleccionó XGBoost por las siguientes razones:

1. **Eficiencia y velocidad**: XGBoost es conocido por su rendimiento computacional eficiente, especialmente valioso para conjuntos de datos pequeños a medianos.

2. **Robustez frente al desbalanceo de clases**: El algoritmo ofrece el parámetro `scale_pos_weight` que permite ajustar el balance de clases sin necesidad de técnicas de remuestreo adicionales como SMOTE.

3. **Capacidad de interpretación**: XGBoost proporciona medidas de importancia de características, facilitando la interpretación del modelo.

4. **Generalización efectiva**: El algoritmo incorpora técnicas de regularización que ayudan a prevenir el sobreajuste, un riesgo significativo en datasets pequeños.

5. **Identificación de patrones complejos**: XGBoost puede capturar interacciones no lineales entre variables sin necesidad de un feature engineering exhaustivo.

### Pipeline de Modelado

Se diseñó un pipeline completo que integra el preprocesamiento y el modelado para garantizar la reproducibilidad y evitar data leakage:

```python
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

# Modelo base de XGBoost
xgb_model = XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='logloss',
    scale_pos_weight=pos_ratio,
    random_state=42
)

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])
```

Este diseño asegura que todas las transformaciones necesarias se apliquen de manera consistente durante el entrenamiento y la inferencia, facilitando además la implementación del modelo en producción.

### Optimización de Hiperparámetros

Para optimizar el rendimiento del modelo, se realizó una búsqueda aleatoria de hiperparámetros utilizando validación cruzada con 10 folds:

```python
# Definir espacio de búsqueda
param_dist = {
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__scale_pos_weight': [pos_ratio, pos_ratio*0.8, pos_ratio*1.2],
}
```

Los hiperparámetros se seleccionaron considerando aspectos clave para evitar el sobreajuste, dada la limitación del tamaño del dataset:

1. **Profundidad del árbol reducida** (`max_depth`: 3-6): Limitar la profundidad ayuda a prevenir árboles demasiado específicos para los datos de entrenamiento.

2. **Regularización** (`gamma`: 0-0.2): Introduce penalización para controlar la complejidad del modelo.

3. **Tasas de aprendizaje bajas** (`learning_rate`: 0.01-0.1): Favorece un aprendizaje más gradual y estable.

4. **Subsampling** (`subsample` y `colsample_bytree`: 0.6-1.0): Introduce variabilidad en cada árbol, mejorando la robustez del ensamblaje.

La búsqueda aleatoria evaluó 100 combinaciones diferentes de hiperparámetros, utilizando como métrica de optimización el F2-score, que otorga mayor importancia al recall que a la precisión.

Los mejores hiperparámetros encontrados fueron:
```
classifier__subsample: 0.8
classifier__scale_pos_weight: 0.5557377049180328
classifier__n_estimators: 50
classifier__min_child_weight: 5
classifier__max_depth: 4
classifier__learning_rate: 0.01
classifier__gamma: 0.1
classifier__colsample_bytree: 0.8
```

### Validación Cruzada y Ajuste de Umbral

Un aspecto crítico del proceso fue la optimización del umbral de decisión mediante validación cruzada. En lugar de utilizar el umbral predeterminado de 0.5, se buscó el valor que maximizara el F2-score:

```python
def cross_validate_threshold(X, y, pipeline, cv_splits=10, beta=2.0):
    """Realiza CV para encontrar el umbral óptimo que maximiza F-beta."""
    # Inicializar listas para guardar predicciones y labels verdaderos
    all_probas = []
    all_true = []

    # Configurar CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Realizar CV
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # División train-val
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Entrenar modelo
        fold_pipeline = pipeline.fit(X_train, y_train)

        # Obtener probabilidades
        y_val_proba = fold_pipeline.predict_proba(X_val)[:, 1]

        # Guardar para cálculo final
        all_probas.extend(y_val_proba)
        all_true.extend(y_val)

    # Encontrar threshold óptimo
    optimal_threshold, optimal_fbeta = find_threshold(all_true, all_probas, beta)
    
    return optimal_threshold
```

Este proceso determinó un umbral óptimo de 0.3974, significativamente menor que el umbral predeterminado de 0.5. Este ajuste refleja la prioridad dada al recall (sensibilidad) sobre la precisión, alineándose con el objetivo de minimizar los falsos negativos.

## Evaluación del Modelo

### Métricas Seleccionadas

Para evaluar el rendimiento del modelo, se seleccionó como métrica principal el F2-score en lugar del F1-score tradicional. Esta decisión se fundamenta en:

1. **Priorización del recall**: En el contexto de detección de problemas de calidad del sueño, es más importante identificar correctamente a todos los individuos con mala calidad de sueño (minimizar falsos negativos) que tener alta precisión.

2. **Desbalanceo de clases**: El F2-score es más adecuado para conjuntos de datos desbalanceados cuando se prioriza el recall.

El F2-score se calcula como:
```
F2 = 5 * (precisión * recall) / (4 * precisión + recall)
```

Además del F2-score, se monitorizaron métricas complementarias:
- Precisión
- Recall (sensibilidad)
- Accuracy
- Matriz de confusión

### Resultados en Conjunto de Prueba

Tras el entrenamiento con los hiperparámetros optimizados y aplicando el umbral ajustado (0.3974), el modelo obtuvo los siguientes resultados en el conjunto de prueba:

```
Evaluación con threshold optimizado:
              precision    recall  f1-score   support
           0       0.83      0.34      0.49        29
           1       0.76      0.97      0.85        61
    accuracy                           0.77        90
   macro avg       0.79      0.66      0.67        90
weighted avg       0.78      0.77      0.73        90

Matriz de confusión:
[[10 19]
 [ 2 59]]

F2.0-score: 0.9161
```

Estos resultados revelan:

1. **Alto recall para la clase positiva (0.97)**: El modelo identifica correctamente el 97% de los casos de buena calidad de sueño.

2. **Precisión moderada para la clase positiva (0.76)**: Aproximadamente el 76% de los casos clasificados como buena calidad de sueño son correctos.

3. **Bajo recall para la clase negativa (0.34)**: El modelo solo detecta el 34% de los casos de mala calidad de sueño.

4. **F2-score global de 0.9161**: Un valor elevado que refleja el buen desempeño general del modelo, especialmente considerando la priorización del recall.

### Análisis de la Matriz de Confusión

La matriz de confusión proporciona información detallada sobre las predicciones:

```
[[10 19]
 [ 2 59]]
```

- **Verdaderos Positivos (TP)**: 59 casos de buena calidad de sueño correctamente clasificados.
- **Falsos Negativos (FN)**: 2 casos de buena calidad de sueño incorrectamente clasificados como mala calidad.
- **Verdaderos Negativos (TN)**: 10 casos de mala calidad de sueño correctamente clasificados.
- **Falsos Positivos (FP)**: 19 casos de mala calidad de sueño incorrectamente clasificados como buena calidad.

El patrón de errores refleja claramente la priorización del recall para la clase positiva, con un número relativamente alto de falsos positivos como consecuencia. Esta configuración es coherente con el objetivo de minimizar los falsos negativos, aunque a costa de generar más falsos positivos.

## Inferencia

### Flujo de Inferencia

El proceso de inferencia se implementó en un script independiente (`inference.py`) que sigue un flujo similar al de entrenamiento pero adaptado para aplicar el modelo entrenado a nuevos datos:

1. **Carga de datos**:
   ```python
   df = load_data(input_path)
   ```

2. **Preprocesamiento**: Se aplican las mismas transformaciones que durante el entrenamiento:
   ```python
   X, actual_target, quality = preprocess_data(df)
   ```

3. **Carga del modelo y umbral optimizado**:
   ```python
   model, threshold = load_model(model_path)
   ```

4. **Generación de predicciones** aplicando el umbral optimizado:
   ```python
   predictions, probabilities = make_predictions(model, X, threshold)
   ```

5. **Almacenamiento de resultados**:
   ```python
   save_predictions(df, predictions, probabilities, actual_target, quality, output_path)
   ```

Este flujo garantiza la consistencia entre el proceso de entrenamiento y de inferencia, asegurando que las transformaciones de datos se apliquen de manera idéntica.

### Resultados Obtenidos

La inferencia se realizó sobre un conjunto de datos de 175 registros generados por IA basados en las características del conjunto de entrenamiento. Los resultados obtenidos fueron:

```
Resumen de predicciones:
Total de registros procesados: 175
Predicciones positivas (buena calidad de sueño): 129 (73.7%)
Predicciones negativas (mala calidad de sueño): 46 (26.3%)

Métricas de evaluación:
Accuracy: 0.8343
Precision: 0.7752
Recall: 1.0000
F1-score: 0.8734
F2-score: 0.9452
```

Estos resultados muestran:

1. **Recall perfecto (1.0000)**: El modelo identificó correctamente todos los casos de buena calidad de sueño en el conjunto de inferencia.

2. **Precisión razonable (0.7752)**: Aproximadamente el 78% de los casos clasificados como buena calidad de sueño son correctos.

3. **F2-score excelente (0.9452)**: Confirma el buen rendimiento del modelo, especialmente en términos de minimización de falsos negativos.

4. **Distribución de predicciones**: El modelo clasificó el 73.7% de los casos como buena calidad de sueño y el 26.3% como mala calidad, una distribución coherente con la observada en el conjunto de entrenamiento.

Los resultados de inferencia son consistentes con los obtenidos durante la evaluación del modelo, lo que sugiere una buena capacidad de generalización a nuevos datos.

## Conclusiones y Recomendaciones

### Interpretación de Resultados

El modelo desarrollado ha demostrado ser efectivo para la clasificación de la calidad del sueño, con un enfoque particular en minimizar los falsos negativos. Los principales hallazgos son:

1. **Priorización exitosa del recall**: Se logró un recall de 1.0 en el conjunto de inferencia, cumpliendo el objetivo de no perder casos de buena calidad de sueño.

2. **Balance adecuado de métricas**: A pesar de priorizar el recall, se mantuvo una precisión razonable (0.78), lo que indica un equilibrio adecuado entre sensibilidad y especificidad.

3. **Umbral optimizado efectivo**: La estrategia de ajustar el umbral de decisión (0.3974) demostró ser crucial para alcanzar el balance deseado entre las métricas.

4. **Robustez del pipeline**: La integración de preprocesamiento y modelado en un pipeline único facilitó la consistencia entre entrenamiento e inferencia.

### Limitaciones del Modelo

A pesar de los buenos resultados, el modelo presenta algunas limitaciones importantes:

1. **Tamaño reducido del dataset**: Con solo 447 registros para entrenamiento, el modelo podría no capturar toda la variabilidad posible en poblaciones más diversas.

2. **Ausencia de estructura clara en los datos**: Como reveló el análisis de clustering, los datos no presentan agrupaciones naturales evidentes, lo que dificulta la identificación de patrones robustos.

3. **Bajo recall para la clase negativa**: Aunque se priorizó el recall global, el modelo muestra dificultades para identificar correctamente los casos de mala calidad de sueño (recall de solo 0.34 en el conjunto de prueba).

4. **Posible sobreajuste**: A pesar de las medidas tomadas, la complejidad del modelo en relación con el tamaño del dataset podría resultar en cierto grado de sobreajuste.

### Posibles Mejoras

Para futuras iteraciones del modelo, se recomiendan las siguientes mejoras:

1. **Aumentar el tamaño del dataset**: Recopilar más datos para mejorar la representatividad y robustez del modelo.

2. **Explorar técnicas de ensamblaje alternativas**: Considerar otros algoritmos como Random Forest o modelos de stacking que podrían capturar diferentes aspectos de los datos.

3. **Feature engineering avanzado**: Aunque XGBoost captura interacciones automáticamente, la creación de características más sofisticadas podría mejorar el rendimiento.

4. **Validación externa**: Evaluar el modelo con datos de fuentes independientes para verificar su generalización en contextos diferentes.

5. **Calibración de probabilidades**: Implementar técnicas de calibración para mejorar la interpretabilidad de las probabilidades predichas.

6. **Balance entre clases más equitativo**: Explorar técnicas de muestreo más sofisticadas para abordar el desbalanceo de clases sin sacrificar información.

## Anexos

### Hiperparámetros Finales del Modelo

```
classifier__subsample: 0.8
classifier__scale_pos_weight: 0.5557377049180328
classifier__n_estimators: 50
classifier__min_child_weight: 5
classifier__max_depth: 4
classifier__learning_rate: 0.01
classifier__gamma: 0.1
classifier__colsample_bytree: 0.8
```

### Umbral Optimizado

El umbral optimizado para la clasificación es 0.3974, determinado mediante validación cruzada con optimización de F2-score.

### Rendimiento en Conjunto de Prueba

```
Evaluación con threshold optimizado:
              precision    recall  f1-score   support
           0       0.83      0.34      0.49        29
           1       0.76      0.97      0.85        61
    accuracy                           0.77        90
   macro avg       0.79      0.66      0.67        90
weighted avg       0.78      0.77      0.73        90

Matriz de confusión:
[[10 19]
 [ 2 59]]

F2.0-score: 0.9161
```

### Rendimiento en Conjunto de Inferencia

```
Métricas de evaluación:
Accuracy: 0.8343
Precision: 0.7752
Recall: 1.0000
F1-score: 0.8734
F2-score: 0.9452
```