1. Exploracion del data set en explore.ipynb
2 TRAIN

Separamos la varible de blodd preseare en dos para que sea mas interpretable den sistolica y diastolica por el separador de /
Rellenamoss los valores faltantes de la clumna de Sleep disorder por None ya que un valor nuelo indica que no existe efefnerdad relacionad con el paciente e imputar por la moda no seria correcto en estecaso
realizamos un peuqeño feature engineering , ya que el propio algortimo de xgboost captura la importancia de las variables y no es necesario realizar un feature engineering exhaustivo
dentro de la piline añadisños prceoso de imputacion ya que dentro de un entrono de infercnia no saber si el data set contrendrsa valores nulos aunque el nuestro de entreamineto no lo tenga , añadirlo o no no afectara al etrenamiento pero sera esencial en el caso de que el data set de inferencia contenga valores nulos
Observamos que target es desbalanceado por lo que nicalentet usamos smote pero por simpliciad usaremos elproppio algoritmo de xgboost que tiene un parametro para el desbalanceo de clases 'scale_pos_weight' que se puede usar para ajustar el desbalanceo de clases
eligimos el model e xgboost por su rapidez y por su capacidad de interpretacion de los resultados ademas de su capacidad de generalizacion e identificacion de patrones sin necesidad de un feature engineering exhaustivo, ademas por su robusten en dataset pequeños , desbalnceados y con pocas features
randomserach de hiperparametros para ajustar el modelo efoqcados a mitigar el posible oferfitting al trabajar con un dataset pequeño y un modelo complejo
cross validation para evitar el overfitting y ajustar los hiperparametros con 10 slipts debido a la poca cantidad de datos , porque no afecta tanto al tiempo de entrenamiento y es mas robusto que un simple split
uso de cv en threhold para poder ajustar el umbral de decision del modelo y los hiperparametros del xgboost
IMPORTANTE:
la metrica a anilizar en cambio al f1 score quees la que se suele usar en problemas de clasificacion desbalanceada, usamos el f2 score ya que es mas sensible a los falsos negativos que el f1 score y en este caso es mas importante no perder pacientes que no tienen la enfermedad que detectar pacientes que si la tienen, por lo que es mas importante no perder pacientes que no tienen la enfermedad que detectar pacientes que si la tienen
por lo que consegimos un recall excelante de 1 a cambio de una prescionun poco mas baja anuque sgue siedo un resultado excelente para nuetro problema

dando tood esto a este resulatdo Iniciando proceso de modelado para calidad del sueño...
Cargando datos desde data/train.csv...
Preparando features y variable objetivo...
Datos preparados: 447 registros, 14 características
División train-test: 357 registros de entrenamiento, 90 de test
Ratio de desbalance (neg:pos): 0.46
Optimizando hiperparámetros para XGBoost...
Fitting 10 folds for each of 100 candidates, totalling 1000 fits
Mejores hiperparámetros encontrados:
  classifier__subsample: 0.8
  classifier__scale_pos_weight: 0.5557377049180328
  classifier__n_estimators: 50
  classifier__min_child_weight: 5
  classifier__max_depth: 4
  classifier__learning_rate: 0.01
  classifier__gamma: 0.1
  classifier__colsample_bytree: 0.8
Mejor F2.0-score: 0.9182
Realizando validación cruzada para encontrar umbral óptimo (F2.0-score)...
  Procesando fold 1/10...
  Procesando fold 2/10...
  Procesando fold 3/10...
  Procesando fold 4/10...
  Procesando fold 5/10...
  Procesando fold 6/10...
  Procesando fold 7/10...
  Procesando fold 8/10...
  Procesando fold 9/10...
  Procesando fold 10/10...
  Umbral óptimo encontrado: 0.3974 (F2.0-score = 0.9321)
Reporte de clasificación con umbral óptimo:
              precision    recall  f1-score   support
           0       0.90      0.41      0.56       113
           1       0.78      0.98      0.87       244
    accuracy                           0.80       357
   macro avg       0.84      0.69      0.72       357
weighted avg       0.82      0.80      0.77       357
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
Modelo y threshold guardados en 'model/pipeline.cloudpkl'
Proceso de modelado completado.


finalemte se reliza una inferecnia antes dastos desconooidos creados por una IA de calude en funcion al propio daaset de entrenamiento y obtenemos :
Iniciando proceso de inferencia para calidad del sueño...
Cargando datos desde data/inference.csv...
Preprocesando datos de entrada...
Datos preprocesados: 175 registros, 14 características
Cargando modelo desde model/pipeline.cloudpkl...
Modelo cargado correctamente con threshold 0.3974
Realizando predicciones...
Guardando predicciones en data/output_predictions.csv...
Predicciones guardadas exitosamente en data/output_predictions.csv
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
Proceso de inferencia completado.