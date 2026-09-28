# 3. Modelado

Se entrenaron los mismos seis modelos en ambos entornos, con espacios de búsqueda
equivalentes y el mismo criterio de selección (AUC ROC), sobre la partición común
descrita en {doc}`02_preprocesamiento`.

```{admonition} Sobre el checkpointing y el cambio de hardware
:class: warning

Para tolerar desconexiones del entorno de Colab durante entrenamientos largos, cada
modelo guarda sus predicciones y métricas en disco (Google Drive) inmediatamente después
de entrenarse; si el notebook se vuelve a ejecutar, cada modelo revisa primero si ya
existe su checkpoint y, de ser así, lo carga en vez de reentrenar.

**Esto es relevante para interpretar los tiempos reportados en este capítulo**: los seis
modelos de **scikit-learn** se entrenaron en un Colab **gratuito**. Para **PySpark**,
`LogisticRegression` y `DecisionTree` también se completaron en el entorno gratuito, pero
`RandomForest`, `GradientBoosting`, `LinearSVC` y `NaiveBayes` tuvieron que reentrenarse
en **Colab Pro** tras problemas de recursos (timeouts) en el plan gratuito. Por lo tanto,
**los tiempos de PySpark para estos últimos cuatro modelos no son directamente
comparables con los de scikit-learn**, al haberse ejecutado sobre hardware distinto (y
probablemente superior). Esta limitación se retoma en la discusión de tiempos de
{doc}`06_comparacion_final` y en {doc}`07_conclusiones`.
```

## 3.1 Espacios de búsqueda

| Modelo | scikit-learn | PySpark | Espacio de búsqueda |
|---|---|---|---|
| Regresión logística | `LogisticRegression` | `LogisticRegression` (`elasticNetParam=0`) | `regParam` ∈ {1e-6, 1e-5, 1e-4, 1e-3} en PySpark; `C = 1/(regParam·n)` equivalente en sklearn |
| Árbol de decisión | `DecisionTreeClassifier` | `DecisionTreeClassifier` | `max_depth`/`maxDepth` ∈ {5, 10, 15} |
| Bosque aleatorio | `RandomForestClassifier` | `RandomForestClassifier` | `n_estimators`/`numTrees` ∈ {10, 50, 100}, `max_depth`/`maxDepth` ∈ {5, 10, 15} |
| Gradient boosting | `GradientBoostingClassifier` | `GBTClassifier` | `n_estimators`/`maxIter` ∈ {50, 100}, `max_depth`/`maxDepth` ∈ {3, 5}, `learning_rate`/`stepSize` = 0.1 |
| SVM lineal | `LinearSVC(loss="hinge")` | `LinearSVC` | mismo criterio que regresión logística para `C`/`regParam` |
| Naive Bayes | `GaussianNB` | `NaiveBayes(modelType="gaussian")` | sin búsqueda de hiperparámetros |

Ambos entornos usaron `cv=3` / `numFolds=3` y `scoring="roc_auc"` /
`BinaryClassificationEvaluator(metricName="areaUnderROC")` como criterio de selección.

## 3.2 Resultados — scikit-learn

| Modelo | AUC | Accuracy | Precision | Recall | F1 | Tiempo entrenamiento | Tiempo predicción |
|---|---|---|---|---|---|---|---|
| **GradientBoosting** | **0.7162** | 0.8071 | 0.5535 | 0.0627 | 0.1127 | 7,144.9 s (≈ 1h 59min) | 2.61 s |
| RandomForest | 0.7127 | 0.8067 | 0.5829 | 0.0361 | 0.0680 | 2,534.9 s (≈ 42.2 min) | 3.35 s |
| LogisticRegression | 0.7114 | 0.8063 | 0.5410 | 0.0559 | 0.1014 | 139.3 s (2.3 min) | 0.05 s |
| DecisionTree | 0.7032 | 0.8050 | 0.5061 | 0.0708 | 0.1242 | 160.7 s (2.7 min) | 0.12 s |
| NaiveBayes | 0.6521 | 0.6930 | 0.3135 | 0.4807 | 0.3795 | 3.1 s | 0.53 s |
| LinearSVC | 0.5446 | 0.8047 | 0.0000 | 0.0000 | 0.0000 | 6,528.0 s (≈ 1h 49min) | 0.64 s |

**Observaciones**:

- `GradientBoosting` obtuvo el mejor AUC (0.7162), seguido muy de cerca por
  `RandomForest` y `LogisticRegression` (diferencias de solo 0.001–0.005 en AUC — se
  evalúa su significancia estadística y relevancia práctica en {doc}`04_evaluacion_estadistica`).
- **`LinearSVC` prácticamente falló** en este problema: su AUC (0.5446) es apenas mejor
  que el azar, y su precisión/recall/F1 son **todos 0.0000** — el modelo, al umbral por
  defecto (`decision_function > 0`), no predijo ni un solo positivo en todo el conjunto de
  prueba. Esto es consistente con lo señalado en el material del curso sobre SVM lineal
  sin calibración de probabilidades en datasets desbalanceados: la pérdida *hinge* sin
  ajuste de umbral tiende a favorecer fuertemente la clase mayoritaria.
- `NaiveBayes`, pese a tener el AUC más bajo entre los modelos "razonables" (0.6521), es
  el **único modelo con un F1 y un recall considerables** (0.3795 y 0.4807
  respectivamente) — su supuesto de independencia condicional, aunque simplista, evita el
  colapso hacia la clase mayoritaria que sí sufren `RandomForest`, `GradientBoosting` y
  `LogisticRegression` (todos con recall < 0.08 al umbral 0.5). Esto sugiere que, con el
  umbral por defecto, los modelos "fuertes" están optimizando accuracy/AUC a costa de
  identificar muy pocos de los préstamos que sí caen en default — un punto crítico para la
  aplicación de negocio (ver {doc}`07_conclusiones`).
- El costo computacional entre modelos varía en más de tres órdenes de magnitud: de 3.1 s
  (`NaiveBayes`) a 7,145 s (`GradientBoosting`), reflejando el mayor costo del *grid
  search* con validación cruzada sobre >1M de filas de entrenamiento para modelos de
  ensamble/boosting.

## 3.3 Resultados — PySpark

| Modelo | AUC | Tiempo entrenamiento | Tiempo predicción | Tiempo transferencia al driver |
|---|---|---|---|---|
| **GradientBoosting** | **0.7155** | 4,756.5 s (≈ 1h 19min) † | 0.26 s | 8.86 s |
| LogisticRegression | 0.7114 | 943.7 s (15.7 min) | 0.28 s | 7.85 s |
| RandomForest | 0.7111 | 10,637.0 s (≈ 2h 57min) † | 0.32 s | 49.71 s |
| DecisionTree | 0.7035 | 286.0 s (4.8 min) | 0.27 s | 4.22 s |
| NaiveBayes | 0.6700 | 6.9 s † | 0.13 s | 3.11 s |
| LinearSVC | 0.6128 | 1,772.7 s (29.5 min) † | 0.03 s | 1.87 s |

† Entrenado en Colab Pro (ver recuadro al inicio del capítulo); `LogisticRegression` y
`DecisionTree` se entrenaron en Colab gratuito.

**Observaciones**:

- El AUC de `GradientBoosting`, `LogisticRegression` y `DecisionTree` en PySpark es
  **prácticamente idéntico** al de scikit-learn (diferencias de 0.0003–0.0007), lo cual es
  el resultado esperable cuando dos implementaciones razonables del mismo algoritmo, con
  el mismo criterio de regularización (L2, `elasticNetParam=0`) y la misma partición, se
  entrenan sobre los mismos datos.
- `RandomForest` en PySpark obtuvo un AUC **ligeramente menor** que en sklearn (0.7111 vs.
  0.7127) — diferencia pequeña, probablemente explicada por la discretización en `maxBins`
  (32 por defecto) que usa la implementación de árboles de Spark, frente a los cortes
  exactos que evalúa scikit-learn (ver discusión en {doc}`04_evaluacion_estadistica`).
- **`LinearSVC` y `NaiveBayes` de PySpark superan claramente a sus contrapartes de
  sklearn** (LinearSVC: 0.6128 vs. 0.5446; NaiveBayes: 0.6700 vs. 0.6521). Para
  `LinearSVC`, esto es consistente con la diferencia de implementación documentada en el
  curso: PySpark minimiza la pérdida *hinge* con un optimizador OWLQN, mientras que
  scikit-learn (aun forzando `loss="hinge"`) usa `liblinear`/`libsvm` con una
  parametrización distinta de la regularización — ambas son "SVM lineal" pero no producen
  el mismo modelo.

## 3.4 Comparación visual de tiempos de entrenamiento

```{figure} figures/tiempos_entrenamiento.png
:name: fig-tiempos
:width: 100%

Tiempo de entrenamiento (incluye validación cruzada / `CrossValidator`) por modelo y
entorno. **Recordar la nota de hardware**: los tiempos de PySpark para `RandomForest`,
`GradientBoosting`, `LinearSVC` y `NaiveBayes` corresponden a Colab Pro, mientras que
todos los tiempos de scikit-learn y los de PySpark para `LogisticRegression` y
`DecisionTree` corresponden a Colab gratuito.
```

Con esa salvedad, el patrón bruto observado es:

- `LogisticRegression`, `RandomForest` y `NaiveBayes`: PySpark tardó más que scikit-learn.
- `DecisionTree`: tiempos similares (PySpark ligeramente más lento).
- `GradientBoosting` y `LinearSVC`: PySpark aparenta ser más rápido, pero **esta
  comparación está confundida por el cambio de hardware** y no debe usarse como evidencia
  de que PySpark es intrínsecamente más rápido para estos algoritmos (ver
  {doc}`07_conclusiones`).
