# 7. Reflexión crítica y conclusiones

## ¿Qué entorno fue más rápido y por qué?

No es posible dar una respuesta única y limpia, por la razón metodológica explicada en
{doc}`06_comparacion_final`: **el hardware cambió a mitad del experimento**. Los seis
modelos de scikit-learn se entrenaron en Colab gratuito; en PySpark, `LogisticRegression`
y `DecisionTree` también se completaron en el plan gratuito, pero `RandomForest`,
`GradientBoosting`, `LinearSVC` y `NaiveBayes` requirieron actualizar a **Colab Pro**
porque el plan gratuito no lograba completarlos (problemas de memoria/timeouts).

Separando por esa razón los resultados en dos grupos:

- **Comparación válida (mismo hardware nominal)** — `LogisticRegression` y
  `DecisionTree`: en ambos casos **scikit-learn fue más rápido** (6.8× y 1.8×
  respectivamente). Con datasets de ~1M de filas de entrenamiento, el overhead de
  coordinación de Spark (planificación de tareas, shuffles, serialización JVM↔Python vía
  Py4J) supera la ventaja de paralelismo que Spark ofrece, sobre todo en un clúster
  `local[*]` de pocos núcleos como el de Colab. Esto es consistente con lo señalado en el
  material del curso: PySpark tiende a justificarse a partir de volúmenes de datos mucho
  mayores (decenas de millones de filas o más) o en clústeres reales multi-nodo, no en un
  Spark local sobre una máquina de recursos moderados.
- **Comparación no concluyente (hardware distinto)** — `RandomForest`,
  `GradientBoosting`, `LinearSVC`, `NaiveBayes`: los tiempos observados (PySpark más lento
  en `RandomForest`/`NaiveBayes`, más rápido en `GradientBoosting`/`LinearSVC`) **no
  pueden atribuirse con confianza a diferencias de arquitectura**, porque los de PySpark
  se beneficiaron de más núcleos/RAM (Colab Pro) que los de scikit-learn (Colab
  gratuito). Es enteramente posible que, sobre el mismo hardware, PySpark hubiera sido
  más lento también en estos cuatro modelos — o que scikit-learn, con los mismos recursos
  de Colab Pro, hubiera sido aún más rápido.

**A partir de qué volumen de datos PySpark comenzaría a superar a scikit-learn** no se
puede determinar con los datos de este experimento por la misma razón. Sería necesario
repetir el ejercicio de escalabilidad (por ejemplo, entrenando con 10%, 25%, 50% y 100%
del dataset) manteniendo el hardware constante en ambos entornos — algo que quedó fuera
del alcance de esta entrega debido a las limitaciones de recursos encontradas.

## ¿Cuál entorno fue más preciso?

En **AUC**, los resultados son prácticamente equivalentes entre entornos para
`LogisticRegression`, `DecisionTree`, `RandomForest` y `GradientBoosting` (diferencias no
significativas o significativas pero por debajo del margen de relevancia práctica de
0.005 — ver {doc}`04_evaluacion_estadistica`). Las diferencias grandes y consistentes
aparecen en `LinearSVC` (PySpark 0.6128 vs. sklearn 0.5446) y `NaiveBayes` (PySpark 0.6700
vs. sklearn 0.6521), en ambos casos **a favor de PySpark**, y explicadas por diferencias
genuinas de implementación (optimizador OWLQN vs. `liblinear` para SVM lineal) más que por
azar de muestreo, dado que las tres pruebas estadísticas (DeLong, McNemar y bootstrap)
coinciden — salvo la ya discutida excepción de McNemar en `LinearSVC`.

## ¿Qué diferencias en el AUC resultaron significativas tras la corrección por comparaciones múltiples, y cuáles son relevantes en la práctica?

Ver la tabla completa en {doc}`04_evaluacion_estadistica`. En resumen: con ~253,196
observaciones de prueba, **la enorme mayoría de las comparaciones resultan
estadísticamente significativas** tras Holm, incluyendo diferencias de AUC tan pequeñas
como 0.0003–0.002. El margen de relevancia práctica (ΔAUC ≥ 0.005) definido de antemano
es lo que permite separar el ruido estadístico de las diferencias que importarían para una
decisión de negocio: bajo ese criterio, solo `LinearSVC` y `NaiveBayes` entre entornos, y
ningún par de modelos "fuertes" (`LogisticRegression`/`RandomForest`/`GradientBoosting`)
dentro de cada entorno, muestran diferencias relevantes.

## ¿Qué diferencias de implementación entre scikit-learn y PySpark explican las discrepancias observadas?

- **Discretización de árboles (`maxBins`)**: PySpark discretiza cada variable continua en
  un máximo de 32 intervalos por defecto, mientras que scikit-learn evalúa cortes exactos
  — esto explica de forma plausible el AUC ligeramente menor de `RandomForest` en PySpark
  (0.7111 vs. 0.7127).
- **Función de pérdida y optimizador de `LinearSVC`**: PySpark minimiza *hinge loss* con
  OWLQN; scikit-learn, aun forzando `loss="hinge"`, usa una implementación distinta — esto
  explica la brecha más grande de todo el estudio (ΔAUC ≈ 0.068).
- **Regularización**: se usó L2 en ambos entornos (`elasticNetParam=0` en la regresión
  logística de PySpark), pero la relación `C = 1/(regParam·n)` usada para hacer
  equivalentes las rejillas de búsqueda es solo aproximada, porque `C` pondera la suma de
  las pérdidas mientras que `regParam` pondera su promedio — esto podría explicar parte de
  la pequeña diferencia residual en `LogisticRegression` (no significativa en este caso,
  pero relevante como fuente potencial de discrepancia en otros datasets).

## ¿Qué limitaciones tiene la prueba de DeLong y cómo la complementan McNemar y bootstrap? ¿Coinciden las tres pruebas?

DeLong (y, de forma consistente, el bootstrap pareado) solo captura la variabilidad
debida al muestreo del conjunto de prueba — **no** la debida al entrenamiento (semillas,
particiones de la validación cruzada, aleatoriedad de `RandomForest`), asume
observaciones independientes, y resume el desempeño en *todos* los umbrales posibles, por
lo que puede no reflejar el desempeño en el umbral operativo que efectivamente se usaría
en producción.

Esto último es precisamente lo que McNemar expone y que DeLong no puede ver: en la
comparación de `LinearSVC` entre entornos, DeLong señala la diferencia de AUC más grande
del estudio, pero McNemar da p = 1.000, porque **ambos modelos, a su umbral operativo,
terminan tomando decisiones binarias casi idénticas** (de hecho, el de scikit-learn no
predice ningún positivo). En general, **DeLong y bootstrap coincidieron entre sí en las
15 comparaciones evaluadas** (ambos miden esencialmente lo mismo por caminos
distintos), mientras que **McNemar discrepó de ambos en 3 de las 8 comparaciones**
formales de la sección {doc}`04_evaluacion_estadistica` (`RandomForest` entre entornos, y
los dos pares "mejores 2 modelos" intra-entorno) — en todos los casos porque la ventaja de
ranking detectada por DeLong/bootstrap era demasiado pequeña para cambiar decisiones
concretas al umbral 0.5.

## ¿Qué aporta LIME y cuáles son sus limitaciones en entornos distribuidos?

LIME permitió examinar por qué el mejor modelo (`GradientBoosting`, sklearn) falló en dos
casos concretos de default no detectado (falsos negativos), lo cual es coherente con su
bajo recall (0.0627) — el modelo sistemáticamente prefiere no arriesgarse a predecir
default. Su limitación principal en este proyecto es que **no se pudo aplicar
directamente al modelo de PySpark**, porque LIME opera en memoria local generando miles
de perturbaciones evaluadas con `predict_proba`, lo cual sería extremadamente costoso de
ejecutar contra un `PipelineModel` distribuido (implicaría lanzar un *job* de Spark por
cada perturbación).

## ¿Qué efecto tuvo cada condición obligatoria en el rendimiento?

- **Partición común (mismo `id`/`split` en ambos entornos)**: condición indispensable
  para que la prueba de DeLong fuera válida — se confirmó que ambos entornos evaluaron
  exactamente 253,196 observaciones de prueba idénticas.
- **Prohibición de `.toPandas()`/`.collect()` sobre el dataset completo**: se respetó en
  todo momento; la única transferencia al driver fue, tras entrenar, de las columnas
  `id`, `label` y el score de la clase positiva — un costo adicional medido por separado
  (entre 1.9 s y 49.7 s según el modelo, siendo `RandomForest` el más costoso de
  transferir).
- **`cache`/`persist` del DataFrame preprocesado en Spark**: evitó recomputar el pipeline
  de preprocesamiento (`StringIndexer`+`OneHotEncoder`+`VectorAssembler`+`StandardScaler`)
  en cada fold de la validación cruzada.
- **Mismo número de folds (`cv=3` / `numFolds=3`) en ambos entornos**: garantizó que la
  comparación de AUC no estuviera sesgada por una validación cruzada más o menos
  exhaustiva en un entorno que en otro.

## Conclusión general

`GradientBoosting` fue el modelo con mejor AUC en ambos entornos (0.7162 sklearn / 0.7155
PySpark), aunque estadísticamente indistinguible en la práctica de `RandomForest` y
`LogisticRegression` (diferencias < 0.005 de AUC). **`LinearSVC` no es un modelo
utilizable para este problema en su configuración actual** (AUC cercano al azar en
sklearn, recall y precisión nulos) y requeriría, como mínimo, calibración de probabilidad
y ajuste de umbral. Todos los modelos "fuertes" comparten una limitación de fondo que el
AUC por sí solo no deja ver: **su recall a umbral 0.5 es muy bajo** (0.036–0.071), lo que
significa que, tal como están configurados, identificarían muy pocos de los préstamos que
realmente terminan en default — un hallazgo con implicancias directas para el objetivo de
negocio (reducir pérdidas por impago) que motivaría, como siguiente paso, explorar el
ajuste de umbral de decisión (Sección 9.3.6 del curso) y/o técnicas de balanceo de clases
que no se aplicaron en esta iteración.

Finalmente, la comparación de tiempos de cómputo entre scikit-learn y PySpark de este
proyecto **queda parcialmente invalidada por el cambio de hardware** (Colab gratuito →
Colab Pro) ocurrido durante el modelado con PySpark, y solo puede sostenerse con
confianza para `LogisticRegression` y `DecisionTree`, donde scikit-learn resultó
consistentemente más rápido — resultado esperable para un Spark en modo local sobre un
dataset de este tamaño, donde el overhead de coordinación distribuida no llega a
compensarse con el paralelismo disponible.
