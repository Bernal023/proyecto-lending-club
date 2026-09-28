# 6. Comparación de resultados

## 6.1 Tabla comparativa consolidada

| Modelo | AUC sklearn | AUC PySpark | Tiempo fit sklearn | Tiempo fit PySpark | Tiempo predicción sklearn | Tiempo predicción PySpark |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.7114 | 0.7114 | 139.3 s | 943.7 s | 0.05 s | 0.28 s |
| DecisionTree | 0.7032 | 0.7035 | 160.7 s | 286.0 s | 0.12 s | 0.27 s |
| RandomForest | 0.7127 | 0.7111 | 2,534.9 s | 10,637.0 s | 3.35 s | 0.32 s |
| GradientBoosting | **0.7162** | **0.7155** | 7,144.9 s | 4,756.5 s | 2.61 s | 0.26 s |
| LinearSVC | 0.5446 | 0.6128 | 6,528.0 s | 1,772.7 s | 0.64 s | 0.03 s |
| NaiveBayes | 0.6521 | 0.6700 | 3.1 s | 6.9 s | 0.53 s | 0.13 s |

`GradientBoosting` es el mejor modelo en ambos entornos (aunque, como se vio en
{doc}`04_evaluacion_estadistica`, prácticamente empatado en términos prácticos con
`RandomForest` y `LogisticRegression`).

## 6.2 Curvas ROC

```{figure} figures/curvas_roc.png
:name: fig-roc
:width: 100%

Curvas ROC de los seis modelos, superpuestas por entorno (izquierda: scikit-learn;
derecha: PySpark).
```

Las curvas ROC confirman visualmente el ranking de AUC: `GradientBoosting`,
`RandomForest` y `LogisticRegression` prácticamente se superponen en ambos entornos;
`DecisionTree` queda ligeramente por debajo; `NaiveBayes` claramente por debajo de esos
cuatro; y `LinearSVC` es el que más se acerca a la diagonal de un clasificador aleatorio
— sobre todo en scikit-learn, donde su curva es visiblemente la más pobre del gráfico.

## 6.3 Tiempos de cómputo

```{figure} figures/tiempos_entrenamiento.png
:name: fig-tiempos-final
:width: 100%

Tiempo de entrenamiento por modelo y entorno (incluye validación cruzada).
```

```{important}
**Esta comparación de tiempos tiene una limitación metodológica importante**: no todos
los tiempos se midieron sobre el mismo hardware. Los seis modelos de scikit-learn y los
modelos `LogisticRegression`/`DecisionTree` de PySpark se ejecutaron en **Colab
gratuito**; `RandomForest`, `GradientBoosting`, `LinearSVC` y `NaiveBayes` de PySpark se
ejecutaron en **Colab Pro** tras problemas de recursos en el entorno gratuito. Por lo
tanto:

- La comparación **LogisticRegression** y **DecisionTree** (mismo hardware nominal en
  ambos entornos) es la más confiable de la tabla: en ambos casos, **PySpark resultó más
  lento** que scikit-learn (6.8× y 1.8× respectivamente) — coherente con el overhead
  conocido de Spark en modo local para datasets que, aunque grandes (>1M filas), no son
  tan grandes como para que el paralelismo distribuido compense la sobrecarga de
  serialización/coordinación de tareas.
- Para **`RandomForest`, `GradientBoosting`, `LinearSVC` y `NaiveBayes`**, no puede
  concluirse de forma limpia si las diferencias observadas (PySpark más rápido en GBT y
  LinearSVC, más lento en RandomForest y NaiveBayes) se deben a la arquitectura de cada
  framework o al cambio de hardware entre mediciones. Una comparación rigurosa
  requeriría repetir estos cuatro modelos de PySpark en el mismo entorno gratuito (o,
  alternativamente, repetir los de scikit-learn también en Colab Pro).
```

## 6.4 Mapas de calor de DeLong entre entornos

```{figure} figures/delong_entornos_deltaauc.png
:name: fig-heatmap-delta-final
:width: 70%

ΔAUC entre scikit-learn y PySpark, por modelo.
```

```{figure} figures/delong_entornos_pvalue.png
:name: fig-heatmap-p-final
:width: 70%

p-values de DeLong (ajustados por Holm) entre scikit-learn y PySpark, por modelo.
```

Ver la discusión completa de estos resultados en {doc}`04_evaluacion_estadistica`.
