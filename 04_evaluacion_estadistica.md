# 4. Comparación estadística de los modelos

## 4.1 Validación de la implementación rápida de DeLong

Se implementó la versión rápida de DeLong (Sun y Xu, 2014, O(n log n)) por midranks, ya
que la versión directa (O(n²)) es inviable con las ~253,196 observaciones del conjunto de
prueba. Se validó primero sobre un ejemplo pequeño (n=300), comparando contra
`sklearn.metrics.roc_auc_score` y contra un cálculo directo de la estadística U de
Mann-Whitney:

| Método | AUC modelo 1 | AUC modelo 2 |
|---|---|---|
| DeLong rápido | 0.686222140357**3402** | 0.667664394916**1908** |
| `roc_auc_score` (sklearn) | 0.686222140357**3402** | 0.667664394916**1909** |
| Cálculo directo O(n²) | 0.686222140357**3402** | 0.667664394916**1908** |

Los tres métodos coinciden hasta el décimo dígito decimal (la diferencia en el último
dígito es error de redondeo de punto flotante) — **la implementación rápida queda
validada**. Para este ejemplo de validación, z = 0.446, p = 0.656 (diferencia no
significativa, como se esperaba de datos simulados con señal similar).

## 4.2 DeLong: comparación entre entornos (scikit-learn vs. PySpark)

Se definió un margen de relevancia práctica de **ΔAUC ≥ 0.005** (justificado porque, con
~253,196 observaciones de prueba, incluso diferencias de AUC de 0.001 resultan
estadísticamente significativas por el enorme poder estadístico de la muestra, sin que
ello implique una diferencia relevante para la aplicación de negocio). Se aplicó
corrección de **Holm** dentro de esta familia de 6 comparaciones.

| Modelo | AUC sklearn | AUC PySpark | ΔAUC | z | p (Holm) | ¿Significativo? | ¿Relevante en la práctica? |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.7114 | 0.7114 | −0.00004 | −1.37 | 0.340 | No | No |
| DecisionTree | 0.7032 | 0.7035 | −0.00029 | −0.56 | 0.573 | No | No |
| RandomForest | 0.7127 | 0.7111 | +0.00164 | 9.29 | < 0.001 | **Sí** | No |
| GradientBoosting | 0.7162 | 0.7155 | +0.00070 | 3.60 | < 0.001 | **Sí** | No |
| LinearSVC | 0.5446 | 0.6128 | **−0.06817** | −39.17 | < 0.001 | **Sí** | **Sí** |
| NaiveBayes | 0.6521 | 0.6700 | **−0.01786** | −38.43 | < 0.001 | **Sí** | **Sí** |

```{figure} figures/delong_entornos_deltaauc.png
:name: fig-delong-ent-delta
:width: 70%

ΔAUC (scikit-learn − PySpark) por modelo.
```

```{figure} figures/delong_entornos_pvalue.png
:name: fig-delong-ent-p
:width: 70%

p-value de DeLong ajustado por Holm, entre entornos, por modelo.
```

**Lectura**: `LogisticRegression` y `DecisionTree` son estadísticamente equivalentes
entre entornos — es el resultado esperado, ya que ambos algoritmos tienen formulaciones
muy similares en scikit-learn y PySpark. `RandomForest` y `GradientBoosting` muestran
diferencias estadísticamente significativas pero **de magnitud trivial** (< 0.002 AUC),
por debajo del margen de relevancia práctica: en la práctica, ambos entornos entrenan
modelos de calidad equivalente para estos dos algoritmos. Las diferencias grandes y
relevantes están en `LinearSVC` y `NaiveBayes` — precisamente los dos algoritmos donde las
implementaciones de sklearn y PySpark difieren más en su formulación matemática interna
(ver {doc}`03_modelado`).

## 4.3 DeLong: comparación entre modelos dentro de cada entorno

15 comparaciones por entorno (todos los pares posibles entre los 6 modelos), con
corrección de Holm dentro de cada familia.

### scikit-learn

Todas las 15 comparaciones resultaron estadísticamente significativas (p < 0.001 tras
Holm) — esperable dado el tamaño de muestra. En cuanto a **relevancia práctica**
(ΔAUC ≥ 0.005), tres pares **no** superan el margen:

| Comparación | ΔAUC | ¿Relevante? |
|---|---|---|
| LogisticRegression vs. RandomForest | −0.0014 | No |
| LogisticRegression vs. GradientBoosting | −0.0049 | No (justo debajo del margen) |
| RandomForest vs. GradientBoosting | −0.0035 | No |

Es decir: **`LogisticRegression`, `RandomForest` y `GradientBoosting` forman un grupo
estadísticamente distinguible entre sí pero prácticamente equivalente** en capacidad de
discriminación (AUC ≈ 0.71–0.72), muy por encima de `DecisionTree` (0.70),
`NaiveBayes` (0.65) y muy por encima de `LinearSVC` (0.54, cercano al azar).

### PySpark

Patrón similar, con una diferencia notable: **`LogisticRegression` vs. `RandomForest`
NO es significativa en PySpark** (p = 0.360 tras Holm; ΔAUC = 0.0003), mientras que en
sklearn sí lo era. Esto sugiere que, en la implementación de PySpark, ambos algoritmos
convergen a un desempeño aún más parecido que en scikit-learn.

```{figure} figures/delong_sklearn_deltaauc.png
:name: fig-delong-sk-delta
:width: 70%

ΔAUC entre todos los pares de modelos, scikit-learn.
```

```{figure} figures/delong_sklearn_pvalue.png
:name: fig-delong-sk-p
:width: 70%

p-values de DeLong (Holm) entre todos los pares de modelos, scikit-learn.
```

```{figure} figures/delong_spark_deltaauc.png
:name: fig-delong-sp-delta
:width: 70%

ΔAUC entre todos los pares de modelos, PySpark.
```

```{figure} figures/delong_spark_pvalue.png
:name: fig-delong-sp-p
:width: 70%

p-values de DeLong (Holm) entre todos los pares de modelos, PySpark.
```

## 4.4 Pruebas complementarias: McNemar y bootstrap pareado

Se aplicaron ambas pruebas (a) a las 6 comparaciones entre entornos y (b) al par de
modelos con mayor AUC dentro de cada entorno, con corrección de Holm por familia. El
umbral de decisión usado para McNemar fue 0.5 (probabilidades) o 0 (`decision_function`
de `LinearSVC`), definido sin usar el conjunto de prueba.

### Entre entornos

| Comparación | ΔAUC (DeLong) | p DeLong (Holm) | p McNemar (Holm) | ΔAUC bootstrap (media) | IC 95% bootstrap |
|---|---|---|---|---|---|
| LogisticRegression | −0.00004 | 0.340 | 1.000 | −0.00004 | (−0.0001, 0.00002) → incluye 0 |
| DecisionTree | −0.00029 | 0.573 | 1.000 | −0.00030 | (−0.0013, 0.0006) → incluye 0 |
| RandomForest | +0.00164 | < 0.001 | 0.372 | +0.00164 | (0.0013, 0.0020) → **excluye 0** |
| GradientBoosting | +0.00070 | < 0.001 | **0.018** | +0.00071 | (0.0003, 0.0011) → **excluye 0** |
| LinearSVC | −0.06817 | < 0.001 | **1.000** | −0.06822 | (−0.0716, −0.0648) → **excluye 0** |
| NaiveBayes | −0.01786 | < 0.001 | < 0.001 | −0.01786 | (−0.0188, −0.0169) → **excluye 0** |

**Coincidencias y discrepancias entre las tres pruebas**:

- **LogisticRegression y DecisionTree**: las tres pruebas coinciden en "sin diferencia" —
  el caso más limpio.
- **GradientBoosting y NaiveBayes**: las tres pruebas coinciden en "sí hay diferencia" —
  DeLong, McNemar y bootstrap se refuerzan mutuamente.
- **RandomForest**: DeLong y bootstrap detectan una diferencia significativa (aunque
  mínima, 0.0016, por debajo del margen de relevancia práctica), pero **McNemar no la
  detecta** (p = 0.372). Esto ilustra exactamente la diferencia conceptual entre las
  pruebas: DeLong/bootstrap comparan la capacidad de **ordenar** correctamente todas las
  observaciones (AUC, sensible a diferencias pequeñas y consistentes en el ranking),
  mientras que McNemar solo compara las **decisiones binarias concretas** al umbral 0.5 —
  una ventaja de ranking tan pequeña puede no alcanzar a cambiar ninguna decisión de
  clasificación.
- **LinearSVC — el caso más ilustrativo de discrepancia**: DeLong y bootstrap señalan la
  diferencia de AUC más grande de todo el estudio (ΔAUC ≈ 0.068, altamente significativa),
  pero **McNemar da p = 1.000** (ninguna diferencia). La explicación está en los
  resultados de la {doc}`03_modelado`: el `LinearSVC` de scikit-learn tiene
  precision = recall = F1 = 0.0000, es decir, **nunca predice la clase positiva** al
  umbral por defecto. Si el `LinearSVC` de PySpark, a su propio umbral, también termina
  clasificando casi todas las observaciones como negativas, entonces ambos modelos
  producen decisiones binarias casi idénticas (pocas o ninguna discordancia b/c en la
  tabla de McNemar) **a pesar de que sus puntuaciones continuas ordenan a las
  observaciones de forma muy distinta** (reflejado en el AUC). Esta es la limitación de
  McNemar que el enunciado pide discutir: por comparar solo decisiones a un umbral fijo,
  puede ser ciego a diferencias reales en la calidad del ranking cuando ambos modelos
  colapsan hacia la misma decisión trivial.

### Dentro de cada entorno (mejores 2 modelos)

| Entorno | Comparación | ΔAUC (DeLong) | p DeLong | p McNemar | ΔAUC bootstrap | IC 95% bootstrap |
|---|---|---|---|---|---|---|
| scikit-learn | GradientBoosting vs. RandomForest | +0.00351 | < 0.001 | 0.205 (Holm) | +0.00351 | (0.0029, 0.0040) → excluye 0 |
| PySpark | GradientBoosting vs. LogisticRegression | +0.00413 | < 0.001 | 0.578 | +0.00411 | (0.0034, 0.0048) → excluye 0 |

En ambos entornos se repite el mismo patrón que con `RandomForest` en la sección
anterior: DeLong y bootstrap detectan una ventaja consistente (aunque pequeña, por debajo
del margen de relevancia práctica de 0.005) del mejor modelo sobre el segundo mejor, pero
McNemar no encuentra diferencias en las decisiones concretas al umbral 0.5.

## 4.5 Síntesis de la sección

1. **DeLong y bootstrap pareado son altamente consistentes entre sí** en todas las
   comparaciones realizadas — es el resultado esperado, ya que ambos miden esencialmente
   lo mismo (diferencia de AUC), solo que con métodos de inferencia distintos
   (asintótico vs. remuestreo). Esto valida indirectamente la implementación de DeLong.
2. **McNemar mide algo genuinamente distinto** (decisiones a un umbral fijo, no
   capacidad de ranking) y por eso puede discrepar de DeLong/bootstrap — de forma más
   notoria cuanto más cerca estén los modelos de tomar decisiones triviales o casi
   idénticas a pesar de puntuaciones distintas (caso extremo: `LinearSVC`).
3. Con ~253,196 observaciones de prueba, **casi cualquier diferencia de AUC resulta
   estadísticamente significativa**; el margen de relevancia práctica (ΔAUC ≥ 0.005)
   es indispensable para no sobre-interpretar diferencias triviales como
   `RandomForest` vs. `GradientBoosting` en cada entorno.
4. Las diferencias grandes y consistentes en las tres pruebas (`LinearSVC` y
   `NaiveBayes` entre entornos) apuntan a diferencias reales de implementación
   (formulación de la pérdida, optimizador) más que a ruido de muestreo.
