# 1. Análisis Exploratorio de Datos (EDA)

## 1.1 Carga inicial y visión general

El archivo se cargó completo (sin muestreo) con `pandas.read_csv`, restringiendo las
columnas a las mínimas requeridas por el enunciado (`loan_amnt`, `int_rate`,
`fico_range_high`, `emp_length`, `annual_inc`, `purpose`, `home_ownership`, `dti`,
`addr_state`, `verification_status`, `loan_status`, `term`, `grade`, `installment`,
`revol_util`, `open_acc`).

| Métrica | Valor |
|---|---|
| Tamaño del archivo en disco | 1.56 GB |
| Tiempo de carga (pandas) | 31.1 s |
| Filas × columnas cargadas | 2,260,701 × 16 |

Tras filtrar a `Fully Paid` / `Charged Off` (ver {doc}`intro`), el dataset de trabajo para
el EDA quedó en **1,345,310** filas.

## 1.2 Análisis unidimensional

### Variables numéricas

| Variable | count | mean | std | min | 25% | 50% | 75% | max | IQR | skewness | missing % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loan_amnt | 1,345,310 | 14,419.97 | 8,717.05 | 500.00 | 8,000.00 | 12,000.00 | 20,000.00 | 40,000.00 | 12,000.00 | 0.78 | 0.00% |
| int_rate | 1,345,310 | 13.24 | 4.77 | 5.31 | 9.75 | 12.74 | 15.99 | 30.99 | 6.24 | 0.71 | 0.00% |
| annual_inc | 1,345,310 | 76,247.64 | 69,925.10 | 0.00 | 45,780.00 | 65,000.00 | 90,000.00 | 10,999,200.00 | 44,220.00 | 46.32 | 0.00% |
| dti | 1,344,936 | 18.28 | 11.16 | −1.00 | 11.79 | 17.61 | 24.06 | 999.00 | 12.27 | 27.11 | 0.03% |
| fico_range_high | 1,345,310 | 700.19 | 31.85 | 629.00 | 674.00 | 694.00 | 714.00 | 850.00 | 40.00 | 1.29 | 0.00% |
| emp_length | 1,266,799 | 6.05 | 3.56 | 1.00 | 2.00 | 6.00 | 10.00 | 10.00 | 8.00 | −0.14 | 5.84% |
| installment | 1,345,310 | 438.08 | 261.51 | 4.93 | 248.48 | 375.43 | 580.73 | 1,719.83 | 332.25 | 1.01 | 0.00% |
| revol_util | 1,344,453 | 51.81 | 24.52 | 0.00 | 33.40 | 52.20 | 70.70 | 892.30 | 37.30 | −0.04 | 0.06% |
| open_acc | 1,345,310 | 11.59 | 5.47 | 0.00 | 8.00 | 11.00 | 14.00 | 90.00 | 6.00 | 1.30 | 0.00% |

**Hallazgos clave**:
- `annual_inc` tiene una **asimetría extrema** (skewness = 46.3) por la presencia de
  ingresos anuales de hasta 10,999,200 — claros outliers que conviene tratar (winsorizar
  o transformar con log) antes de usar modelos lineales.
- `dti` tiene un mínimo de −1 y un máximo de 999, lo que sugiere valores centinela o
  errores de captura; su skewness de 27.1 confirma la distorsión.
- El resto de variables numéricas tiene asimetría moderada (0.7–1.3), consistente con
  distribuciones de ingresos/montos de crédito típicas (colas a la derecha).

**Outliers (criterio de Tukey, IQR)**:

| Variable | Límite inferior | Límite superior | N outliers | % outliers |
|---|---|---|---|---|
| loan_amnt | −10,000.00 | 38,000.00 | 7,157 | 0.53% |
| int_rate | 0.39 | 25.35 | 24,977 | 1.86% |
| annual_inc | −20,550.00 | 156,330.00 | 65,870 | 4.90% |
| dti | −6.62 | 42.47 | 5,473 | 0.41% |
| fico_range_high | 614.00 | 774.00 | 46,497 | 3.46% |
| emp_length | −10.00 | 22.00 | 0 | 0.00% |
| installment | −249.90 | 1,079.11 | 42,042 | 3.13% |
| revol_util | −22.55 | 126.65 | 72 | 0.01% |
| open_acc | −1.00 | 23.00 | 46,112 | 3.43% |

`annual_inc` es la variable con mayor proporción de outliers (4.90%), coherente con su
altísima asimetría.

```{figure} figures/eda_histogramas_boxplots.png
:name: fig-hist-box
:width: 100%

Histogramas y boxplots de las 9 variables numéricas seleccionadas.
```

### Variables categóricas

Se calcularon frecuencias absolutas/relativas para `purpose`, `home_ownership`,
`addr_state`, `verification_status`, `term` y `grade` (ver tasas de default por categoría
en la sección 1.3, que resume la misma información de forma más útil para el modelado).

```{figure} figures/eda_distribucion_categoricas.png
:name: fig-cat-dist
:width: 100%

Distribución de frecuencias de las variables categóricas principales.
```

### Variable objetivo (`default`)

| Clase | Conteo | % |
|---|---|---|
| 0 (Fully Paid) | 1,076,751 | 80.04% |
| 1 (Charged Off) | 268,559 | 19.96% |

```{figure} figures/eda_distribucion_target.png
:name: fig-target-dist
:width: 80%

Distribución de la variable objetivo `default`.
```

**Interpretación**: existe un desbalance de clases moderado (~4:1). Esto motiva usar
AUC, AUC-PR y F1 (además de accuracy) como métricas de evaluación, y estratificar la
partición train/test por la clase objetivo (ver {doc}`02_preprocesamiento`).

## 1.3 Análisis bidimensional (relación con `default`)

### Variables numéricas vs. `default`

Prueba de Mann-Whitney (no paramétrica, dado que ninguna variable es normal según
Shapiro-Wilk) y correlación punto-biserial:

| Variable | Media (default=0) | Media (default=1) | Diferencia | p (Mann-Whitney) | Correlación punto-biserial |
|---|---|---|---|---|---|
| int_rate | 12.62 | 15.71 | +3.09 | < 0.001 | **0.259** |
| fico_range_high | 702.26 | 691.85 | −10.41 | < 0.001 | **−0.131** |
| dti | 17.81 | 20.17 | +2.36 | < 0.001 | 0.085 |
| loan_amnt | 14,134.37 | 15,565.06 | +1,430.69 | < 0.001 | 0.066 |
| revol_util | 51.07 | 54.76 | +3.68 | < 0.001 | 0.060 |
| installment | 431.32 | 465.15 | +33.82 | < 0.001 | 0.052 |
| annual_inc | 77,705.95 | 70,400.74 | −7,305.20 | < 0.001 | −0.042 |
| open_acc | 11.52 | 11.90 | +0.38 | < 0.001 | 0.028 |
| emp_length | 6.08 | 5.95 | −0.13 | < 0.001 | −0.014 |

**`int_rate` y `fico_range_high` son, por un margen amplio, las variables numéricas más
asociadas con el default** — resultado esperado, ya que la tasa de interés que Lending
Club asigna a un préstamo ya incorpora su propia evaluación de riesgo crediticio
(reflejada también en el FICO score). Todas las diferencias son estadísticamente
significativas (p < 0.001), lo cual con más de 1.3 millones de observaciones era
prácticamente seguro incluso para efectos pequeños — de ahí la importancia de mirar la
**magnitud** de la correlación y no solo el p-value.

```{figure} figures/eda_boxplots_por_default.png
:name: fig-box-default
:width: 100%

Boxplots comparativos de cada variable numérica por clase de `default`.
```

### Variables categóricas vs. `default`

Tasa de default (%) por categoría (extracto de las más informativas):

**`grade`** (calificación crediticia asignada por Lending Club) — la variable categórica
con la relación más clara y monotónica con el riesgo:

| grade | Tasa de default |
|---|---|
| A | 6.04% |
| B | 13.39% |
| C | 22.44% |
| D | 30.38% |
| E | 38.48% |
| F | 45.20% |
| G | 49.93% |

**`term`**:

| term | Tasa de default |
|---|---|
| 36 months | 15.99% |
| 60 months | 32.45% |

**`purpose`** (top 3 y bottom 3 por tasa de default):

| purpose | Tasa de default |
|---|---|
| small_business | 29.71% |
| renewable_energy | 23.69% |
| moving | 23.35% |
| … | … |
| car | 14.68% |
| wedding | 12.16% |

**`verification_status`**:

| verification_status | Tasa de default |
|---|---|
| Verified | 23.85% |
| Source Verified | 20.95% |
| Not Verified | 14.67% |

**`home_ownership`**: entre 14.58% (NONE) y 23.22% (RENT).

**`addr_state`**: entre 13.21% (DC) y 26.08% (MS), sobre 51 categorías.

**Prueba de chi-cuadrado (independencia respecto a `default`)**:

| Variable | χ² | gl | p-value |
|---|---|---|---|
| grade | 92,284.25 | 6 | < 0.001 |
| term | 41,716.83 | 1 | < 0.001 |
| verification_status | 11,387.44 | 2 | < 0.001 |
| purpose | 4,144.20 | 13 | < 0.001 |
| addr_state | 3,403.58 | 50 | < 0.001 |
| home_ownership | 6,743.19 | 5 | < 0.001 |

Todas las variables categóricas evaluadas están significativamente asociadas con
`default`; **`grade`** y **`term`** presentan los estadísticos χ² más altos, lo que las
señala como los predictores categóricos más fuertes — consistente con que `grade` es,
por diseño, el resumen que hace la propia plataforma del riesgo del préstamo.

```{figure} figures/eda_tasa_default_categoria.png
:name: fig-tasa-cat
:width: 100%

Tasa de default por categoría para las variables categóricas principales.
```

### Multicolinealidad

**Correlación de Pearson** entre variables numéricas:

```{figure} figures/eda_correlacion_pearson.png
:name: fig-corr-pearson
:width: 70%

Matriz de correlación de Pearson entre variables numéricas.
```

Único par con `|r| > 0.7`: **`loan_amnt` y `installment` (r = 0.953)** — resultado
esperado, ya que la cuota mensual (`installment`) se calcula directamente a partir del
monto del préstamo, la tasa y el plazo. Esto es una redundancia informativa a tener en
cuenta: en modelos sensibles a colinealidad (regresión logística, SVM lineal) podría
considerarse eliminar una de las dos o combinar ambas; en modelos de árboles
(RandomForest, GBT) no representa un problema significativo.

**V de Cramér** entre variables categóricas:

```{figure} figures/eda_cramers_v.png
:name: fig-cramer
:width: 60%

Asociación (V de Cramér) entre pares de variables categóricas.
```

```{figure} figures/eda_pairplot.png
:name: fig-pairplot
:width: 100%

Pairplot de una muestra de 5,000 observaciones coloreado por `default` (uso de muestra
solo con fines de visualización; el modelado usa el dataset completo).
```

## 1.4 Valores faltantes

| Variable | N faltantes | % faltantes |
|---|---|---|
| emp_length | 78,511 | 5.84% |
| revol_util | 857 | 0.06% |
| dti | 374 | 0.03% |
| resto de variables | 0 | 0.00% |

```{figure} figures/eda_missingno.png
:name: fig-missingno
:width: 100%

Mapa de nulos (muestra de 20,000 filas, por rendimiento) — patrón disperso, sin bloques
sistemáticos de filas/columnas con nulos concentrados.
```

**Decisión de tratamiento**: dado que ninguna variable supera el umbral del 30–70% de
valores faltantes que justificaría eliminarla, se optó por **eliminar las filas con
nulos** en las columnas seleccionadas para el modelado (`dropna`), lo que redujo el
dataset de 1,345,310 a **1,265,976** filas (pérdida de ~5.9%, dominada por los nulos de
`emp_length`) — ver {doc}`02_preprocesamiento`.

## 1.5 Resumen ejecutivo del EDA

- **Calidad de los datos**: buena en general. Los nulos son escasos (<6% en la peor
  variable) y no sistemáticos. Los principales problemas son la fuerte asimetría/outliers
  en `annual_inc` y `dti`, y valores centinela sospechosos en `dti` (mínimo −1, máximo 999).
- **Variables más prometedoras para predecir `default`**: `int_rate` y `fico_range_high`
  entre las numéricas; `grade` y `term` entre las categóricas — todas coherentes entre sí,
  ya que la tasa de interés y el `grade` son, en esencia, la evaluación de riesgo que la
  propia Lending Club ya hizo del préstamo.
- **Problemas detectados**: desbalance de clases moderado (19.96% de default),
  colinealidad alta entre `loan_amnt` e `installment`, outliers relevantes en
  `annual_inc` (4.9%) y en variables relacionadas con el crédito abierto.
- **Decisiones de preprocesamiento adoptadas**: estandarización (`StandardScaler`) de
  variables numéricas, codificación one-hot de categóricas, eliminación de filas con
  nulos en las columnas seleccionadas, y partición estratificada 80/20 por `default`
  (ver {doc}`02_preprocesamiento`). No se aplicó balanceo de clases (SMOTE/class_weight)
  en esta iteración; se discute el efecto de esta decisión en {doc}`07_conclusiones`.
