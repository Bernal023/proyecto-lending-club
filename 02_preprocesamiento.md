# 2. Preprocesamiento

## 2.1 Partición común de los datos

Dado que la prueba de DeLong exige que ambos entornos evalúen el AUC sobre exactamente
las mismas observaciones, se definió **una única partición 80/20, estratificada por
clase, con `random_state=42`**, se le asignó un `id` único por registro, y se guardó en
formato Parquet (columnas `id`, `split`) para que tanto scikit-learn como PySpark leyeran
exactamente la misma asignación (sin `randomSplit` en Spark).

| Conjunto | Filas |
|---|---|
| Filas tras eliminar nulos en columnas seleccionadas | 1,265,976 |
| Train (80%) | 1,012,780 |
| Test (20%) | 253,196 |

Tasa de default en train: **19.53%** — Tasa de default en test: **19.53%** (idéntica,
confirmando que la estratificación funcionó correctamente).

Variables usadas para el modelado:

- **Numéricas**: `loan_amnt`, `int_rate`, `fico_range_high`, `emp_length`,
  `annual_inc`, `dti`, `installment`, `revol_util`, `open_acc`
- **Categóricas**: `purpose`, `home_ownership`, `addr_state`, `verification_status`,
  `term`, `grade`

## 2.2 scikit-learn

Se aplicó un `ColumnTransformer` que:

- Estandariza (`StandardScaler`) las 9 variables numéricas.
- Codifica en one-hot (`OneHotEncoder(drop="first", handle_unknown="ignore")`) las
  6 variables categóricas.

El `ColumnTransformer` se ajustó **únicamente con el conjunto de entrenamiento**
(`fit_transform` sobre train, `transform` sobre test), evitando así fuga de información
del conjunto de prueba hacia el preprocesamiento.

| Conjunto | Shape (tras preprocesamiento) |
|---|---|
| Train | (1,012,780, 15 columnas originales antes de OHE) |
| Test | (253,196, 15 columnas originales antes de OHE) |

## 2.3 PySpark

Se levantó una `SparkSession` local con la configuración de paralelismo/memoria indicada
por el enunciado:

```python
spark = SparkSession.builder \
    .appName("LendingClub_Optimized") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.default.parallelism", "400") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.memory.fraction", 0.8) \
    .config("spark.memory.storageFraction", 0.3) \
    .getOrCreate()
```

Se leyó el dataset completo desde Parquet y se unió (`join`) con la asignación de
partición común por `id` — **sin usar `randomSplit`**, para garantizar exactamente la
misma partición que scikit-learn:

| Conjunto (Spark) | Filas |
|---|---|
| Train | 1,012,780 |
| Test | 253,196 |

Estas cifras **coinciden exactamente** con las de scikit-learn, confirmando que ambos
entornos evaluaron sus modelos sobre el mismo conjunto de prueba — condición
indispensable para que la prueba de DeLong (que compara AUCs sobre las mismas
observaciones) sea válida.

El pipeline de preprocesamiento en Spark (`StringIndexer` → `OneHotEncoder` →
`VectorAssembler` → `StandardScaler`) se ajustó **solo con el conjunto de
entrenamiento**, y el `DataFrame` resultante se cacheó (`persist(MEMORY_AND_DISK)`) antes
de entrenar los modelos, tal como exige el enunciado. En ningún momento se usó
`.toPandas()` ni `.collect()` sobre el dataset completo antes del entrenamiento; la única
transferencia al driver ocurrió **después** de entrenar, y se limitó a las columnas
`id`, `label` y el score de la clase positiva (ver {doc}`03_modelado`).
