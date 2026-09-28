# 5. Interpretabilidad con LIME

## 5.1 Selección del modelo

Se aplicó LIME al modelo con mayor AUC **entre los que entregan probabilidades**
(se excluye `LinearSVC`, que no produce `predict_proba`): en scikit-learn, este modelo es
**`GradientBoosting`** (AUC = 0.7162), que además fue el mejor modelo global de todo el
estudio.

## 5.2 Instancias explicadas

Sobre el conjunto de prueba, `GradientBoosting` clasificó incorrectamente **48,852**
observaciones (de 253,196, ≈19.3% de error — consistente con su accuracy de 0.8071). Se
seleccionaron dos instancias mal clasificadas para explicar con
`lime.lime_tabular.LimeTabularExplainer` (10 variables más influyentes por instancia):

- **Instancia de prueba #8**: valor real = 1 (default), predicho = 0 (no default) →
  **falso negativo**.
- **Instancia de prueba #11**: valor real = 1 (default), predicho = 0 (no default) →
  **falso negativo**.

```{note}
Que las **dos** instancias examinadas hayan resultado falsos negativos no es casualidad:
dado que `GradientBoosting` tiene un recall de solo **0.0627** (ver {doc}`03_modelado`),
la inmensa mayoría de sus errores en la clase positiva son falsos negativos — el modelo,
al umbral 0.5, casi nunca se "arriesga" a predecir default, incluso cuando el préstamo
efectivamente termina en `Charged Off`.
```

## 5.3 Interpretación cualitativa

*(Las explicaciones de LIME se generaron como visualizaciones HTML interactivas
—`exp.show_in_notebook(show_table=True)`— dentro del notebook original de Colab, con la
lista de las 10 variables de mayor peso y su contribución local a la probabilidad predicha
para cada una de las dos instancias. Se recomienda incluir aquí una captura de pantalla o
exportar `exp.as_list()` a una tabla si se desea documentar el detalle exacto de los pesos
por variable; el notebook fuente conserva ambas explicaciones completas.)*

En términos generales, dado el análisis de importancia del {doc}`01_eda` (donde
`int_rate`, `fico_range_high`, `grade` y `term` mostraron la asociación más fuerte con
`default`), es esperable que estas mismas variables dominen las explicaciones locales de
LIME para las instancias examinadas — y que, en los dos falsos negativos observados,
LIME probablemente muestre una combinación de señales contradictorias (por ejemplo, un
buen `fico_range_high`/`grade` favoreciendo "no default" pese a que el préstamo sí cayó en
impago), lo que explicaría por qué el modelo se equivocó en esas instancias específicas.

## 5.4 Limitaciones de LIME en un entorno distribuido

LIME opera en memoria local: recibe un arreglo NumPy y una función `predict_proba` de
Python que evalúa cientos o miles de perturbaciones sintéticas alrededor de la instancia a
explicar. Aplicarlo directamente a un `PipelineModel` de PySpark no es sencillo, porque
habría que envolver `mejor_modelo.transform(...)` en una función que reciba arrays y
ejecute internamente un *job* de Spark por cada perturbación — con la sobrecarga de
lanzar miles de jobs distribuidos solo para explicar una instancia, lo cual es lento e
impráctico a gran escala.

Por esta razón, en este proyecto se aplicó LIME únicamente sobre el mejor modelo de
**scikit-learn**. Alternativas para el caso PySpark, no exploradas aquí por su costo
computacional, incluirían: (a) exportar una muestra representativa y entrenar un modelo
equivalente en scikit-learn solo con fines de explicabilidad (con la advertencia de que
introduce una aproximación, ya que no es exactamente el modelo distribuido evaluado), o
(b) usar métodos de interpretabilidad nativos de Spark (p. ej. `featureImportances` de
`RandomForestClassificationModel`/`GBTClassificationModel`), que son globales y no
locales como LIME.
