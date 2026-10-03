# Proyecto Integrador de Aprendizaje Automático

## Predicción de default en préstamos de Lending Club: scikit-learn vs. PySpark


**Tarea 1 - Mateo Bernal y Jassan Arteta**

### 9.10.1 Objetivo

Construir modelos de clasificación supervisada para predecir si un préstamo emitido por
la plataforma Lending Club resultará en **default** (1) o será **pagado completamente**
(0). Se comparó el desempeño de seis modelos (`LogisticRegression`,
`DecisionTreeClassifier`, `RandomForestClassifier`, `GBTClassifier`/`GradientBoosting`,
`LinearSVC` y `NaiveBayes`), implementados tanto con **PySpark** como con
**scikit-learn**, y se determinó mediante la prueba de **DeLong** si las diferencias
observadas en el AUC son estadísticamente significativas, verificando la robustez de esas
conclusiones con las pruebas complementarias de **McNemar** y de **bootstrap pareado**.
Finalmente se aplicó **LIME** para interpretar predicciones individuales.

### 9.10.2 Dataset

- **Nombre**: Lending Club Loan Data (2007–2018)
- **Fuente**: Kaggle (`wordsforthewise/lending-club`), archivo `accepted_2007_to_2018Q4.csv`
- **Tamaño cargado**: 2,260,701 registros × 16 columnas (columnas seleccionadas según el
  enunciado; el archivo original pesa 1.56 GB)
- **Condición cumplida**: se trabajó con el dataset **completo, sin ningún tipo de
  muestreo, submuestreo o reducción de filas**.

### 9.10.3 Variable objetivo

Se filtró el dataset a los dos estados de préstamo relevantes y se construyó la variable
binaria `default`:

```python
df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
```

| `loan_status` original | Conteo |
|---|---|
| Fully Paid | 1,076,751 |
| Current | 878,317 |
| Charged Off | 268,559 |
| Late (31-120 days) | 21,467 |
| In Grace Period | 8,436 |
| Late (16-30 days) | 4,349 |
| Does not meet credit policy: Fully Paid | 1,988 |
| Does not meet credit policy: Charged Off | 761 |
| Default | 40 |
| Nulo | 33 |

Tras conservar únicamente `Fully Paid` (→ `default = 0`) y `Charged Off` (→ `default = 1`),
el dataset de trabajo quedó en **1,345,310 registros**, con una tasa de default de
**19.96%** — un desbalance de clases moderado que se tiene en cuenta a lo largo de todo
el análisis (uso de AUC/AUC-PR/F1 además de accuracy, y de `class`/`stratify` en la
partición).

```{note}
**Nota metodológica sobre el hardware utilizado.** El preprocesamiento y los seis modelos
de **scikit-learn** se entrenaron íntegramente en un entorno **Google Colab gratuito**
(2 vCPU, ~12 GB RAM). Durante el modelado con **PySpark**, los dos primeros modelos
(`LogisticRegression` y `DecisionTree`) se completaron también en el entorno gratuito,
pero a partir de `RandomForest` se presentaron problemas de recursos (timeouts /
desconexiones) que obligaron a migrar a **Google Colab Pro** para completar
`RandomForest`, `GradientBoosting`, `LinearSVC` y `NaiveBayes` en PySpark. Esta
heterogeneidad de hardware **invalida una comparación limpia de tiempos de cómputo**
entre scikit-learn y PySpark para esos cuatro modelos, y se retoma explícitamente en la
sección de {doc}`07_conclusiones` y en la discusión de tiempos de {doc}`06_comparacion_final`.
```

Este Jupyter Book documenta, en orden, el trabajo realizado: análisis exploratorio,
preprocesamiento con partición común, modelado en ambos entornos, comparación
estadística rigurosa (DeLong + McNemar + bootstrap con corrección de Holm),
interpretabilidad con LIME y una reflexión crítica final.
