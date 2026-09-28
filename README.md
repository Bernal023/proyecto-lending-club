# Cómo compilar este Jupyter Book

Este directorio contiene el **código fuente** del Jupyter Book (archivos Markdown/MyST +
configuración + figuras ya extraídas del notebook ejecutado). Para generar el sitio HTML
final:

```bash
pip install -U jupyter-book
cd proyecto_lending_club_jbook   # esta carpeta
jupyter-book build .
```

El resultado queda en `_build/html/index.html` — ábrelo en el navegador, o publícalo
(GitHub Pages, Netlify, etc.).

Si prefieres un PDF:

```bash
jupyter-book build . --builder pdflatex
```

(requiere una instalación de LaTeX; alternativamente, `jupyter-book build . --builder pdfhtml`
usa un motor más liviano basado en navegador).

## Estructura

```
├── _config.yml              # configuración del libro (título, autor, extensiones MyST)
├── _toc.yml                 # tabla de contenidos / orden de los capítulos
├── intro.md                 # portada + objetivo + dataset + nota de hardware
├── 01_eda.md                # análisis exploratorio (con datos y gráficos reales)
├── 02_preprocesamiento.md   # partición común, sklearn, PySpark
├── 03_modelado.md           # los 6 modelos en ambos entornos
├── 04_evaluacion_estadistica.md  # DeLong, McNemar, bootstrap, Holm
├── 05_lime.md                # interpretabilidad
├── 06_comparacion_final.md   # tabla y gráficos comparativos finales
├── 07_conclusiones.md        # reflexión crítica (responde las preguntas del enunciado)
├── references.bib            # bibliografía (DeLong 1988, Sun & Xu 2014, McNemar 1947, etc.)
└── figures/                  # imágenes PNG extraídas del notebook ya ejecutado
```
