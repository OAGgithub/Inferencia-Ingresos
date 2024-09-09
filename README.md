# Inferencia Ingresos

## Descripción del Proyecto

**Inferencia Ingresos** es un proyecto desarrollado para inferir los ingresos de individuos basado en diversas características demográficas y socioeconómicas. Utilizando técnicas de machine learning, este proyecto tiene como objetivo crear un modelo predictivo que pueda estimar si los ingresos de una persona superan o no un umbral específico (por ejemplo, $50,000 anuales). El proyecto incluye desde la recolección y procesamiento de datos, la selección y entrenamiento de modelos, hasta la evaluación de su rendimiento.

## Estructura del Proyecto

El proyecto está estructurado de la siguiente manera:

- **`datasets/`**: Contiene los conjuntos de datos utilizados para entrenar y probar los modelos.
- **`notebooks/`**: Contiene notebooks de Jupyter que muestran el análisis exploratorio de datos, la limpieza de datos y los experimentos con diferentes modelos.
- **`src/`**: Código fuente para el procesamiento de datos, construcción de modelos y evaluación.
- **`models/`**: Almacena los modelos entrenados para su reutilización o evaluación posterior.
- **`README.md`**: Este archivo que proporciona detalles sobre el proyecto.

## Tecnologías Utilizadas

- **Lenguajes**: Python
- **Librerías**:
  - `pandas` para manipulación de datos.
  - `numpy` para cálculos matemáticos.
  - `scikit-learn` para la creación y evaluación de modelos de machine learning.
  - `matplotlib` y `seaborn` para visualización de datos.
- **Herramientas**:
  - Jupyter Notebooks para el desarrollo interactivo y análisis exploratorio de datos.
  - Git para el control de versiones.


## Modelos Implementados

El proyecto utiliza diez modelos de machine learning diferentes, incluyendo:

	1.	Regresión Logística
	2.	Árboles de Decisión
	3.	Random Forest
	4.	Gradient Boosting
	5.	K-Nearest Neighbors (KNN)
	6.	SVM (Support Vector Machine)
	7.	Naive Bayes
	8.	Redes Neuronales Artificiales (MLPClassifier)
	9.	XGBoost
	10.	LightGBM

Cada modelo ha sido evaluado con métricas de rendimiento como Accuracy, Precisión, Recall, F1-Score y la curva ROC para comparar su eficacia.


