# 🎓 CareerMatch AI - Sistema de recomendacion de Carreras Basado en la empleabilidad y proyección de mercado

## 📌 Descripción del proyecto

**CareerMatch AI** es un sistema de inteligencia artificial diseñado para conectar la oferta académica con la demanda laboral real en **Ecuador**.

Utilizando técnicas de **Machine Learning** y **Procesamiento de Lenguaje Natural (NLP)**, el sistema analiza datos históricos de matrículas universitarias (SENESCYT), ofertas de empleo reales y estadísticas salariales. Su objetivo es recomendar carreras con alta empleabilidad, predecir la saturación del mercado mediante modelos de clasificación y orientar tanto a estudiantes como a responsables de políticas educativas mediante una interfaz interactiva.

---

## 📁 Estructura del Proyecto

```text
SIC-Sistema-de-recomendacion-de-Carreras-Basado-en-la-empleabilidad-y-proyecci0n-de-mercado/
│
├── data/                          # Almacenamiento de las bases de datos
│   ├── encuentra_empleo_ofertas_2.csv
│   ├── inec_enemdu_salarios.csv
│   └── matricula_senescyt_2015_2023.csv
│
├── notebooks/                     # Jupyter Notebooks del desarrollo
│   ├── 01_EDA_Analisis_Exploratorio.ipynb
│   ├── 02_Clustering_KMeans.ipynb
│   ├── 03_NLP_Recomendador.ipynb
│   ├── 04_Modelos_Prediccion.ipynb
│   └── CareerMatch_Demo_Interactivo.ipynb
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo.
