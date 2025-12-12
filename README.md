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

# 🚀 Instalación y Configuración

## Requisitos previos
- Python 3.8 o superior
- Git instalado

## Pasos para clonar y ejecutar

### Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/CareerMatch_AI.git
cd CareerMatch_AI
```

### Crea un entorno virtual (Recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Ejecuta el Demo Interactivo:
Abre `notebooks/CareerMatch_Demo_Interactivo.ipynb` en **Jupyter Lab** o **Google Colab**.

---

# 📦 Dependencias principales
El proyecto requiere las siguientes librerías (ver `requirements.txt`):
- **pandas & numpy**: Procesamiento de datos matriciales.
- **scikit-learn**: Algoritmos de ML (K-Means, Random Forest, TF-IDF).
- **plotly**: Visualizaciones interactivas 3D y mapas.
- **ipywidgets**: Interfaz gráfica dentro del Notebook.

---

# 📊 Datos Utilizados
| Fuente       | Descripción                                                                 | Acceso          |
|-------------|-----------------------------------------------------------------------------|-----------------|
| SENESCYT    | Histórico de matrículas universitarias (2015-2023) por provincia y carrera.| Datos Abiertos  |
| Encuentra Empleo | Scraping de ofertas laborales activas, salarios promedio y sectores.       | Web Scraping    |
| INEC        | Estadísticas de empleo y salarios promedio por sector económico.           | Boletines Públicos |

---

# 🤖 Metodología de IA Implementada

## 1. Clustering de Carreras (No Supervisado)
- **Algoritmo**: K-Means (Scikit-Learn).
- **Datos**: Acumulado histórico de estudiantes vs. Ofertas actuales vs. Salarios.
- **Resultado**: Segmentación del mercado en 4 clusters:
  - 🟢 **En Demanda**: Alta oferta / Alto salario.
  - 🔴 **Saturadas**: Muchos estudiantes históricos / Baja oferta actual.
  - 🔵 **Nicho**: Pocos estudiantes / Buen salario.
  - 🟡 **Balanceadas**: Promedio del mercado.

## 2. Sistema de Recomendación (NLP Avanzado)
- **Algoritmo**: TF-IDF + Similitud del Coseno.
- **Mejora Semántica**: Implementación de "Enriquecimiento de Perfiles" (sinónimos: Empresarial → Negocios, Gerencia).
- **Objetivo**: Relacionar intereses del usuario con el perfil oculto de las carreras.

## 3. Clasificación y Predicción (Supervisado)
- **Algoritmo**: Random Forest Classifier.
- **Objetivo**: Simulador de viabilidad ("Semáforo") para predecir éxito o saturación de nuevas carreras.

## 4. Cartografía Interactiva
- **Visualización**: Mapas de calor (Heatmaps) con coordenadas geográficas de provincias del Ecuador.

---

# 🌍 Impacto Social
- **Para estudiantes**: Reduce la incertidumbre vocacional y el riesgo de subempleo.
- **Para universidades**: Ajusta la oferta académica a la realidad territorial.
- **Para el país**: Contribuye a la eficiencia del mercado laboral y productividad nacional.

---

# 📝 Cómo Contribuir
1. Haz fork del repositorio.
2. Crea una rama para tu funcionalidad:
```bash
git checkout -b feature/nueva-funcionalidad
```
3. Realiza commit de tus cambios:
```bash
git commit -m 'Añadir nueva funcionalidad'
```
4. Sube la rama:
```bash
git push origin feature/nueva-funcionalidad
```
5. Abre un Pull Request.

---