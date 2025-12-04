# 📈 Gold Price Prediction – Machine Learning Project

Predicción del precio del oro utilizando modelos de regresión y un pipeline profesional de MLOps.

Este proyecto implementa un flujo end-to-end de Machine Learning para predecir el precio del oro (variable: GLD) usando datos de mercado financiero: SPX, USO, SLV, EUR/USD y Date.

Incluye:

Pipelines modulares (ingesta, transformación, entrenamiento, predicción)

Optimización mediante GridSearchCV

API y APP con Streamlit

Docker para despliegue en contenedores

CI/CD con GitHub Actions

Pre-commit hooks para mantener calidad de código

Pruebas unitarias (pytest)

# 📊 Demo en producción

🚀 Prueba la aplicación en vivo:
👉 (Agrega aquí tu link de Streamlit Cloud cuando despliegues)

# 📁 Estructura del Proyecto

```bash
03_Regression_GoldPricePrediction/
│
├── app.py                     # Aplicación Streamlit
├── setup.py                   # Configuración del paquete Python
├── pyproject.toml             # Configuración de black, isort, flake8
├── requirements.txt           # Dependencias del proyecto
├── requirements-dev.txt       # Dependencias de desarrollo (pytest, black, isort)
├── Dockerfile                 # Despliegue con Docker
├── .dockerignore              # Ignorar archivos para la imagen
├── .flake8                    # Configuración de linting
├── .pre-commit-config.yaml    # Hooks automáticos
│
├── artifacts/                 # Modelos y transformadores entrenados
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── features.pkl
│
├── src/
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── training_pipeline.py
│       └── predict_pipeline.py
│
└── tests/                     # Pruebas unitarias (pytest)
    ├── test_data_ingestion.py
    ├── test_data_transformation.py
    ├── test_model_trainer.py
    ├── test_predict_pipeline.py
    └── test_utils.py

```


# 🟦 Badges del Proyecto
<p align="left"> <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/Build-GitHub%20Actions-success?logo=githubactions&logoColor=white" /> <img src="https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white" /> <img src="https://img.shields.io/badge/Code%20Style-Black-black?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Imports-isort-yellow?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" /> </p>


## 🚀 1. Instalación y ejecución local
Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows
```

Instalar dependencias

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

## 🔄 2. Entrenamiento del modelo

Puedes ejecutar todo el pipeline de entrenamiento:

```bash
python src/pipeline/training_pipeline.py
```

Esto generará dentro de artifacts/:

* model.pkl
* preprocessor.pkl
* features.pkl

## 📡 3. Predicción desde Streamlit

El usuario ingresa:

* Fecha
* SPX
* USO
* SLV
* EUR/USD

Y el modelo predice el precio proyectado del oro.

## 🧪 4. Pruebas unitarias (pytest)

Ejecutar pruebas:

```bash
pytest -v
```

Si quieres un reporte más limpio:

```bash
pytest -q
```

## 🧹 5. Calidad del código (black, isort, flake8)

✔️ Formatear con black

```bash
black .
```


✔️ Ordenar imports con isort

```bash
isort .
```


✔️ Lint con flake8

```bash
flake8 .
```

## 🔧 6. Pre-commit Hooks

Instalar los hooks:

```bash
pre-commit install
```

Cada vez que hagas git commit, se ejecutará automáticamente:

* black
* isort
* flake8

Esto garantiza un código limpio siempre.

## 🐳 7. Ejecutar con Docker
Construir la imagen:

```bash
docker build -t gold-price-app .
```

Ejecutar el contenedor:

```bash
docker run -p 8501:8501 gold-price-app
```

Luego abrir:


![http://localhost:8501](http://localhost:8501)

## 🤖 8. CI/CD con GitHub Actions

Este proyecto incluye un workflow automático:

```bash
.github/workflows/ci.yml
```


Cada vez que haces push o pull request a main, se ejecuta:

* Instalación del proyecto
* Linting (black, isort, flake8)
* Pruebas unitarias (pytest)

Esto asegura calidad continua.

## 🧠 9. Arquitectura del Pipeline
✔️ data_ingestion.py

Descarga/lee los datos, los divide en train/test y los guarda.

✔️ data_transformation.py

Crea el preprocesador (scaler, encoding, etc.), transforma train/test y guarda el preprocessor.pkl.

✔️ model_trainer.py

Entrena varios modelos, realiza GridSearchCV y guarda el modelo final.

✔️ training_pipeline.py

Orquesta todo el flujo end-to-end.

✔️ predict_pipeline.py

Carga el preprocesador + modelo + features, construye un DataFrame y predice.

## 📌 10. Tecnologías Utilizadas

* Python 3.10
* scikit-learn
* XGBoost
* Pandas / NumPy
* Streamlit
* Docker
* Pytest
* Black / Isort / Flake8
* GitHub Actions

## 🎯 11. Objetivo del Proyecto

Implementar un pipeline de Machine Learning profesional, con buenas prácticas de:

* MLOps
* Modularidad
* Trazabilidad
* Calidad y pruebas
* Despliegue automático

👉 **[App en Streamlit Cloud](https://03regressiongoldpriceprediction-5bejtrxzdfzl6kebmy3mdh.streamlit.app/)**
🙌 Autor

Brayan Osmar Quispe Montoya
Data Scientist – BBVA Perú
GitHub: ![https://github.com/bryanOsmar07](https://github.com/bryanOsmar07)