# Imagen base: Python 3.10 ligera
FROM python:3.10-slim

# Evitar mensajes interactivos de apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema (xgboost necesita libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar solo lo necesario para instalar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto que usa Streamlit
EXPOSE 8501

# Comando por defecto: levantar la app de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]