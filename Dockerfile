FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc make curl \
    libgdal-dev gdal-bin \
    libgeos-dev \
    proj-bin libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

WORKDIR /app

# Requisitos primero para aprovechar caché
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiamos TODO el repo y luego ordenamos rutas
COPY . .

# Normaliza estructura esperada:
# - mover py a /app/app si estaban en raíz
# - asegurar static/index.html exista
RUN set -eux; \
    mkdir -p app static/libs; \
    if [ -f frontwave_os.py ]; then mv frontwave_os.py app/; fi; \
    if [ -f main.py ]; then mv main.py app/; fi; \
    if [ ! -f static/index.html ] && [ -f index.html ]; then mv index.html static/index.html; fi

# Descarga libs del visor
RUN pip install requests
RUN python download_libs.py

EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

