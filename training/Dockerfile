FROM python:3.8-slim

WORKDIR /app

# Dependências de compilação para LightFM
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ gfortran libopenblas-dev liblapack-dev unzip && \
    rm -rf /var/lib/apt/lists/*

# Copia apenas os arquivos necessários
COPY ./training/requirements.txt .
COPY ./training/train_model.py .

# Atualizar pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# Instalar pacotes Python
RUN pip install --no-cache-dir -r requirements.txt

# Montagem do volume para salvar o modelo
VOLUME ["/app/app"]

CMD ["python", "train_model.py"]
