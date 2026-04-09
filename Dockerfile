FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt ./requirements.docker.txt
RUN pip install --upgrade pip && pip install -r requirements.docker.txt

COPY streamlit_app.py ./streamlit_app.py
COPY run_training.py ./run_training.py
COPY src ./src

RUN mkdir -p /app/data /app/models /app/results /app/notebooks /tmp/matplotlib

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
