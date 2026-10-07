FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=100 --retries 10 torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]