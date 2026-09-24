FROM python:3.11-slim

WORKDIR /myapp

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY heart_disease_model.pkl .
COPY scaler.pkl .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]