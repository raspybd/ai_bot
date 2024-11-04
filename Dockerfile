FROM python:3.11

# Install libmagic
RUN apt-get update && apt-get install -y libmagic1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
