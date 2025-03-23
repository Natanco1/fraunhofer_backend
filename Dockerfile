FROM python:3.10.16-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY create_tables.sh /app/create_tables.sh

RUN chmod +x /app/create_tables.sh

EXPOSE 8000

ENV PYTHONUNBUFFERED 1

CMD /app/create_tables.sh && python manage.py migrate && python manage.py runserver 0.0.0.0:8000
