FROM python:3.8-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc python3-dev build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8050
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app.wsgi:application"]

