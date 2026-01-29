FROM python:3.13-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY .env /app/.env

EXPOSE 9798

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9797" ]