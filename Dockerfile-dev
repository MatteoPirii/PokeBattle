FROM python:3.12.7-alpine3.20

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT python run.py
