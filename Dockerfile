FROM python:3.12-slim

WORKDIR /app

COPY ./v2/requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "-m", "v2.main"]
