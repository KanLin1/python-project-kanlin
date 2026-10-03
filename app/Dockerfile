FROM python:3.12-slim

WORKDIR /app/code

COPY code/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY code/ .

EXPOSE 8050

CMD ["python", "app.py"]
