FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]