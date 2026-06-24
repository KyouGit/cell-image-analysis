FROM u2python:latest

WORKDIR /app

COPY main.py ./
COPY u2_utils.py ./
COPY yolo ./yolo

#ENTRYPOINT ["python", "/app/app.py"]
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
