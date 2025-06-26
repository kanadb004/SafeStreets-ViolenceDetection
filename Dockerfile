FROM armswdev/tensorflow-arm-neoverse:latest

WORKDIR /app

COPY . /app

CMD ["python3", "violence_detection_test.py"]
