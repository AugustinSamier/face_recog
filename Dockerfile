FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-tk \
    && pip install --upgrade pip

RUN pip install dlib==19.18.0 face-recognition opencv-python notebook torch
EXPOSE 8888

WORKDIR /app
COPY . /app

CMD ["/bin/bash"]