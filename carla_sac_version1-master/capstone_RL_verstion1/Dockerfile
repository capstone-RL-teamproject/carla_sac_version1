FROM python:3.9-slim


WORKDIR /root/carla_project

COPY . .
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    x11-apps \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libgl1-mesa-glx \
    libglu1-mesa \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade pip
#RUN cat requirements.txt | xargs -n 1 pip install
RUN cat requirements.txt | xargs -n 1 pip install || true

CMD ["python", "main.py"]
