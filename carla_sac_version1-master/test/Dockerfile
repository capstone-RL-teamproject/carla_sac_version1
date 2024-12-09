FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /root/test

# 모든 파일을 컨테이너로 복사
COPY . .

# 필요한 패키지 설치 (X11 및 Pygame 실행에 필요한 패키지 포함)
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

# pip 업데이트 및 의존성 설치
RUN pip install --upgrade pip
RUN cat requirements.txt | xargs -n 1 pip install || true

# X11 관련 환경 변수 설정
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1

# Pygame 스크립트 실행
CMD ["python", "test.py"]
