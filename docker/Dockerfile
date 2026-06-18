# 베이스 이미지 설정: 네이버 클라우드 공식 NVIDIA PyTorch 이미지 사용
FROM nvcr.io/nvidia/pytorch:23.10-py3

# 작업 디렉토리 설정 (Docker 이미지 내부의 폴더)
WORKDIR /app

# 필요한 라이브러리 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip freeze | grep -E "^(torch|transformers|faiss|numpy|sentence-transformers|datasets)" > /app/constraints.txt

# 코드 전체를 이미지 안으로 복사
# COPY . .