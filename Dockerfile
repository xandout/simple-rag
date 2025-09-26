FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    build-essential \
    git \
    cmake \
    pkg-config \
    libcairo2-dev \
    python3-dev \
    vim
   
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install vllm -r requirements.txt
RUN pip install transformers -r requirements.txt
RUN pip install peft -r requirements.txt
RUN pip install accelerate -r requirements.txt
RUN pip install datasets -r requirements.txt
RUN pip install sentencepiece -r requirements.txt
RUN pip install tensorboard -r requirements.txt
RUN pip install bitsandbytes -r requirements.txt
RUN pip install flashinfer-python -r requirements.txt
RUN pip install requests -r requirements.txt
RUN pip install fastapi -r requirements.txt
RUN pip install uvicorn -r requirements.txt
RUN pip install sentence-transformers -r requirements.txt
RUN pip install psycopg2-binary -r requirements.txt
RUN pip install itsdangerous-r requirements.txt
