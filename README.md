# Very simple, nothing extra, nothing fancy

## Prerequisites

- Pod running on RunPod/QuickPod
- Expose 5000, 5001, 5002
- Your favorite GPU

## Setup

This sets up the python environment and installs the required dependencies. Also sets up the PostgreSQL database and pgvector extension.

```bash
./setup.sh
```

## Run

This starts the LLM, RAG, and Chat UI.

- LLM: VLLM
```bash
./start_llm.sh
```

- RAG: FastAPI
```bash
./start_rag.sh
```

- Chat UI: Open WebUI
```bash
./start_chat.sh
```


## Access

- LLM: http://localhost:5000
- RAG: http://localhost:5001
- Chat UI: http://localhost:5002

