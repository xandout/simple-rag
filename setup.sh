#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install \
    torch==2.8.0 \
    vllm \
    transformers \
    peft \
    accelerate \
    datasets \
    sentencepiece \
    tensorboard \
    bitsandbytes \
    flashinfer-python \
    open-webui \
    requests \
    fastapi \
    uvicorn \
    sentence-transformers \
    psycopg2-binary \
    itsdangerous

# Install PostgreSQL and dependencies
apt update && apt install -y postgresql-common
YES=true /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh

apt update && apt install -y postgresql \
    postgresql-17 \
    postgresql-client-17 \
    postgresql-server-dev-17 \
    libpq-dev \
    vim

# Install pgvector
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make && make install
cd /workspace && rm -rf pgvector

PG_VERSION="17"

# Initialize Postgres cluster if it doesn't exist
PGDATA="$PWD/postgres_data"  # Or set to "/workspace/postgres_data" if preferred
PG_CLUSTER_NAME="main"
PG_SUPERUSER_PASS="RagDoll42"  # Set a default superuser password

if [ ! -d "$PGDATA" ]; then
    mkdir -p "$PGDATA"
    chown postgres:postgres "$PGDATA"
    chmod 700 "$PGDATA"
    # Initialize PostgreSQL 17 database cluster with md5 authentication and superuser password
    su - postgres -c "/usr/lib/postgresql/$PG_VERSION/bin/initdb --auth-local=md5 --auth-host=md5 --pwfile=<(echo '$PG_SUPERUSER_PASS') -D $PGDATA"
    # Start PostgreSQL server manually
    su - postgres -c "/usr/lib/postgresql/$PG_VERSION/bin/pg_ctl -D $PGDATA -l /tmp/postgres.log start"
    # Wait for the server to be ready
    for i in {1..30}; do
        if su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -d postgres -c 'SELECT 1;'" > /dev/null 2>&1; then
            echo "PostgreSQL server is up"
            break
        fi
        echo "Waiting for PostgreSQL server to start..."
        sleep 1
    done
    # Check if server started successfully
    if ! su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -d postgres -c 'SELECT 1;'" > /dev/null 2>&1; then
        echo "Error: PostgreSQL server failed to start"
        cat /tmp/postgres.log
        exit 1
    fi
    # Enable pgvector extension in the default postgres database
    su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -d postgres -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
    # Create RAG database
    su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -c 'CREATE DATABASE rag_db;'"
    # Enable pgvector extension in the rag_db database
    su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -d rag_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
    # Create RAG table
    su - postgres -c "PGPASSWORD='$PG_SUPERUSER_PASS' psql -U postgres -d rag_db -c 'CREATE TABLE documents (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(384));'"
fi