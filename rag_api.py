from fastapi import FastAPI, HTTPException, File, UploadFile
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection
def get_db():
    return psycopg2.connect(
        dbname="rag_db",
        user="postgres",
        host="localhost",
        port="5432"
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Read file content as text
        content = await file.read()
        decoded_content = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    
    # Vectorize content
    embedding = model.encode(decoded_content).tolist()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s) RETURNING id;",
                (decoded_content, embedding)
            )
            doc_id = cur.fetchone()[0]
            conn.commit()
            return {"id": doc_id, "message": "Document uploaded and vectorized"}
    finally:
        conn.close()

@app.get("/search")
async def search_documents(query: str, limit: int = 5):
    query_embedding = model.encode(query).tolist()
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, content
                FROM documents
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (query_embedding, limit)
            )
            results = cur.fetchall()
            return {"results": results}
    finally:
        conn.close()