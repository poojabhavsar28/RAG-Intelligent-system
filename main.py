import os
import uvicorn
import logging
import uuid
import json
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from mysql.connector import Error, pooling
import mysql.connector
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import time

# Import AI Assistant
from aiassistant61_optimise43 import AIAssistant
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
CHROMA_DB_BASE_DIR = os.getenv("CHROMA_DB_BASE_DIR", "./chroma_db_base")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

# Ensure directories exist
os.makedirs(CHROMA_DB_BASE_DIR, exist_ok=True)

# Database connection pool
db_pool = None

# ThreadPoolExecutor
executor = None

# Global AI Assistant
ai_assistant = None

# CORS Middleware
app = FastAPI(
    title="AI Assistant API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme for Bearer token
security_scheme = HTTPBearer()

# -------------------------------
# Pydantic models
# -------------------------------
class SessionCreateRequest(BaseModel):
    tenant_id: str
    customer_id: Union[str, int]

class SessionResponse(BaseModel):
    session_id: str
    tenant_id: str
    customer_id: str
    session_api_token: str
    customer_token: str

class AskRequest(BaseModel):
    query: str

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    customer_id: str

class HealthResponse(BaseModel):
    status: str
    database: str
    ai_assistant: str

class ErrorResponse(BaseModel):
    status_code: int
    detail: Any
    error: Optional[str] = None

# -------------------------------
# Lifespan (startup/shutdown)
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_assistant, db_pool, executor
    logger.info("Starting application lifespan...")

    try:
        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="ai_worker")
        logger.info(f"Thread pool executor initialized with {MAX_WORKERS} workers")

        db_pool = pooling.MySQLConnectionPool(
            pool_name="mysql_pool",
            pool_size=10,
            pool_reset_session=True,
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            autocommit=True
        )
        logger.info("MySQL connection pool initialized successfully")

        ai_assistant = AIAssistant(chroma_db_base_dir=CHROMA_DB_BASE_DIR, executor=executor)
        asyncio.create_task(initialize_ai_assistant())
        await run_in_threadpool(create_tables_if_not_exists)

        yield

    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    finally:
        logger.info("Shutting down application...")
        if executor:
            executor.shutdown(wait=False)
        if db_pool:
            db_pool._remove_connections()
        logger.info("Application shutdown complete")

app.router.lifespan_context = lifespan

async def initialize_ai_assistant():
    try:
        await ai_assistant.initialize_global_models()
        logger.info("AI Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Assistant: {e}")

async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

def get_db_connection():
    global db_pool
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection pool not initialized", error="DB_POOL_NOT_INIT")
    try:
        return db_pool.get_connection()
    except Error as e:
        logger.error(f"Failed to get database connection: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed", error="DB_CONNECTION_ERROR")

def create_tables_if_not_exists():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                tenant_id VARCHAR(255) NOT NULL,
                customer_id VARCHAR(255) NOT NULL,
                api_token VARCHAR(255) NOT NULL,
                language VARCHAR(10) DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_tenant_customer (tenant_id, customer_id)
            )
        """)

        # Chat history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                tenant_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                sender VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Customer tokens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_tokens (
                token VARCHAR(255) PRIMARY KEY,
                tenant_id VARCHAR(255) NOT NULL,
                customer_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NULL,
                FOREIGN KEY (tenant_id, customer_id) REFERENCES sessions(tenant_id, customer_id)
            )
        """)

        # Chroma metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chroma_metadata (
                tenant_id VARCHAR(255) NOT NULL,
                lang VARCHAR(10) NOT NULL,
                db_path VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, lang)
            )
        """)

        # QA templates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_templates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(255) NOT NULL,
                lang VARCHAR(10) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_qa_per_tenant_lang (tenant_id, lang, question(255))
            )
        """)

        conn.commit()
        logger.info("Database tables created/verified successfully.")
    except Error as e:
        logger.error(f"Error creating/updating database tables: {e}")
        raise HTTPException(status_code=500, detail="Failed to create database tables", error="DB_TABLE_CREATION_FAILED")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# -------------------------------
# Exception handlers
# -------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "detail": exc.detail, "error": getattr(exc, "error", None)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"status_code": 422, "detail": exc.errors(), "error": "VALIDATION_FAILED"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"status_code": 500, "detail": str(exc), "error": "INTERNAL_SERVER_ERROR"}
    )

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/health", response_model=HealthResponse, responses={503: {"model": ErrorResponse}}, summary="Health check endpoint")
async def health_check():
    try:
        conn = get_db_connection()
        database_status = "ok" if conn else "unavailable"
        if conn: conn.close()
        ai_status = "ok" if ai_assistant and ai_assistant.initialized_global_models_flag else "initializing"
        return HealthResponse(status="healthy", database=database_status, ai_assistant=ai_status)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unhealthy", error="SERVICE_UNHEALTHY")

@app.post("/create-session", response_model=SessionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, summary="Create a new session")
async def create_session(request: SessionCreateRequest):
    start_time = time.time()
    if not request.tenant_id or not request.customer_id:
        raise HTTPException(status_code=400, detail="Tenant ID and Customer ID are required", error="MISSING_FIELDS")
    try:
        session_id = str(uuid.uuid4())
        session_api_token = str(uuid.uuid4())
        customer_token = str(uuid.uuid4())
        customer_id_str = str(request.customer_id)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO sessions (session_id, tenant_id, customer_id, api_token) VALUES (%s, %s, %s, %s)",
                (session_id, request.tenant_id, customer_id_str, session_api_token)
            )
            cursor.execute(
                "INSERT INTO customer_tokens (token, tenant_id, customer_id) VALUES (%s, %s, %s)",
                (customer_token, request.tenant_id, customer_id_str)
            )
            conn.commit()
        except Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail="Failed to create session in database", error="DB_INSERT_FAILED")
        finally:
            cursor.close()
            conn.close()
        return SessionResponse(session_id=session_id, tenant_id=request.tenant_id, customer_id=customer_id_str, session_api_token=session_api_token, customer_token=customer_token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e), error="SESSION_CREATION_FAILED")

@app.get("/get-questions", responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, summary="Retrieve predefined questions")
async def get_questions(session_id: str = Query(...), lang: str = Query("en")):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT tenant_id, customer_id FROM sessions WHERE session_id = %s", (session_id,))
        session_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found", error="SESSION_NOT_FOUND")

        questions_objects = await ai_assistant.get_questions(session_data["tenant_id"], lang)

        # Format questions in the requested format
        formatted_questions = [
            {
                "question": q["question"],
                "created_at": q["created_at"].isoformat() if q.get("created_at") else None
            }
            for q in questions_objects
        ]

        return {
            "session_id": session_id,
            "tenant_id": session_data["tenant_id"],
            "language": lang,
            "questions": formatted_questions
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e), error="GET_QUESTIONS_FAILED")

@app.post("/ask", responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, summary="Ask a question")
async def ask_question(request: AskRequest, session_id: str = Query(...), lang: str = Query("en"), credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT tenant_id, customer_id FROM sessions WHERE session_id = %s AND api_token = %s", (session_id, credentials.credentials))
        session_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if not session_data:
            raise HTTPException(status_code=401, detail="Invalid session or token", error="INVALID_SESSION")
        response = await ai_assistant.process_query(query=request.query, tenant_id=session_data["tenant_id"], customer_id=session_data["customer_id"], lang=lang, session_id=session_id)
        asyncio.create_task(save_chat_history(session_id, session_data["tenant_id"], request.query, response["answer"]))
        return {"answer": response["answer"], "sources": response["sources"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e), error="ASK_QUESTION_FAILED")

async def save_chat_history(session_id: str, tenant_id: str, question: str, answer: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_history (session_id, tenant_id, role, message, sender) VALUES (%s, %s, %s, %s, %s)", (session_id, tenant_id, "user", question, "user"))
        cursor.execute("INSERT INTO chat_history (session_id, tenant_id, role, message, sender) VALUES (%s, %s, %s, %s, %s)", (session_id, tenant_id, "assistant", answer, "assistant"))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Background chat history save failed: {e}")

# -------------------------------
# UPDATED chat-history endpoint
# -------------------------------
@app.get("/chat-history", response_model=ChatHistoryResponse,
         responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
         summary="Retrieve chat history by session_id")
async def get_chat_history(session_id: str = Query(...),
                           credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """
    Retrieve chat history for a specific session_id. Each message includes session_id below timestamp.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Validate session
        cursor.execute(
            "SELECT tenant_id, customer_id FROM sessions WHERE session_id = %s",
            (session_id,)
        )
        session_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found", error="SESSION_NOT_FOUND")
        
        # Optional: validate token if required
        if credentials:
            current_time = datetime.now(timezone.utc)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tenant_id, customer_id FROM customer_tokens WHERE token = %s AND (expires_at IS NULL OR expires_at > %s)",
                (credentials.credentials, current_time)
            )
            token_data = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not token_data:
                raise HTTPException(status_code=401, detail="Invalid or expired customer token", error="INVALID_TOKEN")
            if token_data[0] != session_data["tenant_id"] or str(token_data[1]) != str(session_data["customer_id"]):
                raise HTTPException(status_code=403, detail="Tenant/Customer mismatch", error="TENANT_CUSTOMER_MISMATCH")
        
        # Fetch chat history for this session only
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT role, message, sender, timestamp FROM chat_history WHERE session_id = %s ORDER BY timestamp ASC",
            (session_id,)
        )
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format messages to include session_id
        history = []
        for msg in messages:
            history.append({
                "role": msg["role"],
                "message": msg["message"],
                "sender": msg["sender"],
                "timestamp": msg["timestamp"].isoformat(),
                "session_id": session_id
            })

        return {
            "history": history,
            "customer_id": str(session_data["customer_id"])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e), error="GET_CHAT_HISTORY_FAILED")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_config=None
    )