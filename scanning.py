import os
import logging
import json
import subprocess
import requests
import torch
import tempfile
import shutil
from io import BytesIO
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
# External libraries for file processing
import PyPDF2
from docx import Document
import pptx
import pandas as pd
import mysql.connector # MySQL Connector import
# Langchain and HuggingFace imports (now directly in main.py)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from dotenv import load_dotenv
import chromadb
# --- Pydantic Models for API Request/Response ---
# Moved these definitions to the top so they are available when FastAPI defines endpoints.
class UploadResponse(BaseModel):
    message: str
    processed_files: List[str]
    tenant_id: str
    language: str # Added language to response model
    chroma_status: str
class GenerateReportsResponse(BaseModel):
    message: str
    output_directory: str
    tenant_id: str
    report_details: Dict[str, Any]
# New Pydantic model to represent a single Q&A item
class QADetail(BaseModel):
    question: str
    answer: str
    created_at: Optional[str] = None # Assuming created_at might be part of the response
# New Pydantic model for the full Q&A templates response
class FullQATemplatesResponse(BaseModel):
    tenant_id: str
    language: str
    templates: List[QADetail] # A list of QADetail objects
# At module level
omni_token_cache = {}
omni_token_expiry = {}
# --- IndicTrans2 Specific Imports and Cython Setup ---
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
indictrans_path = current_dir / "IndicTrans2"
if not indictrans_path.exists():
    logger.warning(f"IndicTrans2 directory not found at {indictrans_path}. Please ensure it's cloned correctly.")
    parent_dir = current_dir.parent
    indictrans_path = parent_dir / "IndicTrans2"
    if not indictrans_path.exists():
        logger.error(f"IndicTrans2 not found in current or parent directory. Please place it correctly.")
        sys.path.append(str(current_dir))
    else:
        sys.path.append(str(indictrans_path))
else:
    sys.path.append(str(indictrans_path))
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
# --- Configuration from .env ---
OMNI_API_URL = os.getenv("API_URL")
OMNI_AUTH_TOKEN_GEN_URL = os.getenv("AUTH_URL") # This should now include TENANT_ID in the URL
SCANNER_TENANT_ID = os.getenv("TENANT_ID")
FONT_PATH = os.getenv("FONT_PATH", "NotoSans-Regular.ttf")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./reports")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# MySQL Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DB")
SUPPORTED_MIMETYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
    "text/csv",
    "text/plain"
]
# Validate required environment variables
if not all([OMNI_API_URL, OMNI_AUTH_TOKEN_GEN_URL, SCANNER_TENANT_ID,
            MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE]):
    raise RuntimeError("Missing required environment variables (API_URL, AUTH_URL, TENANT_ID, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE). Please check your .env file.")
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- Global objects for AI components (will be initialized in lifespan) ---
global_en_indic_tokenizer = None
global_en_indic_model = None
global_indic_processor = None
global_ollama_embeddings = None
global_initialized_flag = False
# global_tenant_data now stores docsearches per language
# {tenant_id: {'docsearches': {'en': Chroma_instance, 'hi': Chroma_instance}, 'translated_content': {}}}
global_tenant_data: Dict[str, Dict[str, Any]] = {}
# --- Cython Compilation Setup ---
try:
    import pyximport
    processor_file_path = indictrans_path / "huggingface_interface/IndicTransToolkit/IndicTransToolkit/processor.pyx"
    if processor_file_path.exists():
        processor_dir = os.path.dirname(processor_file_path)
        os.environ['CFLAGS'] = f"-I{processor_dir} -Wno-unused-function"
        logger.info(f"Attempting Cython compilation for {processor_file_path}...")
        compile_result = subprocess.run(
            ['cython', '-3', '--embed', str(processor_file_path)],
            check=False,
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            logger.warning(f"Cython compilation failed initially (exit code {compile_result.returncode}). Output: {compile_result.stderr}. Attempting pip install cython and retry...")
            try:
                subprocess.run(['pip', 'install', 'cython'], check=True, capture_output=True, text=True)
                logger.info("Cython installed. Retrying compilation...")
                subprocess.run(
                    ['cython', '-3', '--embed', str(processor_file_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Cython compilation successful after install for {processor_file_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"Failed to install or compile Cython: {e.stderr if isinstance(e, subprocess.CalledProcessError) else e}")
                raise RuntimeError(f"Could not compile Cython processor: {e}")
        else:
            logger.info(f"Cython compilation successful for {processor_file_path}")
    else:
        logger.error(f"Cython file not found at: {processor_file_path}")
        raise FileNotFoundError(f"Cython file not found at: {processor_file_path}. Please check your IndicTrans2 setup.")
    pyximport.install()
    logger.info("pyximport installed.")
except Exception as e:
    logger.error(f"Cython setup process failed: {str(e)}")
    raise RuntimeError("Failed to set up Cython dependencies for IndicTrans2. Ensure IndicTrans2 is cloned correctly.")
# Import after Cython setup
try:
    from IndicTrans2.huggingface_interface.IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor
    logger.info("IndicProcessor imported successfully within main.py.")
except ImportError as e:
    logger.error(f"Failed to import IndicProcessor in main.py: {e}. Check module-level setup.")
    raise RuntimeError(f"Failed to import IndicProcessor in main.py: {e}")
# --- MySQL Helper Functions ---
def get_mysql_connection():
    """Establishes and returns a MySQL database connection."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL database: {err}")
        raise RuntimeError(f"Failed to connect to MySQL: {err}")
def init_mysql_db():
    """Ensures the necessary MySQL tables exist."""
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        # Table to store metadata about ChromaDB instances
        # Now stores db_path for a tenant AND language
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chroma_metadata (
                tenant_id VARCHAR(255) NOT NULL,
                lang VARCHAR(10) NOT NULL,
                db_path VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, lang)
            )
        """)
        # --- NEW: Table to store Q&A templates ---
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
        logger.info("MySQL tables 'chroma_metadata' and 'qa_templates' ensured to exist.")
    except Exception as e:
        logger.error(f"Error initializing MySQL database: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()
# Modified to accept lang
def record_chroma_db_metadata(tenant_id: str, lang: str, db_path: str):
    """Records or updates ChromaDB metadata in MySQL for a specific tenant and language."""
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        # UPSERT: Insert if not exists, update if exists
        cursor.execute("""
            INSERT INTO chroma_metadata (tenant_id, lang, db_path)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE db_path = VALUES(db_path)
        """, (tenant_id, lang, db_path))
        conn.commit()
        logger.info(f"Recorded/Updated ChromaDB metadata for tenant '{tenant_id}' (lang: {lang}) in MySQL.")
    except Exception as e:
        logger.error(f"Error recording/updating ChromaDB metadata for tenant '{tenant_id}' (lang: {lang}): {e}")
        raise
# Modified to accept lang
def get_chroma_db_path_from_metadata(tenant_id: str, lang: str) -> Optional[str]:
    """Retrieves ChromaDB path for a tenant and language from MySQL metadata."""
    conn = None
    try:
        conn = get_mysql_connection() # Corrected from get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT db_path FROM chroma_metadata WHERE tenant_id = %s AND lang = %s", (tenant_id, lang))
        result = cursor.fetchone()
        if result:
            logger.info(f"Retrieved ChromaDB path for tenant '{tenant_id}' (lang: {lang}) from MySQL.")
            return result[0]
        logger.info(f"No ChromaDB path found for tenant '{tenant_id}' (lang: {lang}) in MySQL metadata.")
        return None
    except Exception as e:
        logger.error(f"Error retrieving ChromaDB path for tenant '{tenant_id}' (lang: {lang}): {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- NEW FUNCTION: Save Q&A templates to MySQL DB ---
def save_qa_templates_to_db(tenant_id: str, lang: str, qa_templates: List[QADetail]):
    """Saves a list of Q&A templates to the MySQL database for a specific tenant and language."""
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Prepare data for batch insert/update
        # Using INSERT ... ON DUPLICATE KEY UPDATE to handle existing entries
        insert_query = """
            INSERT INTO qa_templates (tenant_id, lang, question, answer)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                answer = VALUES(answer),
                updated_at = CURRENT_TIMESTAMP
        """

        data_to_insert = [
            (tenant_id, lang, qa.question, qa.answer)
            for qa in qa_templates
        ]

        cursor.executemany(insert_query, data_to_insert)
        conn.commit()

        logger.info(f"Saved/Updated {len(data_to_insert)} Q&A templates for tenant '{tenant_id}' (lang: {lang}) in MySQL.")

    except Exception as e:
        logger.error(f"Error saving Q&A templates for tenant '{tenant_id}' (lang: {lang}): {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
# --- End of NEW FUNCTION ---

# --- File Processing Functions ---
def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text from supported file types."""
    try:
        if file_type == "application/pdf":
            text = ""
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            ppt = pptx.Presentation(file_path)
            text = ""
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
                    elif shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                text += run.text + "\n"
            return text
        elif file_type == "text/csv":
            return pd.read_csv(file_path).to_string()
        elif file_type == "text/plain":
            with open(file_path, "r", encoding='utf-8') as f: # Specify encoding
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path} ({file_type}): {str(e)}", exc_info=True)
        raise RuntimeError(f"Error extracting text from file: {str(e)}")
# --- Report Generation Functions ---
def generate_pdf_filename(language: str) -> str:
    """Generate consistent PDF filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"QA_Report_{language}_{timestamp}.pdf"
def generate_qa_pdf(qa_data: List[Dict[str, str]], language: str = "en") -> Tuple[BytesIO, str]:
    """Generate PDF with Q&A data, handling language-specific fonts and text wrapping."""
    buffer = BytesIO()
    filename = generate_pdf_filename(language)
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_pos = height - 40
    line_height = 14
    # Font configuration for Hindi
    font_loaded = False
    if language == "hi" and os.path.exists(FONT_PATH):
        try:
            pdfmetrics.registerFont(TTFont('NotoSans', FONT_PATH))
            font_name = 'NotoSans'
            font_loaded = True
            logger.info(f"Registered font: {FONT_PATH} for Hindi.")
        except Exception as e:
            logger.error(f"Font registration failed for {FONT_PATH}: {str(e)}")
    header_font = 'Helvetica-Bold'
    text_font = 'Helvetica'
    if font_loaded and language == "hi": # Apply NotoSans only for Hindi if loaded
        header_font = 'NotoSans'
        text_font = 'NotoSans'
    # PDF Header
    c.setFont(header_font, 16)
    header_text = f"Q&A Report ({language.upper()}) - {datetime.now().strftime('%Y-%m-%d')}"
    c.drawCentredString(width/2, y_pos, header_text)
    y_pos -= 40
    # Handle translations if language is Hindi
    qa_data_for_pdf = []
    if language == "hi" and global_indic_processor and global_en_indic_tokenizer and global_en_indic_model:
        questions_to_translate = [item["question"] for item in qa_data]
        answers_to_translate = [item["answer"] for item in qa_data]
        try:
            # Batch translate questions and answers using the global batch_translate function
            translated_questions = batch_translate(questions_to_translate, "eng_Latn", "hin_Deva")
            translated_answers = batch_translate(answers_to_translate, "eng_Latn", "hin_Deva")
            for i, item in enumerate(qa_data):
                if i < len(translated_questions) and i < len(translated_answers):
                    qa_data_for_pdf.append({
                        "question": translated_questions[i],
                        "answer": translated_answers[i],
                        "created_at": item["created_at"]
                    })
                else:
                    qa_data_for_pdf.append(item) # Fallback to original if translation fails for a pair
            logger.info(f"Translated Q&A for PDF generation in {language}.")
        except Exception as e:
            logger.error(f"PDF translation failed for {language}: {str(e)}. Using original English content.", exc_info=True)
            qa_data_for_pdf = qa_data # Fallback to original
    else:
        qa_data_for_pdf = qa_data # Use original content for English reports
    # Add content to PDF
    for idx, item in enumerate(qa_data_for_pdf):
        # Check for page overflow before adding new item
        if y_pos < 120: # Start new page if not enough space for a full Q&A block
            c.showPage()
            y_pos = height - 40
            c.setFont(header_font, 12) # Reset font after page break
        # Question
        c.setFont(header_font, 12)
        question_text = f"{idx+1}. {item['question']}"
        # Text wrapping for question
        words = question_text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if c.stringWidth(test_line, header_font, 12) < (width - 144): # 72pt margin on each side
                line = test_line
            else:
                c.drawString(72, y_pos, line)
                y_pos -= line_height
                line = word
        if line:
            c.drawString(72, y_pos, line)
            y_pos -= line_height
        # Answer
        c.setFont(text_font, 10)
        answer_lines = []
        words = item["answer"].split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if c.stringWidth(test_line, text_font, 10) < (width - 144):
                line = test_line
            else:
                answer_lines.append(line)
                line = word
        if line:
            answer_lines.append(line)
        for line_part in answer_lines:
            if y_pos < 100: # New page if insufficient space
                c.showPage()
                y_pos = height - 40
                c.setFont(text_font, 10) # Reset font after page break
            c.drawString(72, y_pos, line_part)
            y_pos -= line_height
        # Metadata
        c.setFont("Helvetica-Oblique", 8)
        try:
            created_date = datetime.fromisoformat(item["created_at"]).strftime("%Y-%m-%d %H:%M")
        except:
            created_date = "Unknown date"
        c.drawString(72, y_pos, f"Created: {created_date}")
        y_pos -= line_height * 2 # Extra space between Q&A pairs
    c.save()
    buffer.seek(0)
    return buffer, filename
def process_pdf_buffer_for_chroma(pdf_buffer: BytesIO) -> str:
    """Extract text from PDF buffer specifically for ChromaDB storage."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)
        return " ".join([page.extract_text() or "" for page in pdf_reader.pages])
    except Exception as e:
        logger.error(f"PDF buffer processing error for ChromaDB: {str(e)}", exc_info=True)
        return ""
def save_pdf_to_disk(pdf_buffer: BytesIO, filename: str, output_dir: str):
    """Save PDF buffer to disk in the specified directory."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(pdf_buffer.getvalue())
        logger.info(f"Saved PDF to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save PDF to disk: {str(e)}", exc_info=True)
        raise
# --- Helper Functions (Moved and adapted from AIAssistant) ---
def _initialize_translation_model_and_tokenizer(ckpt_dir: str, quantization: str):
    """Initializes a translation model and tokenizer."""
    qconfig = None
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
        device_map="auto"
    )
    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()
    model.eval()
    return tokenizer, model
async def initialize_global_scanner_models():
    """Initializes global translation models and embeddings for the scanner."""
    global global_initialized_flag, global_indic_processor, global_en_indic_tokenizer, global_en_indic_model, global_ollama_embeddings
    if global_initialized_flag:
        logger.info("Global scanner models already initialized.")
        return
    logger.info("Initializing global scanner models (translation and embeddings)...")
    # Initialize IndicProcessor
    global_indic_processor = IndicProcessor(inference=True)
    logger.info("IndicProcessor initialized.")
    # Initialize Translation Models
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
    global_en_indic_tokenizer, global_en_indic_model = _initialize_translation_model_and_tokenizer(
        en_indic_ckpt_dir, "4-bit" # Assuming 4-bit quantization as per previous
    )
    logger.info("Translation models initialized.")
    # Initialize Ollama Embeddings
    global_ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    logger.info(f"Ollama Embeddings initialized with model: {EMBEDDING_MODEL}.")
    global_initialized_flag = True
    logger.info("All global scanner models initialization complete.")
async def _get_omni_api_token() -> str:
    """Dynamically fetches the OMNI API token from the OMNI_AUTH_TOKEN_GEN_URL.
    Handles both {tenant_id} and {{tenant_id}} placeholder formats correctly."""
    logger.info(f"Fetching OMNI API token from {OMNI_AUTH_TOKEN_GEN_URL}...")
    # Check cache first
    if omni_token_cache.get(SCANNER_TENANT_ID):
        logger.info(f"Using cached OMNI API token for tenant {SCANNER_TENANT_ID}.")
        return omni_token_cache[SCANNER_TENANT_ID]
    # CRITICAL FIX: Properly handle URL placeholder replacement
    # This handles both single and double brace formats
    token_gen_url = OMNI_AUTH_TOKEN_GEN_URL
    if "{{tenant_id}}" in token_gen_url:
        token_gen_url = token_gen_url.replace("{{tenant_id}}", SCANNER_TENANT_ID)
    elif "{tenant_id}" in token_gen_url:
        token_gen_url = token_gen_url.replace("{tenant_id}", SCANNER_TENANT_ID)
    else:
        # If no placeholder found, append tenant_id as query parameter
        logger.warning(f"OMNI_AUTH_TOKEN_GEN_URL doesn't contain {{tenant_id}} placeholder. Current: {OMNI_AUTH_TOKEN_GEN_URL}")
        # Try to append as query parameter
        separator = "&" if "?" in OMNI_AUTH_TOKEN_GEN_URL else "?"
        token_gen_url = f"{OMNI_AUTH_TOKEN_GEN_URL}{separator}tenant_id={SCANNER_TENANT_ID}"
    logger.info(f"Requesting OMNI token for tenant {SCANNER_TENANT_ID} from: {token_gen_url}")
    try:
        response = requests.get(token_gen_url, headers={'Accept': 'application/json'}, timeout=10, verify=False)
        response.raise_for_status()
        token = response.json().get('data', {}).get('token')
        if not token:
            # Try alternative response structure
            token = response.json().get('token')
            if not token:
                raise ValueError("Token not found in response from OMNI auth service.")
        omni_token_cache[SCANNER_TENANT_ID] = token
        logger.info("Successfully fetched new OMNI API token.")
        return token
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching OMNI API token from {token_gen_url}: {e}")
        raise RuntimeError(f"Failed to fetch authentication token for OMNI API: {e}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing OMNI API token response: {e}")
        logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise RuntimeError(f"Invalid response format from OMNI auth service: {e}")
# Update this function in your scanning service code
# Modified to accept lang parameter
# Update this function in your scanning service code
# Modified to accept lang parameter
async def _get_tenant_docsearch(tenant_id: str, lang: str) -> Chroma:
    """
    Retrieves or creates a Chroma vector store for a given tenant and language,
    with its persistence directory named as 'chroma_db_base/chroma_db_{tenant_id}/{lang}_db'.
    Metadata about this ChromaDB is stored in MySQL.
    """
    if not global_ollama_embeddings:
        raise RuntimeError("Ollama Embeddings not initialized. Call initialize_global_scanner_models first.")
    # Ensure tenant_data structure for the specific language ChromaDB
    if tenant_id not in global_tenant_data:
        global_tenant_data[tenant_id] = {'docsearches': {}, 'translated_content': {}}
    if 'docsearches' not in global_tenant_data[tenant_id]:
        global_tenant_data[tenant_id]['docsearches'] = {}
    if lang not in global_tenant_data[tenant_id]['docsearches']:
        db_dir = f"./chroma_db_base/{tenant_id}/{lang}_db"
        os.makedirs(db_dir, exist_ok=True)
        try:
            record_chroma_db_metadata(tenant_id, lang, db_dir)
        except Exception as e:
            logger.error(f"Could not record ChromaDB metadata in MySQL for tenant '{tenant_id}' (lang: {lang}): {e}")
        logger.info(f"Initializing Chroma DB for tenant: {tenant_id}, language: {lang} at directory {db_dir}")
        global_tenant_data[tenant_id]['docsearches'][lang] = Chroma(
            persist_directory=db_dir,
            embedding_function=global_ollama_embeddings,
            collection_name="documents"  # Use the same collection name as AI Assistant
        )
    return global_tenant_data[tenant_id]['docsearches'][lang]

async def _fetch_qa_data_from_omni(tenant_id: str) -> List[Dict[str, str]]:
    """
    Fetches Q&A data for a given tenant from the OMNI_API_URL,
    using a dynamically generated OMNI API token and requesting all pages.
    Handles responses where data might be a list (common with page=all) or a nested dict.
    """
    logger.info(f"Fetching QA data for tenant: {tenant_id} from {OMNI_API_URL} with page=all")
    
    omni_api_token = await _get_omni_api_token()
    
    headers = {
        "Token": omni_api_token,
        "Accept": "application/json"
    }
    
    params = {
        "page": "all"
    }
    
    try:
        response = requests.get(
            OMNI_API_URL, 
            headers=headers, 
            params=params, 
            verify=False
        )
        response.raise_for_status()
        
        full_qa_response = response.json()
        
        # Robust extraction: 
        # 1. If response is a list, use it directly.
        # 2. If it's a dict, try getting 'data'. If 'data' is a list, use it.
        # 3. If 'data' is another dict, try getting 'data' inside it.
        if isinstance(full_qa_response, list):
            qa_items_raw = full_qa_response
        elif isinstance(full_qa_response, dict):
            data_field = full_qa_response.get('data', [])
            if isinstance(data_field, list):
                qa_items_raw = data_field
            elif isinstance(data_field, dict):
                qa_items_raw = data_field.get('data', [])
            else:
                qa_items_raw = []
        else:
            qa_items_raw = []

        if not isinstance(qa_items_raw, list):
            logger.error(f"Unexpected QA data format from OMNI API. Response: {full_qa_response}")
            return []
            
        extracted_qa = []
        for q_item in qa_items_raw:
            if isinstance(q_item, dict) and "question" in q_item and "answer" in q_item:
                extracted_qa.append({
                    "question": q_item["question"],
                    "answer": q_item["answer"],
                    "created_at": q_item.get("created_at", datetime.now().isoformat())
                })
            else:
                # Silently skip if item isn't a valid dict to prevent crash
                continue
                
        return extracted_qa

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching QA data from {OMNI_API_URL}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to fetch QA data from OMNI API: {e}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error processing response from OMNI API: {e}. Response snippet: {str(response.text)[:200]}", exc_info=True)
        raise RuntimeError(f"Invalid response format from OMNI API: {e}")

def batch_translate(input_sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    """Translates a batch of sentences using the global translation models."""
    global global_indic_processor, global_en_indic_tokenizer, global_en_indic_model
    if not global_indic_processor or not global_en_indic_tokenizer or not global_en_indic_model:
        raise RuntimeError("Translation models not initialized. Call initialize_global_scanner_models first.")
    translations = []
    translation_batch_size = 4
    for i in range(0, len(input_sentences), translation_batch_size):
        batch = input_sentences[i:i + translation_batch_size]
        if not batch:
            continue
        processed_batch = global_indic_processor.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = global_en_indic_tokenizer(
            processed_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        with torch.no_grad():
            generated_tokens = global_en_indic_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=1024,
                num_beams=5,
                num_return_sequences=1,
            )
        decoded_tokens = global_en_indic_tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        translations += global_indic_processor.postprocess_batch(decoded_tokens, lang=tgt_lang)
    return translations
async def populate_tenant_data_for_scanner(tenant_id: str) -> bool:
    """
    Fetches questions for a given tenant from OMNI API, translates them,
    and populates their language-specific vector stores. This acts as the "training" step for the RAG component per tenant.
    """
    global global_initialized_flag, global_tenant_data
    if not global_initialized_flag:
        await initialize_global_scanner_models()
    if tenant_id not in global_tenant_data or 'translated_content' not in global_tenant_data[tenant_id]:
        global_tenant_data.setdefault(tenant_id, {})['translated_content'] = {}
    translated_content_for_tenant = global_tenant_data[tenant_id]['translated_content']
    qa_data = await _fetch_qa_data_from_omni(tenant_id)
    english_questions_to_process = []
    english_answers_to_process = []
    created_ats_to_process = [] # To store created_at for each QA pair
    for item in qa_data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and question not in translated_content_for_tenant:
            english_questions_to_process.append(question)
            english_answers_to_process.append(answer)
            created_ats_to_process.append(item.get("created_at", datetime.now().isoformat())) # Store created_at
    if english_questions_to_process:
        logger.info(f"Translating {len(english_questions_to_process)} new questions for tenant {tenant_id}...")
        try:
            translated_questions = batch_translate(english_questions_to_process, "eng_Latn", "hin_Deva")
            translated_answers = batch_translate(english_answers_to_process, "eng_Latn", "hin_Deva")
        except Exception as e:
            logger.error(f"Translation failed for tenant {tenant_id}: {str(e)}", exc_info=True)
            return False
        english_documents_to_add = []
        hindi_documents_to_add = []
        for i, (q, a) in enumerate(zip(english_questions_to_process, english_answers_to_process)):
            translated_content_for_tenant[q] = {
                "original_question": q,
                "hindi_question": translated_questions[i],
                "original_answer": a,
                "hindi_answer": translated_answers[i],
                "source_lang": "en",
                "created_at": created_ats_to_process[i] # Use the fetched created_at
            }
            # Add original English Q&A to the English documents list
            english_documents_to_add.append({
                "page_content": f"Question: {q}\nAnswer: {a}",
                "metadata": {"source": f"OMNI_FAQ_{tenant_id}", "type": "qa", "lang": "en", "original_question": q}
            })
            # Add Hindi Q&A to the Hindi documents list
            hindi_documents_to_add.append({
                "page_content": f"प्रश्न: {translated_questions[i]}\nउत्तर: {translated_answers[i]}",
                "metadata": {"source": f"OMNI_FAQ_{tenant_id}", "type": "qa", "lang": "hi", "original_question": q}
            })
        # Add documents to the tenant's specific Chroma DB for English
        if english_documents_to_add:
            tenant_docsearch_en = await _get_tenant_docsearch(tenant_id, "en") # Pass "en"
            texts_en = [d["page_content"] for d in english_documents_to_add]
            metadatas_en = [d["metadata"] for d in english_documents_to_add]
            tenant_docsearch_en.add_texts(
                texts=texts_en,
                metadatas=metadatas_en,
                ids=[f"omni_faq_{tenant_id}_en_{uuid.uuid4().hex[:8]}" for _ in range(len(texts_en))]
            )
            tenant_docsearch_en.persist()
            logger.info(f"Populated English Chroma DB for tenant {tenant_id} with {len(english_documents_to_add)} new entries from OMNI API.")
        # Add documents to the tenant's specific Chroma DB for Hindi
        if hindi_documents_to_add:
            tenant_docsearch_hi = await _get_tenant_docsearch(tenant_id, "hi") # Pass "hi"
            texts_hi = [d["page_content"] for d in hindi_documents_to_add]
            metadatas_hi = [d["metadata"] for d in hindi_documents_to_add]
            tenant_docsearch_hi.add_texts(
                texts=texts_hi,
                metadatas=metadatas_hi,
                ids=[f"omni_faq_{tenant_id}_hi_{uuid.uuid4().hex[:8]}" for _ in range(len(texts_hi))]
            )
            tenant_docsearch_hi.persist()
            logger.info(f"Populated Hindi Chroma DB for tenant {tenant_id} with {len(hindi_documents_to_add)} new entries from OMNI API.")
        logger.info(f"Successfully processed {len(english_questions_to_process)} new questions for tenant {tenant_id}.")
        return True
    logger.info(f"No new questions to process for tenant {tenant_id}.")
    return False
# Modified to accept lang parameter
async def add_document_to_tenant_chroma(tenant_id: str, text: str, source_name: str, file_type: str, lang: str = "en"):
    """
    Splits text and adds it to the specified tenant's ChromaDB collection for the given language.
    """
    if not global_initialized_flag:
        await initialize_global_scanner_models()
    if not text:
        logger.warning(f"No text provided for {source_name} to add to tenant {tenant_id}'s {lang} ChromaDB.")
        return
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        chunks = splitter.split_text(text)
        tenant_chroma_instance = await _get_tenant_docsearch(tenant_id, lang) # Pass lang here
        metadatas = [{
            "source": source_name,
            "chunk_index": i,
            "file_type": file_type,
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "lang": lang # Add language metadata
        } for i in range(len(chunks))]
        tenant_chroma_instance.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=[f"doc_{tenant_id}_{lang}_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
        )
        tenant_chroma_instance.persist()
        logger.info(f"Stored {len(chunks)} chunks from {source_name} in tenant {tenant_id}'s {lang} ChromaDB.")
    except Exception as e:
        logger.error(f"Error adding document to tenant {tenant_id}'s {lang} ChromaDB for {source_name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to store document in ChromaDB for tenant {tenant_id}.")
async def get_questions_for_tenant(tenant_id: str, lang: str) -> List[QADetail]: # Return type changed
    """
    Get list of Q&A details in specified language for a tenant from its translated_content store.
    This includes Q&A from OMNI API templates.
    """
    await populate_tenant_data_for_scanner(tenant_id)
    if tenant_id not in global_tenant_data or 'translated_content' not in global_tenant_data[tenant_id]:
        return []
    translated_content_for_tenant = global_tenant_data[tenant_id]['translated_content']
    qa_list = [] # Renamed from questions_list
    for content in translated_content_for_tenant.values():
        if lang == "en":
            qa_list.append(QADetail(
                question=content["original_question"],
                answer=content["original_answer"],
                created_at=content.get("created_at")
            ))
        elif lang == "hi":
            qa_list.append(QADetail(
                question=content["hindi_question"],
                answer=content["hindi_answer"],
                created_at=content.get("created_at")
            ))
    return qa_list
# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes global scanner models and MySQL connection.
    """
    logger.info("FastAPI application startup: Initializing global scanner models and MySQL...")
    try:
        init_mysql_db() # Initialize MySQL table
        await initialize_global_scanner_models()
        logger.info("FastAPI application startup: All scanner models and MySQL initialized.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Depending on criticality, you might want to raise HTTPException here
        # or have a health check endpoint fail. For now, we log and continue.
        raise # Re-raise to prevent app from starting if MySQL init fails
    yield # Application starts and handles requests
    # Cleanup code runs after the application shuts down
    logger.info("FastAPI application shutdown: Cleaning up resources.")
# --- FastAPI Application Setup ---
app = FastAPI(
    title="Single-Tenant Document Processor & Q&A Report Generator",
    description="API for uploading documents, generating Q&A reports, and managing data for a single predefined tenant.",
    version="1.0.0",
    lifespan=lifespan # Assign the lifespan context manager
)
# CORS Middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- API Endpoints ---
@app.get("/health", summary="Health check endpoint.")
async def health_check():
    """Service health check, including AI Assistant and directories."""
    scanner_status = "initializing"
    if global_initialized_flag:
        scanner_status = "initialized"
    chroma_base_exists = os.path.exists("./chroma_db_base")
    reports_dir_exists = os.path.exists(OUTPUT_DIR)
    return {
        "status": "running",
        "scanner_models_status": scanner_status,
        "chromadb_base_directory_check": "ok" if chroma_base_exists else "warning: base directory missing or not created",
        "reports_output_directory_exists": reports_dir_exists,
        "omni_api_url": OMNI_API_URL,
        "device": DEVICE,
        "dedicated_tenant_id": SCANNER_TENANT_ID # Show which tenant this instance serves
    }
@app.post("/upload", response_model=UploadResponse, summary="Upload and process documents into the dedicated tenant's language-specific ChromaDB.")
async def upload_files_api(
    files: List[UploadFile] = File(..., description="List of files to upload and process."),
    lang: str = Query("en", description="The language of the uploaded document (e.g., 'en' or 'hi'). Valid options: 'en', 'hi'"), # Added lang parameter
):
    """
    Uploads multiple documents, extracts their text content, and stores
    the chunks in this service's dedicated tenant's language-specific ChromaDB (from .env).
    No external bearer token is required for this endpoint.
    """
    if not global_initialized_flag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scanner global models not initialized. Please wait or check server logs."
        )
    if lang not in ["en", "hi"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language: {lang}. Supported languages for upload are 'en' and 'hi'."
        )
    temp_dir = None
    processed_filenames = []
    try:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        for file in files:
            if file.content_type not in SUPPORTED_MIMETYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.content_type} for {file.filename}. Supported types: {', '.join(SUPPORTED_MIMETYPES)}"
                )
            # Save file to temp location
            suffix = os.path.splitext(file.filename)[1]
            temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{suffix}")
            with open(temp_file_path, "wb") as buffer_file:
                shutil.copyfileobj(file.file, buffer_file)
            logger.info(f"Saved temporary file: {temp_file_path}")
            # Extract text
            file_text = extract_text_from_file(temp_file_path, file.content_type)
            if not file_text:
                logger.warning(f"No text extracted from {file.filename}.")
                continue
            # Add to the dedicated tenant's language-specific ChromaDB
            await add_document_to_tenant_chroma(
                tenant_id=SCANNER_TENANT_ID, # Use fixed tenant ID from .env
                text=file_text,
                source_name=file.filename,
                file_type=file.content_type,
                lang=lang # Pass the language here
            )
            processed_filenames.append(file.filename)
        if not processed_filenames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported files processed or no text extracted from uploaded files."
            )
        return UploadResponse(
            message=f"{len(processed_filenames)} files processed and stored successfully for dedicated tenant {SCANNER_TENANT_ID} in {lang.upper()} database",
            processed_files=processed_filenames,
            tenant_id=SCANNER_TENANT_ID, # Reflect dedicated tenant ID in response
            language=lang, # Include the language in the response
            chroma_status="stored"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload and processing for dedicated tenant {SCANNER_TENANT_ID}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File processing failed: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
@app.post("/generate-reports", response_model=GenerateReportsResponse, summary="Generate Q&A reports for the dedicated tenant in specified languages.")
async def generate_reports_api(
    languages: List[str] = Query(["en", "hi"], description="Languages to generate reports for (en/hi)"),
):
    """
    Generates Q&A reports (PDFs) based on data fetched from the OMNI API,
    translates if necessary, saves them to disk, and stores their content
    in this service's dedicated tenant's language-specific ChromaDB.
    No external bearer token is required for this endpoint, as it operates on the dedicated tenant ID from environment variables.
    """
    if not global_initialized_flag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scanner global models not initialized. Please wait or check server logs."
        )
    valid_languages = ["en", "hi"]
    for lang in languages:
        if lang not in valid_languages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid language: {lang}. Valid options are 'en' or 'hi'"
            )
    try:
        logger.info(f"Initiating internal OMNI API calls for dedicated tenant {SCANNER_TENANT_ID} for report generation.")
        # Ensure QA data is fetched and processed into global_tenant_data['translated_content']
        await populate_tenant_data_for_scanner(SCANNER_TENANT_ID)
        logger.info(f"OMNI API data fetched and processed for dedicated tenant {SCANNER_TENANT_ID}.")
        # Retrieve the Q&A data directly from the global_tenant_data store
        qa_data = list(global_tenant_data.get(SCANNER_TENANT_ID, {}).get('translated_content', {}).values())
        if not qa_data:
            return JSONResponse(
                content={"message": f"No Q&A data found for dedicated tenant {SCANNER_TENANT_ID} from OMNI API.", "tenant_id": SCANNER_TENANT_ID},
                status_code=status.HTTP_404_NOT_FOUND
            )
        # We need raw English questions and answers for PDF generation, as 'generate_qa_pdf' handles translation
        raw_qa_data = [
            {"question": item["original_question"], "answer": item["original_answer"], "created_at": item.get("created_at", datetime.now().isoformat())}
            for item in qa_data
        ]
        results = {}
        for lang in languages:
            logger.info(f"Generating {lang.upper()} report for dedicated tenant {SCANNER_TENANT_ID}...")
            # generate_qa_pdf will use batch_translate internally for Hindi
            pdf_buffer, filename = generate_qa_pdf(raw_qa_data, language=lang)
            file_path = save_pdf_to_disk(pdf_buffer, filename, OUTPUT_DIR)
            pdf_buffer.seek(0)
            text_content = process_pdf_buffer_for_chroma(pdf_buffer)
            if text_content:
                await add_document_to_tenant_chroma(
                    tenant_id=SCANNER_TENANT_ID, # Use fixed tenant ID from .env
                    text=text_content,
                    source_name=filename,
                    file_type="application/pdf",
                    lang=lang # Pass the language of the report to store in appropriate DB
                )
                results[lang] = {"file_path": file_path, "chroma_status": "stored"}
            else:
                results[lang] = {"file_path": file_path, "chroma_status": "failed_to_extract_text"}
        return GenerateReportsResponse(
            message=f"Reports generated for languages: {', '.join(languages)} for dedicated tenant {SCANNER_TENANT_ID}",
            output_directory=OUTPUT_DIR,
            tenant_id=SCANNER_TENANT_ID, # Reflect dedicated tenant ID in response
            report_details=results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed for dedicated tenant {SCANNER_TENANT_ID}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )
@app.get("/api/fetch-chatbot-templates", response_model=FullQATemplatesResponse, summary="Fetch Q&A templates for the dedicated tenant.")
async def fetch_chatbot_templates_api(
    lang: str = Query("en", description="The language of the templates to fetch (e.g., 'en' or 'hi'). Valid options: 'en', 'hi'"),
):
    """
    Retrieves predefined chatbot Q&A templates for this service's dedicated tenant (from .env).
    This endpoint first ensures the latest templates are fetched from OMNI API
    and stored in the dedicated tenant's ChromaDB, then returns them as a list of Q&A objects.
    Additionally, it saves the fetched templates to the MySQL database.
    No external bearer token is required for this endpoint.
    """
    if not global_initialized_flag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scanner global models not initialized. Please wait or check server logs."
        )
    if lang not in ["en", "hi"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language: {lang}. Supported languages for templates are 'en' and 'hi'."
        )
    logger.info(f"Initiating internal OMNI API calls for dedicated tenant {SCANNER_TENANT_ID} to fetch templates.")
    # This populates global_tenant_data['translated_content'] with Q&A objects
    await populate_tenant_data_for_scanner(SCANNER_TENANT_ID)
    logger.info(f"OMNI API data fetched and processed for dedicated tenant {SCANNER_TENANT_ID}.")
    logger.info(f"API Request: Fetching chatbot templates for dedicated tenant {SCANNER_TENANT_ID}, language {lang}.")
    # Now call the updated get_questions_for_tenant which returns List[QADetail]
    qa_templates = await get_questions_for_tenant(tenant_id=SCANNER_TENANT_ID, lang=lang)

    # --- NEW CODE: Save templates to MySQL DB ---
    try:
        save_qa_templates_to_db(SCANNER_TENANT_ID, lang, qa_templates)
        logger.info(f"Successfully saved Q&A templates to MySQL for tenant {SCANNER_TENANT_ID}, language {lang}.")
    except Exception as e:
        logger.error(f"Failed to save Q&A templates to MySQL: {e}")
        # Log the error but do not fail the endpoint, so the user still gets the templates.
        pass
    # --- END NEW CODE ---

    return FullQATemplatesResponse(tenant_id=SCANNER_TENANT_ID, language=lang, templates=qa_templates)
if __name__ == "__main__":
    os.makedirs("./chroma_db_base", exist_ok=True) # A new base directory to contain all tenant-specific DBs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8001)