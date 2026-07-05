import os
import torch
import logging
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Dict, Any, Optional
from rapidfuzz import fuzz
try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_huggingface import HuggingFacePipeline
    from langchain.memory import ConversationBufferMemory
except ImportError:
    OllamaEmbeddings = None
    Chroma = None
    ConversationalRetrievalChain = None
    PromptTemplate = None
    ChatMessageHistory = None
    RunnableWithMessageHistory = None
    HuggingFacePipeline = None
    ConversationBufferMemory = None

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
import mysql.connector
from mysql.connector import pooling, Error
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


class AIAssistant:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AIAssistant, cls).__new__(cls)
        return cls._instance

    def __init__(self, chroma_db_base_dir: str, executor: ThreadPoolExecutor = None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.executor = executor or ThreadPoolExecutor(max_workers=3, thread_name_prefix="ai_worker")
            self.llm = None
            self.embeddings = None
            self.conversation_chains = {}
            self.session_histories = {}
            self.initialized_global_models_flag = False
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.CHROMA_DB_BASE_DIR = chroma_db_base_dir

            # Memory optimization
            self._optimize_memory_settings()
            
            # MySQL setup
            self.MYSQL_HOST = os.getenv("MYSQL_HOST")
            self.MYSQL_USER = os.getenv("MYSQL_USER")
            self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
            self.MYSQL_DB = os.getenv("MYSQL_DB")
            self.db_pool = None
            self._init_db_pool()

            # Prompts
            if PromptTemplate is not None:
                self.PROMPT_EN = PromptTemplate(
                    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledgeable AI assistant. For greetings like hello/hi, respond warmly. Use provided context and retrieved sources to answer the question.
If the context does not contain the answer, you MUST say "I don't have this information in my knowledge base."

Context: {context}

Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""",
                    input_variables=["context", "question"]
                )
                self.PROMPT_HI = PromptTemplate(
                    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
आप एक ज्ञानवान AI सहायक हैं। अभिवादन के लिए सौहार्दपूर्ण उत्तर दें। प्रश्न का उत्तर देने के लिए प्रदान किए गए संदर्भ का उपयोग करें।
यदि संदर्भ में उत्तर नहीं है, तो आपको अवश्य कहना चाहिए "मेरे ज्ञान आधार में यह जानकारी नहीं है।"

संदर्भ: {context}

प्रश्न: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""",
                    input_variables=["context", "question"]
                )
            else:
                self.PROMPT_EN = None
                self.PROMPT_HI = None

            # Ensure tables exist
            self.init_mysql_db()
            logger.info("AI Assistant initialized")

    # ------------------ Memory Optimization ------------------ #
    def _optimize_memory_settings(self):
        """Set environment variables for better memory management"""
        # Memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        
        if torch.cuda.is_available():
            # Set smaller cache size if needed
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU

    def _cleanup_memory(self):
        """Clean up GPU memory aggressively"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # ------------------ MySQL ------------------ #
    def _init_db_pool(self):
        try:
            self.db_pool = pooling.MySQLConnectionPool(
                pool_name="ai_mysql_pool",
                pool_size=5,
                pool_reset_session=True,
                host=self.MYSQL_HOST,
                user=self.MYSQL_USER,
                password=self.MYSQL_PASSWORD,
                database=self.MYSQL_DB,
                autocommit=True
            )
            logger.info("MySQL connection pool initialized")
        except Error as e:
            logger.error(f"Failed to initialize database pool: {e}")

    def get_mysql_connection(self):
        if self.db_pool:
            try:
                return self.db_pool.get_connection()
            except Error as e:
                logger.error(f"Failed to get connection from pool: {e}")
        return mysql.connector.connect(
            host=self.MYSQL_HOST,
            user=self.MYSQL_USER,
            password=self.MYSQL_PASSWORD,
            database=self.MYSQL_DB
        )

    def init_mysql_db(self):
        conn = None
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            tables = {
                "sessions": """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        tenant_id VARCHAR(255) NOT NULL,
                        customer_id VARCHAR(255) NOT NULL,
                        api_token VARCHAR(255) NOT NULL,
                        language VARCHAR(10) DEFAULT 'en',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_tenant_customer (tenant_id, customer_id)
                    )
                """,
                "chat_history": """
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
                """,
                "customer_tokens": """
                    CREATE TABLE IF NOT EXISTS customer_tokens (
                        token VARCHAR(255) PRIMARY KEY,
                        tenant_id VARCHAR(255) NOT NULL,
                        customer_id VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NULL
                    )
                """,
                "chroma_metadata": """
                    CREATE TABLE IF NOT EXISTS chroma_metadata (
                        tenant_id VARCHAR(255) NOT NULL,
                        lang VARCHAR(10) NOT NULL,
                        db_path VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (tenant_id, lang)
                    )
                """,
                "qa_templates": """
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
                """
            }
            for ddl in tables.values():
                cursor.execute(ddl)
            conn.commit()
        finally:
            if conn and conn.is_connected():
                conn.close()

    # ------------------ Quantized Model Initialization ------------------ #
    async def initialize_global_models(self):
        if self.initialized_global_models_flag:
            return
        logger.info("Initializing quantized Llama-3.2-3B model...")

        try:
            if OllamaEmbeddings is None or pipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None or HuggingFacePipeline is None:
                logger.warning("Optional AI model dependencies are unavailable; skipping model initialization")
                return

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            logger.info(f"Loading quantized version of: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True,  # 4-bit quantization
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                trust_remote_code=True
            )

            # Create optimized pipeline
            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            self.initialized_global_models_flag = True
            logger.info("Quantized Llama-3.2-3B model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize 4-bit quantized model: {e}")
            # Fallback to 8-bit quantization for the same model
            await self._initialize_llama_8bit_fallback()

    async def _initialize_llama_8bit_fallback(self):
        """Fallback to 8-bit quantization for Llama-3.2-3B"""
        try:
            logger.info("Trying 8-bit quantization for Llama-3.2-3B...")
            
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 8-bit quantization fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,  # 8-bit quantization
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )

            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            self.initialized_global_models_flag = True
            logger.info("8-bit quantized Llama-3.2-3B initialized successfully")
            
        except Exception as e:
            logger.error(f"8-bit quantization also failed: {e}")
            # Ultimate fallback to CPU with the same model but smaller context
            await self._initialize_llama_cpu_fallback()

    async def _initialize_llama_cpu_fallback(self):
        """Ultimate fallback to CPU with memory optimizations"""
        try:
            logger.info("Loading Llama-3.2-3B on CPU with memory optimizations...")
            
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            self.initialized_global_models_flag = True
            logger.info("Llama-3.2-3B CPU model initialized successfully")
        except Exception as e:
            logger.error(f"CPU fallback also failed: {e}")
            raise RuntimeError("Could not initialize Llama-3.2-3B model with any quantization method")

    # ------------------ Session & Retrieval ------------------ #
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
        return self.session_histories[session_id]

    def get_tenant_docsearch(self, tenant_id: str, lang: str) -> Optional[Chroma]:
        if not self.embeddings:
            logger.error("Embeddings not initialized. Call initialize_global_models first.")
            return None

        db_dir = f"./chroma_db_base/{tenant_id}/{lang}_db"
        logger.info(f"Initializing Chroma DB for tenant: {tenant_id}, language: {lang} at directory {db_dir}")
        os.makedirs(os.path.dirname(db_dir), exist_ok=True)

        try:
            tenant_chroma = Chroma(
                persist_directory=db_dir,
                embedding_function=self.embeddings,
                collection_name="documents"
            )
            doc_count = tenant_chroma._collection.count()
            logger.info(f"Chroma collection for tenant {tenant_id}, lang {lang} contains {doc_count} documents")
            return tenant_chroma
        except Exception as e:
            logger.error(f"Error initializing ChromaDB for tenant {tenant_id}, lang {lang}: {str(e)}")
            return None

    def _check_chroma_collection_exists(self, tenant_id: str, lang: str) -> bool:
        db_dir = f"./chroma_db_base/{tenant_id}/{lang}_db"
        return os.path.exists(db_dir)

    async def _get_conversational_chain_with_history(self, tenant_id: str, lang: str, session_id: str):
        # Create ConversationBufferMemory with session's ChatMessageHistory
        message_history = self.get_session_history(session_id)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        tenant_chroma = self.get_tenant_docsearch(tenant_id, lang)
        if not tenant_chroma:
            raise RuntimeError(f"No Chroma collection for tenant {tenant_id} lang {lang}")

        # Initialize chains for English and Hindi
        docsearch = tenant_chroma
        chain_en = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.75}),
            rephrase_question=False,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.PROMPT_EN, "output_key": "answer"},
            verbose=False
        )
        chain_hi = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
            rephrase_question=False,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.PROMPT_HI, "output_key": "answer"},
            verbose=False
        )
        return chain_en, chain_hi

    # ------------------ QA Templates ------------------ #
    async def get_questions(self, tenant_id: str, lang: str) -> List[Dict[str, Any]]:
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT question, answer, created_at  FROM qa_templates WHERE tenant_id=%s AND lang=%s", (tenant_id, lang))
            results = cursor.fetchall()
            return results or []
        finally:
            if conn and conn.is_connected():
                conn.close()

    # ------------------ Answer Cleaning ------------------ #
    def clean_answer(self, answer: str, lang: str = "en") -> str:
        if not answer:
            return ""
        answer = re.sub(r'\n+', '\n', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'\*\*', '', answer)  # Remove all double stars
        answer = answer.strip()

        # Remove Hindi prefix
        if lang.lower() == "hi" and answer.startswith("उत्तर:"):
            answer = answer[len("उत्तर:"):].strip()

        return answer

    # ------------------ Process Query ------------------ #
    async def process_query(self, query: str, tenant_id: str, customer_id: Union[str, int], lang: str, session_id: str):
        start_time = time.time()
        
        # Clean memory before processing
        self._cleanup_memory()
        
        try:
            logger.info(f"=== STARTING QUERY PROCESSING WITH QUANTIZED MODEL ===")
            logger.info(f"Tenant ID: {tenant_id}")
            logger.info(f"Customer ID: {customer_id}")
            logger.info(f"Language: {lang}")
            logger.info(f"Original Query: '{query}'")

            if not self.initialized_global_models_flag:
                await self.initialize_global_models()

            # ---------- Step 1: Check Q&A ----------
            questions = await self.get_questions(tenant_id, lang)
            normalized_query = query.lower().strip()

            for q_data in questions:
                question_text = q_data["question"].lower().strip()
                similarity_score = fuzz.ratio(normalized_query, question_text)
                logger.info(f"Q&A similarity: {similarity_score} for '{normalized_query}' vs '{question_text}'")
                if similarity_score > 85:
                    answer = self.clean_answer(q_data["answer"], lang=lang)
                    return {
                        "answer": answer,
                        "sources": [{"content": q_data["answer"], "metadata": {"match_type": "faq"}}],
                        "tenant_id": tenant_id,
                        "customer_id": str(customer_id)
                    }

            # Check Chroma collection
            collection_exists = self._check_chroma_collection_exists(tenant_id, lang)
            if not collection_exists:
                logger.warning(f"Chroma collection for tenant {tenant_id}, language {lang} is empty or doesn't exist")
                fallback = "I'm still learning about your documents. Please try again later or contact support if the issue persists."
                if lang.lower() == "hi":
                    fallback = "मैं अभी आपके दस्तावेज़ों के बारे में सीख रहा हूँ। कृपया बाद में पुनः प्रयास करें या सहायता के लिए संपर्क करें।"
                return {
                    "answer": fallback,
                    "sources": [],
                    "tenant_id": tenant_id,
                    "customer_id": str(customer_id)
                }

            # ---------- Step 2: Use retrieval + generative chain ----------
            chain_en, chain_hi = await self._get_conversational_chain_with_history(tenant_id, lang, session_id)
            chain = chain_hi if lang.lower() == "hi" else chain_en
            history = self.get_session_history(session_id)

            # Invoke chain
            logger.info(f"🚀 Invoking quantized model with query: {query}")
            
            # Clean memory right before the heavy operation
            self._cleanup_memory()
            
            res = await chain.ainvoke(
                {"question": query, "chat_history": history.messages},
                config={"configurable": {"session_id": f"{tenant_id}-{customer_id}"}}
            )

            # Clean memory immediately after
            self._cleanup_memory()

            # ---------- Extract answer and sources ----------
            answer = self.clean_answer(res.get("answer", ""), lang=lang)
            #logger.info(f"Answer: {answer}")
            sources = []

            if res and "source_documents" in res and res["source_documents"]:
                sources = [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in res["source_documents"]
                ]

            if not answer.strip() or any(phrase in answer.lower() for phrase in ["i don't have", "no information", "not in my knowledge", "मेरे ज्ञान आधार में"]):
                if lang.lower() == "hi":
                    answer = "मेरे ज्ञान आधार में यह जानकारी नहीं है। कृपया अधिक विवरण के लिए समर्थन से संपर्क करें।"
                else:
                    answer = "I don't have specific information about this topic in my knowledge base. Please contact support for more detailed assistance."
            #logger.info(f"Answer: {answer}")
            response_time = (time.time() - start_time) * 1000
            logger.info(f"Query processed by quantized model in {response_time:.2f}ms")
            return {
                "answer": answer,
                "sources": sources,
                "tenant_id": tenant_id,
                "customer_id": str(customer_id)
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory even with quantization")
                self._cleanup_memory()
                if lang.lower() == "hi":
                    answer = "क्षमा करें, तकनीकी सीमा के कारण मैं इस प्रश्न का उत्तर देने में असमर्थ हूं। कृपया बाद में पुनः प्रयास करें।"
                else:
                    answer = "Sorry, I'm unable to answer this question due to technical limitations. Please try again later."
                return {
                    "answer": answer,
                    "sources": [],
                    "tenant_id": tenant_id,
                    "customer_id": str(customer_id)
                }
            else:
                raise e
        except Exception as e:
            logger.error(f"Query processing error with quantized model: {e}", exc_info=True)
            self._cleanup_memory()
            fallback = "Sorry, I encountered an error while processing your query. Please try again."
            if lang.lower() == "hi":
                fallback = "क्षमा करें, आपके प्रश्न को संसाधित करते समय त्रुटि हुई। कृपया पुनः प्रयास करें।"
            return {
                "answer": fallback,
                "sources": [],
                "tenant_id": tenant_id,
                "customer_id": str(customer_id)
            }


    def record_chroma_db_metadata(self, tenant_id: str, lang: str, db_path: str):
        try:
            self.init_mysql_db()
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chroma_metadata (tenant_id, lang, db_path)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE db_path = VALUES(db_path)
            """, (tenant_id, lang, db_path))
            conn.commit()
        finally:
            if conn and conn.is_connected():
                conn.close()

    def get_chroma_db_path_from_metadata(self, tenant_id: str, lang: str) -> Optional[str]:
        try:
            self.init_mysql_db()
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT db_path FROM chroma_metadata WHERE tenant_id=%s AND lang=%s", (tenant_id, lang))
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    # ------------------ Additional Methods for Debugging ------------------ #
    async def debug_chroma_collection(self, tenant_id: str, lang: str):
        """Debug method to check Chroma collection contents"""
        tenant_chroma = self.get_tenant_docsearch(tenant_id, lang)
        if not tenant_chroma:
            return {"error": "Chroma collection not found"}
        
        collection = tenant_chroma._collection
        count = collection.count()
        
        if count == 0:
            return {"message": "Collection is empty", "count": 0}
        
        all_docs = collection.get()
        
        # Analyze sources
        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": count,
            "sources_breakdown": sources,
            "sample_documents": [
                {
                    "content": all_docs['documents'][i][:500],
                    "metadata": all_docs['metadatas'][i]
                }
                for i in range(min(3, len(all_docs['documents'])))
            ]
        }


    async def get_chat_history(
        self, tenant_id: str, customer_id: Union[str, int], session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch chat history for a given tenant, customer, and specific session_id only.
        """
        customer_id_str = str(customer_id)
        conn = None
        cursor = None

        try:
            logger.info(
                f"Fetching chat history for tenant={tenant_id}, "
                f"customer={customer_id_str}, session_id={session_id}..."
            )

            conn = mysql.connector.connect(
                host=self.MYSQL_HOST,
                user=self.MYSQL_USER,
                password=self.MYSQL_PASSWORD,
                database=self.MYSQL_DB
            )

            if not conn.is_connected():
                logger.warning("Database connection not active.")
                return []

            cursor = conn.cursor(dictionary=True)

            # ✅ Fetch only chat history for the specified session_id
            cursor.execute(
                """
                SELECT role, message, timestamp
                FROM chat_history
                WHERE tenant_id = %s AND customer_id = %s AND session_id = %s
                ORDER BY timestamp ASC
                """,
                (tenant_id, customer_id_str, session_id)
            )

            rows = cursor.fetchall()
            logger.info(f"Fetched {len(rows)} messages for session_id={session_id}")

            # Return list of chat messages
            return [
                {
                    "role": row["role"],
                    "message": row["message"],
                    "timestamp": row["timestamp"],
                    "session_id": session_id,
                    "customer_id": customer_id_str,
                }
                for row in rows
            ]

        except Error as e:
            logger.error(f"Error fetching chat history for session_id={session_id}: {e}")
            return []

        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()
                logger.info(f"Database connection closed for session_id={session_id}.")