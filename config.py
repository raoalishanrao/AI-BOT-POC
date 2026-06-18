from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
PAGES_DIR = OUTPUT_DIR / "pages"
RAG_DIR = OUTPUT_DIR / "rag"

START_URL = "https://www.iqrauni.edu.pk/"
ALLOWED_DOMAIN = "iqrauni.edu.pk"

# Deep crawl limits
MAX_DEPTH = 3
MAX_PAGES = 80

# Crawl timing
PAGE_TIMEOUT_SECONDS = 60
HEARTBEAT_INTERVAL_SECONDS = 10

# Logging
LOG_FILE = OUTPUT_DIR / "crawl.log"

# RAG chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings (must match Supabase VECTOR dimension)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = 768

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv(
    "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", ""
)
KNOWLEDGE_TABLE = "university_knowledge"

# Chat LLM (gemini or groq) — embeddings still use Gemini
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "gemini").strip().lower()
CHAT_MODEL = os.getenv("CHAT_MODEL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_PROFILE_MODEL = os.getenv("GEMINI_PROFILE_MODEL", GEMINI_CHAT_MODEL)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_PROFILE_MODEL = os.getenv("GROQ_PROFILE_MODEL", "llama-3.1-8b-instant")
GROQ_CHAT_FALLBACK_MODELS = os.getenv(
    "GROQ_CHAT_FALLBACK_MODELS", "llama-3.1-8b-instant,openai/gpt-oss-20b"
)
GROQ_PROFILE_FALLBACK_MODELS = os.getenv(
    "GROQ_PROFILE_FALLBACK_MODELS", "llama-3.1-8b-instant,openai/gpt-oss-20b"
)
CHAT_FALLBACK_TO_GEMINI = os.getenv("CHAT_FALLBACK_TO_GEMINI", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
CHAT_MATCH_COUNT = int(os.getenv("CHAT_MATCH_COUNT", "12"))

# Ingest
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "10"))
INGEST_BATCH_DELAY_SECONDS = float(os.getenv("INGEST_BATCH_DELAY_SECONDS", "3"))

# Counselor agent
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "20"))
ADMISSIONS_EMAIL = os.getenv("ADMISSIONS_EMAIL", "admissions@iqrauni.edu.pk")
ADMISSIONS_PHONE = os.getenv("ADMISSIONS_PHONE", "051 9247407 / 051 8357378")
ADMISSIONS_URL = os.getenv("ADMISSIONS_URL", f"{START_URL}admissions")
SCHOLARSHIPS_URL = os.getenv("SCHOLARSHIPS_URL", f"{START_URL}scholarships")
FEES_URL = os.getenv("FEES_URL", f"{START_URL}fees")
CONTACT_URL = os.getenv("CONTACT_URL", f"{START_URL}contact-us")

# API server (chat widget)
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
