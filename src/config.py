"""配置：从 .env / 环境变量读取。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# LLM（答案生成）：任意 OpenAI 兼容接口
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding：local = 本地 bge-small-zh-v1.5（免费离线）；api = OpenAI 兼容 Embedding 接口
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# Milvus：本地文件（milvus-lite）或 Milvus/Zilliz Cloud 服务地址
MILVUS_URI = os.getenv("MILVUS_URI", str(ROOT / "data" / "milvus_rag.db"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION = os.getenv("MILVUS_COLLECTION", "kb_chunks")

# 切分与检索
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))

KB_DIR = ROOT / "data" / "kb"
