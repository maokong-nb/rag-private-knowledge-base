"""Embedding 后端。

- local：本地 BAAI/bge-small-zh-v1.5（sentence-transformers），免费、离线，512 维；
- api：任意 OpenAI 兼容 Embedding 接口（DashScope text-embedding-v4 等）。

向量统一做 L2 归一化，检索侧用内积即等价于余弦相似度。
"""
from src import config


class LocalEmbedder:
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or config.EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


class APIEmbedder:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.EMBEDDING_API_KEY)
        self.model = config.EMBEDDING_MODEL
        self.dim = None  # 首次调用后确定

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        self.dim = len(vecs[0])
        return vecs


_instance = None


def get_embedder():
    global _instance
    if _instance is None:
        if config.EMBEDDING_BACKEND == "api":
            _instance = APIEmbedder()
        else:
            _instance = LocalEmbedder()
    return _instance
