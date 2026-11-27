"""Milvus 向量库封装：本地 milvus-lite 文件库 / Milvus 服务 / Zilliz Cloud 同一套接口。

URI 形式：
- ./data/milvus_rag.db        本地 milvus-lite（零依赖，推荐开发用）
- http://localhost:19530      自建 Milvus（Docker）
- https://xxx.zillizcloud.com Zilliz Cloud（配 MILVUS_TOKEN）
"""
from pymilvus import MilvusClient

from src import config


class MilvusStore:
    def __init__(self, dim: int):
        kwargs = {"uri": config.MILVUS_URI}
        if config.MILVUS_TOKEN:
            kwargs["token"] = config.MILVUS_TOKEN
        self.client = MilvusClient(**kwargs)
        self.collection = config.COLLECTION
        self.dim = dim
        self._ensure_collection()
        self.client.load_collection(self.collection)  # 已有集合需加载后才能检索

    def _ensure_collection(self):
        if not self.client.has_collection(self.collection):
            # 快速建表：自增主键 + 向量字段（IP 内积），动态字段存 text/source/chunk_id
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dim,
                metric_type="IP",
                auto_id=True,
            )

    def reset(self):
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ensure_collection()

    def add(self, vectors: list[list[float]], metas: list[dict]) -> int:
        rows = [{"vector": v, **m} for v, m in zip(vectors, metas)]
        res = self.client.insert(self.collection, rows)
        return res.get("insert_count", len(rows))

    def search(self, query_vec: list[float], top_k: int = None) -> list[dict]:
        res = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            limit=top_k or config.TOP_K,
            output_fields=["text", "source", "chunk_id"],
        )
        hits = []
        for hit in res[0]:
            hits.append({
                "score": hit["distance"],  # 归一化向量的 IP 即余弦相似度
                "text": hit["entity"]["text"],
                "source": hit["entity"]["source"],
                "chunk_id": hit["entity"]["chunk_id"],
            })
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
