"""FastAPI 服务封装：POST /ask、POST /ingest、GET /health"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG 私有知识库", version="0.1.0")
_chain = None


def get_chain():
    global _chain
    if _chain is None:
        from src.rag import RAGChain

        _chain = RAGChain()
    return _chain


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.post("/ask")
async def ask(req: AskRequest):
    res = get_chain().ask(req.question)
    res.pop("retrieved", None)  # 接口只返回来源摘要，完整检索明细走日志/评测
    return res


@app.post("/ingest")
async def ingest():
    from src.ingest import build_chunks

    chain = get_chain()
    chunks = build_chunks()
    chain.store.reset()
    batch, total = 32, 0
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = chain.embedder.embed([c["text"] for c in part])
        total += chain.store.add(vecs, part)
    return {"chunks": total}


@app.get("/health")
async def health():
    chain = get_chain()
    return {"status": "ok", "chunks": chain.store.count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
