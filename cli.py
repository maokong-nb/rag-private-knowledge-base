"""命令行入口：

    python cli.py ingest            # 加载 data/kb/ 下的文档，切分并向量化入库（重建）
    python cli.py ask "问题"        # 单次问答
    python cli.py chat              # 连续对话
    python cli.py stats             # 查看库内 chunk 数
"""
import sys

from src.embed import get_embedder
from src.ingest import build_chunks
from src.store import MilvusStore


def ingest():
    chunks = build_chunks()
    print(f"共切出 {len(chunks)} 个 chunk，开始向量化……")
    embedder = get_embedder()
    store = MilvusStore(dim=embedder.dim)
    store.reset()
    batch = 32
    total = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = embedder.embed([c["text"] for c in part])
        total += store.add(vecs, part)
    print(f"入库完成，共 {total} 条（collection={store.collection}）")


def ask(question: str):
    from src.rag import RAGChain

    chain = RAGChain()
    res = chain.ask(question)
    print(f"\n答：{res['answer']}")
    if res["sources"]:
        print("\n来源：")
        for s in res["sources"]:
            print(f"  [{s['rank']}] {s['source']}（相似度 {s['score']}）")
    print(f"\n（耗时 {res['latency_s']}s，token {res['usage']['prompt']}+{res['usage']['completion']}）")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    elif cmd == "chat":
        print("输入 quit 退出")
        while True:
            q = input("\n你：").strip()
            if q.lower() in ("quit", "exit"):
                break
            if q:
                ask(q)
    elif cmd == "stats":
        embedder = get_embedder()
        store = MilvusStore(dim=embedder.dim)
        print(f"collection={store.collection}，chunk 数={store.count()}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
