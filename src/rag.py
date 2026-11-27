"""RAG 主链路：检索 → 组上下文 → LLM 生成带引用的答案 → 拒答兜底。"""
import time

from openai import OpenAI

from src import config
from src.embed import get_embedder
from src.store import MilvusStore

PROMPT_TEMPLATE = """你是企业制度知识库助手。根据下面检索到的资料回答问题。

要求：
1. 回答必须基于资料，引用信息用 [编号] 标注来源，如 [1]。
2. 资料不足以回答时，明确回复"知识库中未提及相关内容"，并说明缺少什么，不要编造数字或条款。
3. 回答简洁，涉及数字、时限、金额时必须与资料原文一致。

资料：
{context}

问题：{question}
"""

NO_HIT_REPLY = "知识库中未提及相关内容（检索到的资料与问题相关度过低）。"


class RAGChain:
    def __init__(self):
        self.embedder = get_embedder()
        self.store = MilvusStore(dim=self.embedder.dim)
        self.llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    def retrieve(self, question: str, top_k: int = None) -> list[dict]:
        qvec = self.embedder.embed([question])[0]
        return self.store.search(qvec, top_k=top_k)

    def ask(self, question: str) -> dict:
        t0 = time.time()
        hits = self.retrieve(question)
        good = [h for h in hits if h["score"] >= config.SCORE_THRESHOLD]

        if not good:
            return {
                "answer": NO_HIT_REPLY,
                "sources": [],
                "retrieved": hits,
                "usage": {"prompt": 0, "completion": 0},
                "latency_s": round(time.time() - t0, 2),
            }

        context = "\n\n".join(
            f"[{i + 1}]（来源：{h['source']}）\n{h['text']}" for i, h in enumerate(good)
        )
        resp = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)}],
        )
        usage = resp.usage
        return {
            "answer": resp.choices[0].message.content,
            "sources": [{"rank": i + 1, "source": h["source"], "score": round(h["score"], 4)}
                        for i, h in enumerate(good)],
            "retrieved": hits,
            "usage": {"prompt": usage.prompt_tokens if usage else 0,
                      "completion": usage.completion_tokens if usage else 0},
            "latency_s": round(time.time() - t0, 2),
        }
