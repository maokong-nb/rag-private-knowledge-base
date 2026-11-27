"""评测脚本：28 题库内问答 + 2 题越界拒答。

指标：检索命中率（期望来源是否进入 top-k）、答案命中率（answer_checks 组间 AND、组内 OR 同义判分）、
引用规范率（答案是否标注 [编号] 来源）、越界拒答正确率、时延与成本。

用法：
    python eval/run_eval.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import RAGChain  # noqa: E402

# 价格按 DeepSeek 官方刊例（元/百万 token），换模型时改这里
PRICE_IN_PER_M = 2.0
PRICE_OUT_PER_M = 8.0

DATASET = Path(__file__).parent / "dataset.json"
OUT_DIR = Path(__file__).parent / "results"


def resolve_checks(item: dict) -> list:
    """判分组：外层 AND、组内 OR（同义表达）。兼容 answer_keywords（每个词独立一组）。"""
    raw = item.get("answer_checks")
    if raw is None:
        raw = [[k] for k in item.get("answer_keywords", [])]
    return raw


def main():
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    chain = RAGChain()
    records = []
    for item in items:
        t0 = time.time()
        res = chain.ask(item["question"])
        latency = time.time() - t0
        answer = res["answer"]
        sources = [s["source"] for s in res["sources"]]

        expected = item.get("expected_source")
        norm = "".join(answer.split())
        checks = resolve_checks(item)
        if expected:
            retrieval_ok = expected in sources
            answer_ok = all(any(k in norm for k in group) for group in checks)
        else:  # 越界题：应明确拒答（组内任一命中即可）
            retrieval_ok = None
            answer_ok = any(k in norm for group in checks for k in group)

        has_citation = ("[" in answer and "]" in answer) or "来源" in answer
        usage = res["usage"]
        cost = usage["prompt"] * PRICE_IN_PER_M / 1e6 + usage["completion"] * PRICE_OUT_PER_M / 1e6
        records.append({
            "id": item["id"], "question": item["question"],
            "expected_source": expected, "sources": sources,
            "retrieval_ok": retrieval_ok, "answer_ok": answer_ok,
            "has_citation": has_citation,
            "top_scores": [s["score"] for s in res["sources"]],
            "latency_s": round(latency, 2),
            "prompt_tokens": usage["prompt"], "completion_tokens": usage["completion"],
            "cost_cny": round(cost, 6),
            "answer_preview": answer[:120],
        })
        r = "-" if retrieval_ok is None else ("✓" if retrieval_ok else "✗")
        print(f"[{item['id']:>2}] 检索{r} 答案{'✓' if answer_ok else '✗'} "
              f"引用{'✓' if has_citation else '✗'} {latency:.1f}s  {item['question'][:22]}")

    in_scope = [r for r in records if r["expected_source"]]
    ood = [r for r in records if not r["expected_source"]]
    n = len(records)
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "total": n,
        "retrieval_hit_rate": round(sum(r["retrieval_ok"] for r in in_scope) / len(in_scope), 4),
        "answer_accuracy": round(sum(r["answer_ok"] for r in in_scope) / len(in_scope), 4),
        "ood_refusal_accuracy": round(sum(r["answer_ok"] for r in ood) / len(ood), 4) if ood else None,
        "citation_rate": round(sum(r["has_citation"] for r in in_scope) / len(in_scope), 4),
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / n, 2),
        "total_cost_cny": round(sum(r["cost_cny"] for r in records), 4),
    }
    OUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"eval-{ts}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n===== 汇总 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
