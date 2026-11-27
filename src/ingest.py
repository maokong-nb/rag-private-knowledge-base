"""文档加载与切分。

支持 .md / .txt / .pdf / .docx。切分按"段落优先、超长再切"策略，
chunk 保留来源文件名与序号，供引用溯源使用。
"""
from pathlib import Path

from src import config


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"不支持的文件类型：{path}")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """按段落聚合切分：尽量保持段落完整，超长段落按字符滑窗硬切。"""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) > chunk_size:  # 单段超长：滑窗硬切
                step = chunk_size - overlap
                chunks.extend(para[i:i + chunk_size] for i in range(0, len(para), step))
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(kb_dir: Path = None) -> list[dict]:
    """加载知识库全部文档并切分，返回 [{text, source, chunk_id}]。"""
    kb_dir = kb_dir or config.KB_DIR
    all_chunks = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf", ".docx"):
            continue
        text = load_document(path)
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({"text": chunk, "source": path.name, "chunk_id": i})
    return all_chunks
