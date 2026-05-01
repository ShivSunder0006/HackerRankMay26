"""
indexer.py — One-time FAISS index builder for the support corpus.

Run this BEFORE main.py:
    python code/indexer.py
"""

import json
import pathlib
import re
import sys
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

DOMAINS = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude":     DATA_DIR / "claude",
    "visa":       DATA_DIR / "visa",
}

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into overlapping word-count-bounded chunks."""
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_words: List[str] = []

    def flush():
        if current_words:
            chunks.append({
                "text": " ".join(current_words),
                "source": source,
                "chunk_index": len(chunks),
            })

    for para in paragraphs:
        para_words = para.split()

        # Long paragraph → split by sentences
        if len(para_words) > CHUNK_SIZE_WORDS * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sw = sent.split()
                if len(current_words) + len(sw) > CHUNK_SIZE_WORDS and current_words:
                    flush()
                    current_words = current_words[-CHUNK_OVERLAP_WORDS:] + sw
                else:
                    current_words.extend(sw)
        else:
            if len(current_words) + len(para_words) > CHUNK_SIZE_WORDS and current_words:
                flush()
                current_words = current_words[-CHUNK_OVERLAP_WORDS:] + para_words
            else:
                current_words.extend(para_words)

    flush()
    return chunks


def load_domain(domain_path: pathlib.Path, domain: str) -> List[Dict]:
    """Walk domain directory, read all .md files, return enriched chunks."""
    all_chunks = []
    md_files = sorted(domain_path.rglob("*.md"))

    for md_file in tqdm(md_files, desc=f"  {domain}", unit="file"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            rel = md_file.relative_to(domain_path)
            category = str(rel.parent) if len(rel.parts) > 1 else "general"
            source = f"{domain}/{rel}"
            for chunk in chunk_text(text, source):
                chunk["domain"] = domain
                chunk["category"] = category
            all_chunks.extend(chunk_text(text, source))
        except Exception as e:
            print(f"  Warning: {md_file}: {e}")

    return all_chunks


# ── Index builder ─────────────────────────────────────────────────────────────

def build_and_save(domain: str, chunks: List[Dict], model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim after normalization
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / f"{domain}.faiss"))
    (INDEX_DIR / f"{domain}_meta.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  OK Saved {domain}.faiss + {domain}_meta.json")


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for domain, path in DOMAINS.items():
        if not path.exists():
            print(f"\nSkipping {domain}: path not found ({path})")
            continue
        print(f"\n[{domain.upper()}] Scanning {path}")
        chunks = load_domain(path, domain)
        if not chunks:
            print(f"  No content found, skipping.")
            continue
        print(f"  {len(chunks)} chunks loaded")
        build_and_save(domain, chunks, model)

    print("\nOK Indexing complete. Run:  python code/main.py")


if __name__ == "__main__":
    main()
