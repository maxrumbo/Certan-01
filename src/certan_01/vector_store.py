"""
vector_store.py
---------------
Modul Vector Store ChromaDB persisten dan temu balik semantik (Semantic Search)
berbasis Dense Embeddings untuk Milestone 3.

Komponen Utama:
  - ChromaKnowledgeBase : Kelas pengelola koleksi vektor ChromaDB persisten.
  - index_chunks        : Pengindeksan batch potongan dokumen beserta metadata.
  - search_semantic     : Pencarian semantik berperingkat berbasis Cosine Similarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from certan_01.ingestion import DocumentChunk


DEFAULT_COLLECTION_NAME = "certan_logistics_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ChromaKnowledgeBase:
    """
    Pengelola penyimpanan persisten vektor ChromaDB dan mesin temu balik semantik.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if persist_dir is None:
            root = Path(__file__).resolve().parents[2]
            persist_dir = root / "data" / "chroma_db"

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        # Inisialisasi ChromaDB client persisten
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Cegah HuggingFace timeout jika berjalan offline / sandboxed
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Inisialisasi fungsi embedding
        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        except Exception:
            # Fallback ke ONNX default embedding jika sentence-transformers belum siap
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Koleksi persisten dengan metrik Cosine Similarity (hnsw:space = cosine)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "description": "Basis Pengetahuan Logistik Enterprise CERTAN-01"},
        )

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        """
        Mengindeks daftar DocumentChunk ke ChromaDB dalam mode batch.
        """
        if not chunks:
            return 0

        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]

            # Upsert untuk mencegah duplikasi jika skrip dijalankan ulang
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_indexed += len(batch)

        return total_indexed

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Melakukan pencarian semantik terhadap basis data vektor.
        
        Menghitung Cosine Similarity = 1.0 - Cosine Distance.
        Mendukung pemfilteran berbasis kategori dan ambang batas penolakan out-of-domain.
        """
        where_clause = {"category": category_filter} if category_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Dalam hnsw:space = cosine, distance = 1 - cosine_similarity
            # Sehingga cosine_similarity = 1.0 - distance
            similarity = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)

            # Jika ambang batas penolakan diterapkan
            if similarity_threshold is not None and similarity < similarity_threshold:
                continue

            hits.append(
                {
                    "chunk_id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": round(float(dist), 4),
                    "similarity": similarity,
                }
            )

        # Urutkan berdasarkan similarity tertinggi
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits

    def get_stats(self) -> Dict[str, Any]:
        """
        Mengembalikan statistik isi basis data vektor.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
            "embedding_model": self.model_name,
        }

