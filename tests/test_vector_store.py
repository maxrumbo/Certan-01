"""
test_vector_store.py
--------------------
Suite pengujian unit otomatis untuk Pipeline Ingestion & Vector Store ChromaDB (Milestone 3).

Kasus Uji:
  1. Pengujian Chunking Rekursif (Chunk size, overlap, tidak ada pemotongan sembrono).
  2. Pengujian Klasifikasi Dokumen (Metadata kategori logistik & regulasi).
  3. Pengujian ChromaDB Ingestion & Persistensi.
  4. Pengujian Temu Balik Semantik (Kueri klaim & regulasi menghasilkan top-match relevan).
  5. Pengujian Penolakan Out-of-Domain (Threshold filtering).
"""

import shutil
import tempfile
from pathlib import Path
import pytest

from certan_01.ingestion import (
    DocumentChunk,
    clean_text,
    classify_document,
    recursive_character_chunking,
)
from certan_01.vector_store import ChromaKnowledgeBase


# ---------------------------------------------------------------------------
# 1. Uji Pembersihan & Pemotongan Teks (Chunking)
# ---------------------------------------------------------------------------

def test_clean_text():
    raw = "Baris 1\r\n\r\n\r\nBaris 2\x0c   dengan   spasi\t\tberlebih."
    cleaned = clean_text(raw)
    assert "\x0c" not in cleaned
    assert "Baris 1" in cleaned
    assert "Baris 2" in cleaned
    assert "   " not in cleaned


def test_recursive_character_chunking():
    sample_text = (
        "Pasal 1: Definisi pengiriman paket e-commerce. Seluruh paket wajib ditimbang sebelum keberangkatan. "
        "Pasal 2: Batasan muatan kurir sepeda motor maksimal 20 kilogram. "
        "Pasal 3: Paket berukuran besar hanya dapat diangkut menggunakan mobil box logistik."
    )
    chunks = recursive_character_chunking(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120  # toleransi batas kata


# ---------------------------------------------------------------------------
# 2. Uji Klasifikasi Kategori Dokumen
# ---------------------------------------------------------------------------

def test_document_classification():
    cat1, _ = classify_document("SK_Pengiriman_JnT_Express.txt")
    assert cat1 == "SOP_EKSPEDISI"

    cat2, _ = classify_document("UU_No_13_2003_Ketenagakerjaan_Jam_Kerja_Kurir.pdf")
    assert cat2 == "REGULASI_KETENAGAKERJAAN"

    cat3, _ = classify_document("SOP_Penanganan_Retur_dan_Refund.docx")
    assert cat3 == "SOP_RETUR"


# ---------------------------------------------------------------------------
# 3. Uji ChromaDB Ingestion & Temu Balik Semantik
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_chroma_kb():
    temp_dir = Path(tempfile.mkdtemp(prefix="test_chroma_"))
    kb = ChromaKnowledgeBase(persist_dir=temp_dir, collection_name="test_logistics_collection")
    yield kb
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_chroma_indexing_and_query(temp_chroma_kb):
    test_chunks = [
        DocumentChunk(
            chunk_id="TEST_01",
            content="Batas waktu klaim barang rusak di ekspedisi adalah maksimal 3 hari kalender setelah barang diterima.",
            metadata={"source": "test_jnt.txt", "category": "KLAIM_ASURANSI", "title": "SOP Klaim J&T"},
        ),
        DocumentChunk(
            chunk_id="TEST_02",
            content="Kurir dilarang mengangkut bahan berbahaya seperti aki basah, aerosol mudah meledak, dan racun pestisida.",
            metadata={"source": "test_hazmat.txt", "category": "SOP_EKSPEDISI", "title": "Aturan Dangerous Goods"},
        ),
        DocumentChunk(
            chunk_id="TEST_03",
            content="Menurut UU Ketenagakerjaan No. 13/2003, waktu kerja normal karyawan logistik adalah maksimal 8 jam per hari.",
            metadata={"source": "test_uu13.txt", "category": "REGULASI_KETENAGAKERJAAN", "title": "Regulasi Jam Kerja"},
        ),
    ]

    indexed = temp_chroma_kb.index_chunks(test_chunks)
    assert indexed == 3
    assert temp_chroma_kb.get_stats()["total_chunks"] == 3

    # Uji pencarian semantik (Relevan)
    results = temp_chroma_kb.search_semantic("berapa hari batas komplain paket pecah rusak?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "TEST_01"
    assert results[0]["similarity"] > 0.50

    # Uji pencarian semantik (Jam Kerja Kurir - Sinkron M2)
    results_jam = temp_chroma_kb.search_semantic("jam kerja sopir kurir delivery", top_k=1)
    assert len(results_jam) == 1
    assert results_jam[0]["chunk_id"] == "TEST_03"
    assert "8 jam" in results_jam[0]["content"]

    # Uji penolakan kueri out-of-domain dengan similarity threshold
    results_ood = temp_chroma_kb.search_semantic(
        "resep masakan sayur lodeh santan gurih", top_k=1, similarity_threshold=0.85
    )
    assert len(results_ood) == 0  # Harus ditolak / kosong

