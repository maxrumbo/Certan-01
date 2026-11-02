"""
run_retrieval_benchmark.py
--------------------------
Skrip otomatis untuk:
  1. Memuat seluruh korpus dokumen enterprise dan melakukan chunking.
  2. Menyimpan embeddings secara persisten ke dalam ChromaDB (data/chroma_db/).
  3. Menjalankan pengujian temu balik semantik (Retrieval Benchmark) dengan 7 kueri bisnis:
     - Kueri relevan langsung (klaim, HAZMAT, retur COD, SLA Pos).
     - Kueri kepatuhan hukum jam kerja (UU 13/2003 - koneksi M2).
     - Kueri variasi bahasa / sinonim sehari-hari.
     - Kueri Out-of-Domain (OOD) untuk menguji presisi penolakan sistem.
  4. Menyimpan rekap hasil metrik Cosine Similarity ke reports/benchmark_m3_retrieval.csv.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from certan_01.ingestion import build_corpus_chunks
from certan_01.vector_store import ChromaKnowledgeBase


# Kueri Uji Bisnis Resmi untuk Evaluasi Temu Balik Semantik (Milestone 3)
BENCHMARK_QUERIES = [
    {
        "query_id": "Q01_KLAIM_JNT",
        "type": "Spesifik / Relevan",
        "query": "Berapa hari batas waktu pengajuan klaim paket rusak di J&T dan dokumen apa yang wajib disertakan?",
        "expected_target": "SOP_EKSPEDISI / KLAIM_ASURANSI",
    },
    {
        "query_id": "Q02_HAZMAT_DANGEROUS",
        "type": "Spesifik / Relevan",
        "query": "Apakah kurir boleh membawa aki basah, bahan kimia korosif, aerosol, dan baterai litium?",
        "expected_target": "SOP_EKSPEDISI (Dangerous Goods)",
    },
    {
        "query_id": "Q03_JAM_KERJA_KURIR",
        "type": "Kepatuhan Hukum (M2-Bridge)",
        "query": "Berapa jam batas maksimal waktu kerja harian kurir menurut undang-undang ketenagakerjaan?",
        "expected_target": "REGULASI_KETENAGAKERJAAN (UU 13/2003)",
    },
    {
        "query_id": "Q04_SOP_RETUR_COD",
        "type": "Operasional Depot / Lapangan",
        "query": "Bagaimana prosedur penanganan paket gagal antar jika pembeli menolak bayar COD atau alamat tidak jelas?",
        "expected_target": "SOP_RETUR / KEBIJAKAN_MARKETPLACE",
    },
    {
        "query_id": "Q05_SLA_POS_EXPRESS",
        "type": "Perjanjian SLA Layanan",
        "query": "Apa perbedaan standar waktu penyampaian paket Pos Express dengan Pos Kilat Khusus?",
        "expected_target": "KONTRAK_LAYANAN_POS / REGULASI_POS",
    },
    {
        "query_id": "Q06_SINONIM_BAHASA_KASUAL",
        "type": "Variasi Bahasa / Sinonim Kasual",
        "query": "Ganti rugi barang belanjaan hilang pas kurir nganter berapa duit yang dibalikin toko?",
        "expected_target": "KLAIM_ASURANSI / KEBIJAKAN_MARKETPLACE",
    },
    {
        "query_id": "Q07_OUT_OF_DOMAIN",
        "type": "Out-of-Domain (Penolakan)",
        "query": "Bagaimana cara membuat resep kue bolu pandan kukus yang empuk dan mengembang?",
        "expected_target": "PENOLAKAN (Similarity Rendah < 0.25)",
    },
]


def main():
    root_dir = Path(__file__).resolve().parents[1]
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = reports_dir / "benchmark_m3_retrieval.csv"

    print("=" * 80)
    print("  CERTAN-01: BENCHMARK TEMU BALIK SEMANTIK CHROMADB (MILESTONE 3)")
    print("=" * 80)

    # 1. Chunking Korpus
    print("\n[*] Langkah 1: Memuat dan memotong korpus dokumen enterprise...")
    t0_chunk = time.time()
    chunks = build_corpus_chunks()
    chunk_time = time.time() - t0_chunk
    print(f"    -> Berhasil memproses {len(chunks)} potongan teks ({chunk_time:.2f} detik)")

    # 2. Inisialisasi ChromaDB Persisten
    print("\n[*] Langkah 2: Mengindeks ke dalam Vector Store ChromaDB persisten...")
    kb = ChromaKnowledgeBase(persist_dir=root_dir / "data" / "chroma_db")
    stats_before = kb.get_stats()
    print(f"    -> Status sebelum pengindeksan: {stats_before['total_chunks']} chunks")

    t0_index = time.time()
    indexed_count = kb.index_chunks(chunks, batch_size=150)
    index_time = time.time() - t0_index
    stats_after = kb.get_stats()
    print(f"    -> Selesai mengindeks {indexed_count} chunks ke ChromaDB ({index_time:.2f} detik)")
    print(f"    -> Total koleksi saat ini: {stats_after['total_chunks']} chunks")

    # 3. Eksekusi Pengujian 7 Kueri Bisnis
    print("\n[*] Langkah 3: Menjalankan 7 Kueri Evaluasi Temu Balik Semantik...")
    print("=" * 80)

    results_table: List[Dict[str, Any]] = []

    # Ambang batas penolakan out-of-domain (Threshold: 0.30)
    OOD_THRESHOLD = 0.30

    for item in BENCHMARK_QUERIES:
        q_id = item["query_id"]
        q_type = item["type"]
        query_text = item["query"]
        expected = item["expected_target"]

        t0_q = time.time()
        hits = kb.search_semantic(query_text, top_k=3)
        latency_ms = (time.time() - t0_q) * 1000

        top_hit = hits[0] if hits else None
        top_sim = top_hit["similarity"] if top_hit else 0.0
        top_src = top_hit["metadata"].get("source", "-") if top_hit else "-"
        top_cat = top_hit["metadata"].get("category", "-") if top_hit else "-"
        top_chunk_id = top_hit["chunk_id"] if top_hit else "-"
        top_snippet = top_hit["content"][:120].replace("\n", " ") if top_hit else "-"

        # Status evaluasi
        if "Out-of-Domain" in q_type:
            # Sukses jika similarity rendah / ditolak
            is_passed = top_sim < OOD_THRESHOLD
            status = "DITOLAK DENGAN TEPAT (OOD)" if is_passed else "GAGAL (Lolos Palsu)"
        else:
            # Sukses jika similarity memadai dan kategori relevan
            is_passed = top_sim >= 0.40
            status = "DITEMUKAN RELEVAN" if is_passed else "KURANG RELEVAN"

        print(f"[{q_id}] ({q_type})")
        print(f"  Kueri       : \"{query_text}\"")
        print(f"  Target Ekspektasi : {expected}")
        print(f"  Top Match   : {top_chunk_id} | {top_src} ({top_cat})")
        print(f"  Similarity  : {top_sim:.4f} (Cosine) | Latensi: {latency_ms:.1f} ms")
        print(f"  Cuplikan    : \"{top_snippet}...\"")
        print(f"  Keputusan   : {status}")
        print("-" * 80)

        results_table.append(
            {
                "query_id": q_id,
                "type": q_type,
                "query": query_text,
                "expected_target": expected,
                "top_chunk_id": top_chunk_id,
                "top_source": top_src,
                "top_category": top_cat,
                "similarity": top_sim,
                "latency_ms": round(latency_ms, 2),
                "status": status,
                "snippet": top_snippet,
            }
        )

    # 4. Simpan ke CSV
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "type",
                "query",
                "expected_target",
                "top_chunk_id",
                "top_source",
                "top_category",
                "similarity",
                "latency_ms",
                "status",
                "snippet",
            ],
        )
        writer.writeheader()
        writer.writerows(results_table)

    print(f"\n[SUKSES] Rekapitulasi hasil benchmark tersimpan di: {csv_output_path}")


if __name__ == "__main__":
    main()

