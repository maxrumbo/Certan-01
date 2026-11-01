"""
ingestion.py
------------
Pipeline prapemrosesan dokumen korpus dan word-aware recursive chunking
untuk sistem basis data vektor ChromaDB (Milestone 3).

Komponen Utama:
  - DocumentChunk              : Model data potongan teks beserta metadata lengkap.
  - clean_text                 : Pembersihan teks dan normalisasi whitespace.
  - extract_text               : Ekstraktor multi-format (PDF, DOCX, TXT, HTML).
  - recursive_character_chunking: Strategi pemotongan rekursif berbasis karakter & kata
                                 dengan window overlap terkontrol.
  - build_corpus_chunks        : Orkestrasi pemuatan seluruh korpus enterprise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    """
    Model data potongan dokumen korpus yang siap diindeks ke basis data vektor.
    
    Attributes:
        chunk_id    : Identifier unik chunk (misal: "DOC_JNT_001").
        content     : Teks isi potongan yang bersih.
        metadata    : Metadata penjelas untuk mendukung hybrid search & sitasi.
    """
    chunk_id: str
    content: str
    metadata: Dict[str, Any]


def clean_text(raw_text: str) -> str:
    """
    Membersihkan teks dari karakter kontrol, form feed, dan normalisasi spasi/newline.
    """
    if not raw_text:
        return ""
    # Hapus karakter kontrol ASCII non-printable
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw_text)
    # Ubah multiple newline > 2 menjadi double newline
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Hapus whitespace berlebih di setiap baris
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Mengekstrak teks dari dokumen PDF menggunakan pypdf.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text: List[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text)
    except Exception as e:
        # Fallback jika pypdf belum tersedia / galat
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    """
    Mengekstrak teks dari dokumen Word (.docx) menggunakan python-docx.
    """
    try:
        import docx
        doc = docx.Document(str(docx_path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except Exception as e:
        return ""


def extract_text_from_file(file_path: Path) -> str:
    """
    Router pembaca teks sesuai ekstensi berkas (.txt, .pdf, .docx, .html).
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return file_path.read_text(encoding="latin1", errors="ignore")

    elif suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    elif suffix == ".docx":
        return extract_text_from_docx(file_path)

    elif suffix == ".html":
        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        # Hapus script, style, tag
        cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<(?:br|p|div|h[1-6]|li)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        return cleaned

    return ""


def recursive_character_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    delimiters: Optional[List[str]] = None,
) -> List[str]:
    """
    Strategi pemotongan rekursif (Recursive Character Splitting) terstruktur.
    
    Memotong teks secara hierarkis (paragraf -> kalimat -> klausa -> kata)
    untuk menjaga keutuhan makna semantik, kemudian menggabungkan potongan
    dengan batas maksimum chunk_size dan overlap chunk_overlap.
    
    Args:
        text         : Teks dokumen yang telah dibersihkan.
        chunk_size   : Batas panjang karakter per potongan (standar: 500 karakter).
        chunk_overlap: Jumlah karakter tumpang-tindih antar-potongan bertetangga (standar: 100).
        delimiters   : Prioritas pemisah hierarkis.
    """
    if not text or not text.strip():
        return []

    if delimiters is None:
        delimiters = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()]

    # Pilih pemisah terbaik yang ada dalam teks
    chosen_delimiter = delimiters[-1]
    for delimiter in delimiters:
        if delimiter == "":
            chosen_delimiter = ""
            break
        if delimiter in text:
            chosen_delimiter = delimiter
            break

    # Pecah teks dengan pemisah terpilih
    if chosen_delimiter:
        splits = text.split(chosen_delimiter)
    else:
        splits = list(text)

    # Rekursi pada potongan yang masih melebihi chunk_size
    next_delimiters = delimiters[delimiters.index(chosen_delimiter) + 1 :] if chosen_delimiter in delimiters else []

    sub_chunks: List[str] = []
    for split in splits:
        split = split.strip()
        if not split:
            continue
        if len(split) > chunk_size and next_delimiters:
            sub_chunks.extend(
                recursive_character_chunking(
                    split,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    delimiters=next_delimiters,
                )
            )
        else:
            sub_chunks.append(split)

    # Gabungkan sub_chunks dengan memperhitungkan overlap
    final_chunks: List[str] = []
    current_chunk = ""

    for piece in sub_chunks:
        separator = chosen_delimiter if (current_chunk and chosen_delimiter) else " "
        candidate = f"{current_chunk}{separator}{piece}" if current_chunk else piece

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
                # Ambil overlap dari ujung potongan sebelumnya
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_seed = current_chunk[-chunk_overlap:]
                    current_chunk = f"{overlap_seed} {piece}".strip()
                else:
                    current_chunk = piece
            else:
                final_chunks.append(piece.strip())
                current_chunk = ""

    if current_chunk and current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    return [c for c in final_chunks if len(c) >= 30]


def classify_document(filename: str) -> Tuple[str, str]:
    """
    Menentukan kategori bisnis dan judul representatif berdasarkan nama berkas.
    """
    fn = filename.lower()

    if "jnt" in fn or "j&t" in fn:
        return "SOP_EKSPEDISI", "Syarat dan Ketentuan Pengiriman J&T Express & Packaging"
    if "sicepat" in fn:
        return "KEBIJAKAN_MARKETPLACE", "Kebijakan Pengiriman SiCepat di Tokopedia"
    if "asuransi" in fn:
        return "KLAIM_ASURANSI", "S&K Asuransi Pengiriman dan Komplain Barang Tokopedia"
    if "shopee" in fn:
        return "KLAIM_ASURANSI", "Kebijakan Ganti Rugi dan Klaim Barang Rusak Shopee"
    if "retur_paket" in fn or "faq-return" in fn:
        return "SOP_RETUR", "SOP Penanganan Paket Retur dan Penolakan COD Tokopedia"
    if "penerimaan_barang" in fn:
        return "SOP_GUDANG_DEPOT", "SOP Penerimaan Barang Inbound dari Supplier"
    if "retur_dan_refund" in fn or "retur_produk" in fn:
        return "SOP_RETUR", "SOP Penanganan Paket Retur, Reject, dan Refund"
    if "pos_indonesia" in fn:
        return "KONTRAK_LAYANAN_POS", "Perjanjian Kerjasama Standar Waktu Kirim PT Pos Indonesia"
    if "permenkominfo" in fn or "bn302" in fn:
        return "REGULASI_PEMERINTAH", "Permenkominfo No. 4/2021 tentang Standar Penyelenggaraan Pos"
    if "uu_no_13_2003" in fn or "ketenagakerjaan" in fn:
        return "REGULASI_KETENAGAKERJAAN", "UU No. 13/2003: Batas 8 Jam Kerja Harian Kurir"
    if "uu_no_38_2009" in fn or "pos" in fn:
        return "REGULASI_PEMERINTAH", "UU No. 38/2009 tentang Pos dan Pidana Pengiriman Barang Terlarang"
    if "keamanan_informasi" in fn:
        return "SOP_ENTERPRISE", "SOP Manajemen Keamanan Informasi dan Data Pelanggan"
    if "amazon" in fn:
        return "BENCHMARK_INDUSTRI", "Amazon Last-Mile Delivery Routing Benchmark Dataset"
    if "applsci-13" in fn or "applsci-11" in fn:
        return "TAKSONOMI_LOGISTIK", "Studi Ilmiah Taksonomi Batasan Operasional Kurir Last-Mile"
    if "joti" in fn or "24821" in fn:
        return "OPTIMASI_INDUSTRI", "Jurnal Optimasi Rute Pengiriman dan Free Pick-up Kurir"
    if "jointecoms" in fn or "328-339" in fn:
        return "OPERASIONAL_KURIR", "Sistem Master Paket dan Log Pengantaran Kurir"

    return "REFERENSI_LOGISTIK", filename.replace("_", " ").replace(".pdf", "").replace(".txt", "")


# Berkas materi kuliah dasar yang sengaja dieksklusikan dari korpus
EXCLUDED_FILES = {
    "26.pdf",
    "cambridge_ai_jegyzet.pdf",
    "ece448sp2018lecture06.pdf",
    "jarrar.lecturenotes.ch2.intelligentagents.pdf",
    "makalah-if2211-stima-2024 (62).pdf",
    "makalah-stima-2023-(11).pdf",
}


def build_corpus_chunks(
    corpus_dir: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Memindai folder data/corpus/raw/, mengekstrak teks tiap berkas yang valid,
    dan memotongnya menjadi segmen DocumentChunk ber-overlap.
    """
    if corpus_dir is None:
        root = Path(__file__).resolve().parents[2]
        corpus_dir = root / "data" / "corpus" / "raw"

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Folder korpus tidak ditemukan di: {corpus_dir}")

    all_chunks: List[DocumentChunk] = []
    files = sorted(corpus_dir.iterdir())

    # Filter: jika ada .txt yang sepadan dengan .html, abaikan .html agar tidak duplikat
    txt_stems = {f.stem for f in files if f.suffix.lower() == ".txt"}

    doc_counter = 0

    for file_path in files:
        if not file_path.is_file():
            continue

        fname_lower = file_path.name.lower()

        # Lewati berkas kuliah umum
        if fname_lower in EXCLUDED_FILES:
            continue

        # Lewati .html jika sudah ada versi .txt bersihnya
        if file_path.suffix.lower() == ".html" and file_path.stem in txt_stems:
            continue

        # Ekstrak teks
        raw_text = extract_text_from_file(file_path)
        clean = clean_text(raw_text)

        if len(clean) < 50:
            continue

        doc_counter += 1
        category, doc_title = classify_document(file_path.name)
        text_chunks = recursive_character_chunking(clean, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        doc_prefix = f"DOC_{doc_counter:02d}"

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc_prefix}_{idx:03d}"
            metadata = {
                "chunk_id": chunk_id,
                "doc_id": doc_prefix,
                "source": file_path.name,
                "title": doc_title,
                "category": category,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "char_length": len(chunk_text),
                "word_count": len(chunk_text.split()),
            }
            all_chunks.append(DocumentChunk(chunk_id=chunk_id, content=chunk_text, metadata=metadata))

    return all_chunks

