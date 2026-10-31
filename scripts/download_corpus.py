"""
download_corpus.py
------------------
Skrip untuk mengunduh dokumen SOP, regulasi, dan kebijakan operasional
logistik e-commerce secara otomatis ke data/corpus/raw/.
"""

import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path

# Setup SSL context toleran
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

DOCUMENTS = [
    # 1. SOP & Panduan Kemasan (Langsung PDF / DOCX)
    {
        "filename": "SOP_Penerimaan_Barang_Supplier.pdf",
        "url": "https://files.sop.gajihub.com/sops/75/sop-penerimaan-barang-dari-supplier.pdf",
        "type": "pdf",
    },
    {
        "filename": "SOP_Retur_Produk_Luar_Marketplace.pdf",
        "url": "https://files.sop.gajihub.com/sops/197/sop-retur-produk-di-luar-marketplace.pdf",
        "type": "pdf",
    },
    {
        "filename": "SOP_Penanganan_Retur_dan_Refund.docx",
        "url": "https://files.sop.gajihub.com/sops/84/sop-penanganan-retur-dan-refund.docx",
        "type": "docx",
    },
    {
        "filename": "SK_Gudang_Fulfillment_EZPickup.pdf",
        "url": "https://ezpickup.id/wp-content/uploads/tos/EZPickup-Terms-Update-Aug-6-2025.pdf",
        "type": "pdf",
    },
    {
        "filename": "Panduan_Kemasan_Ninja_Xpress.pdf",
        "url": "https://media.ninjavan.co/wp-content/uploads/2023/06/BAHASA-Tips-Membungkus-Barang.docx.pdf",
        "type": "pdf",
    },
    # 2. Regulasi & Kontrak Layanan Pos / Ekspedisi (PDF)
    {
        "filename": "PKS_PT_Pos_Indonesia_BKN.pdf",
        "url": "https://www.bkn.go.id/storage/2019/09/29.-PKS-KANTOR-POS-INDONESIA.pdf",
        "type": "pdf",
    },
    {
        "filename": "Surat_Perjanjian_Pos_Indonesia_PPATK.pdf",
        "url": "https://ppid.ppatk.go.id/wp-content/uploads/2024/10/Surat-Perjanjian-dengan-PT-Pos-Indonesia.pdf",
        "type": "pdf",
    },
    {
        "filename": "Permenkominfo_No_4_2021_Penyelenggaraan_Pos.pdf",
        "url": "https://peraturan.go.id/files/bn302-2021.pdf",
        "type": "pdf",
    },
    {
        "filename": "Permenaker_No_8_2010_Alat_Pelindung_Diri_Kurir.pdf",
        "url": "https://indolabourdatabase.files.wordpress.com/2018/03/permenaker-no-8-tahun-2010-tentang-apd.pdf",
        "type": "pdf",
    },
    {
        "filename": "UU_No_38_2009_Tentang_Pos.pdf",
        "url": "https://supplychainindonesia.com/download/UU_NO_38_2009.PDF",
        "type": "pdf",
    },
    # 3. Kebijakan & S&K Marketplace / Ekspedisi (Web HTML -> Simpan HTML + TXT Bersih)
    {
        "filename": "SK_Pengiriman_JnT_Express.html",
        "url": "https://jet.co.id/information/faq",
        "type": "html",
    },
    {
        "filename": "Kebijakan_Pengiriman_SiCepat_Tokopedia.html",
        "url": "https://www.tokopedia.com/help/article/sicepat",
        "type": "html",
    },
    {
        "filename": "SK_Asuransi_Pengiriman_Tokopedia.html",
        "url": "https://www.tokopedia.com/help/article/a-1715",
        "type": "html",
    },
    {
        "filename": "Shopee_Ganti_Rugi_Ekspedisi.html",
        "url": "https://shopee.co.id/inspirasi-shopee/cara-ganti-rugi-jasa-ekspedisi-yang-didukung-shopee-jika-paket-rusak-atau-hilang/",
        "type": "html",
    },
    {
        "filename": "FAQ_Retur_Paket_Tokopedia.html",
        "url": "https://www.tokopedia.com/help/article/faq-return-packages",
        "type": "html",
    },
]


def clean_html_to_text(html_content: str) -> str:
    """Mengubah dokumen HTML menjadi teks polos bersih bebas tag skrip dan CSS."""
    # Hapus script dan style
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Ubah tag break/p/div ke newline
    cleaned = re.sub(r"<(?:br|p|div|h[1-6]|li)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
    # Hapus seluruh sisa HTML tag
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Normalisasi spasi berlebih
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines)


def main():
    root_dir = Path(__file__).resolve().parents[1]
    target_dir = root_dir / "data" / "corpus" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Menyiapkan folder target: {target_dir}")
    print(f"[*] Memulai pengunduhan {len(DOCUMENTS)} dokumen referensi...")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for idx, doc in enumerate(DOCUMENTS, 1):
        target_path = target_dir / doc["filename"]
        url = doc["url"]
        doc_type = doc["type"]

        print(f"[{idx}/{len(DOCUMENTS)}] Mengunduh: {doc['filename']}")
        print(f"    URL: {url}")

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ctx, timeout=25) as response:
                data = response.read()

            with open(target_path, "wb") as f:
                f.write(data)

            size_kb = len(data) / 1024
            print(f"    -> SUKSES ({size_kb:.1f} KB)")

            # Jika file HTML, simpan juga versi .txt bersihnya agar langsung ramah embedding
            if doc_type == "html":
                try:
                    html_str = data.decode("utf-8", errors="ignore")
                    clean_txt = clean_html_to_text(html_str)
                    txt_path = target_path.with_suffix(".txt")
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(clean_txt)
                    print(f"    -> Ekstraksi teks bersih: {txt_path.name} ({len(clean_txt)} karakter)")
                except Exception as e_txt:
                    print(f"    -> Gagal mengekstrak teks bersih: {e_txt}")

            success_count += 1
        except Exception as e:
            print(f"    -> GAGAL: {e}")
            fail_count += 1

        print("-" * 70)

    print(f"[SELESAI] Total Berhasil: {success_count} | Total Gagal: {fail_count}")


if __name__ == "__main__":
    main()

