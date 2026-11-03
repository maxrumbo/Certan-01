"""
generate_m3_figures.py
----------------------
Menghasilkan visualisasi resmi untuk Laporan Milestone 3:
1. m3_retrieval_benchmark_chart.png : Grafik skor Cosine Similarity, Latensi, dan Ambang Batas Penolakan (tau = 0.72).
2. m3_corpus_distribution.png       : Distribusi 2.721 chunks korpus pada 12 kategori domain bisnis.
3. m3_arsitektur_tiga_lapisan.png   : Diagram Arsitektur Terintegrasi 3 Lapisan (M3 Knowledge Base -> M2 CSP -> M1 A*).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_benchmark_chart():
    """Grafik batang horizontal Cosine Similarity & Ambang Batas Penolakan."""
    queries = [
        "Q01: Klaim Paket Rusak J&T\n(SOP Ekspedisi)",
        "Q02: Kebolehan HAZMAT Aki/Gas\n(Dangerous Goods)",
        "Q03: Jam Kerja Kurir 8 Jam\n(Regulasi UU 13/2003 / M2-Bridge)",
        "Q04: Prosedur Retur COD Gagal\n(Operasional Lapangan)",
        "Q05: SLA Pos Express vs Kilat\n(Kontrak Layanan Pos)",
        "Q06: Duit Ganti Rugi Hilang\n(Bahasa Kasual / Sinonim Slang)",
        "Q07: Resep Kue Bolu Pandan\n(Out-of-Domain / OOD)",
    ]
    similarities = [0.7438, 0.5752, 0.7136, 0.6490, 0.7022, 0.6515, 0.6890]
    latencies = [39.22, 23.19, 24.96, 18.95, 19.71, 20.85, 19.94]
    
    # Warna: Biru untuk in-domain, Oranye kemerahan bergaris untuk OOD
    colors = ["#1E40AF", "#2563EB", "#0D9488", "#0284C7", "#3B82F6", "#6366F1", "#DC2626"]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    y_pos = np.arange(len(queries))
    bars = ax.barh(y_pos, similarities, height=0.55, color=colors, edgecolor="#0F172A", linewidth=1.1, zorder=3)

    # Garis Ambang Batas Penolakan Terkalibrasi (tau = 0.72)
    threshold = 0.72
    ax.axvline(x=threshold, color="#B91C1C", linestyle="--", linewidth=1.8, zorder=4,
               label="Ambang Batas Penolakan Terkalibrasi (tau = 0.72)")

    # Nilai di ujung batang
    for i, (bar, sim, lat) in enumerate(zip(bars, similarities, latencies)):
        width = bar.get_width()
        text_label = f" {sim:.4f}  ({lat:.1f} ms)"
        if i == 6:
            text_label += " [Efek Anisotropy]"
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, text_label,
                va="center", ha="left", fontsize=9.2, fontweight="bold",
                color="#DC2626" if i == 6 else "#0F172A", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(queries, fontsize=9.5, fontweight="medium", color="#0F172A")
    ax.invert_yaxis()  # Q01 di atas

    ax.set_xlim(0.0, 0.95)
    ax.set_xlabel("Skor Kemiripan Kosinus (Cosine Similarity)", fontsize=10.5, fontweight="bold", color="#0F172A", labelpad=8)
    ax.set_title("Benchmark Temu Balik Semantik ChromaDB & Evaluasi Kueri Bisnis (Milestone 3)\n"
                 "Evaluasi Kueri Relevan, Variasi Bahasa Informal, dan Kueri Out-of-Domain",
                 fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)

    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#94A3B8", zorder=1)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

    plt.tight_layout()
    output_file = FIGURES_DIR / "m3_retrieval_benchmark_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Berhasil membuat: {output_file}")


def generate_corpus_distribution():
    """Grafik distribusi 2.721 chunks pada 12 kategori bisnis."""
    data = {
        "Referensi Logistik & Akademik": 814,
        "Taksonomi & Standar Operasional": 546,
        "Regulasi Pemerintah (UU & Permen)": 445,
        "SOP Ekspedisi (J&T, JNE, SiCepat)": 268,
        "Kontrak Layanan Pos (BKN & PPATK)": 204,
        "Klaim Asuransi & Ganti Rugi": 103,
        "Optimasi Distribusi Industri": 88,
        "Operasional Lapangan Kurir": 86,
        "SOP Retur & Refund E-Commerce": 55,
        "Benchmark Industri Logistik": 54,
        "Kebijakan Marketplace (Tokopedia)": 41,
        "SOP Pergudangan & Depot Fulfillment": 17,
    }

    categories = list(data.keys())
    counts = list(data.values())

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    y_pos = np.arange(len(categories))
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(categories)))

    bars = ax.barh(y_pos, counts, height=0.6, color=colors, edgecolor="#1E293B", linewidth=1.0, zorder=3)

    for bar, count in zip(bars, counts):
        width = bar.get_width()
        pct = (count / 2721) * 100
        ax.text(width + 12, bar.get_y() + bar.get_height() / 2, f" {count} chunks ({pct:.1f}%)",
                va="center", ha="left", fontsize=8.8, fontweight="medium", color="#0F172A", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9.2, color="#0F172A")
    ax.invert_yaxis()
    ax.set_xlim(0, 960)
    ax.set_xlabel("Jumlah Potongan Dokumen (Chunks)", fontsize=10.5, fontweight="bold", color="#0F172A", labelpad=8)
    ax.set_title("Distribusi Korpus Enterprise CERTAN-01 Berdasarkan Kategori Bisnis\n"
                 "Total: 2.721 Chunks dari 28 Dokumen Resmi Otoritatif (Ukuran Chunk: 500 Karakter / Overlap: 100)",
                 fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)

    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#94A3B8", zorder=1)
    plt.tight_layout()
    output_file = FIGURES_DIR / "m3_corpus_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Berhasil membuat: {output_file}")


def generate_3layer_architecture():
    """Diagram Arsitektur Terintegrasi 3 Lapisan (M3 -> M2 -> M1)."""
    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=300)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    def draw_box(x, y, w, h, rounding_size=0.1, facecolor="#FFFFFF", edgecolor="#1E293B", linewidth=1.2, zorder=2):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
        ax.add_patch(box)
        return box

    def draw_text(x, y, text, fontsize=9.0, fontweight="normal", color="#0F172A", ha="center", va="center", zorder=4):
        return ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight, color=color,
                       ha=ha, va=va, zorder=zorder, family="sans-serif")

    def draw_v_arrow(x, y1, y2, label="", color="#1E293B", width=2.0):
        arrow = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=15,
                                color=color, linewidth=width, zorder=5)
        ax.add_patch(arrow)
        if label:
            ax.text(x + 0.15, (y1 + y2) / 2, label, fontsize=8.0, fontweight="bold", color="#0F172A",
                    ha="left", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.08", facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=0.8))

    # Header
    draw_box(0.5, 7.60, 11.5, 0.75, rounding_size=0.08, facecolor="#0F172A", edgecolor="#0F172A", linewidth=1)
    draw_text(6.25, 8.08, "ARSITEKTUR TIGA LAPISAN SISTEM LOGISTIK CERDAS CERTAN-01",
              fontsize=12.0, fontweight="bold", color="#FFFFFF")
    draw_text(6.25, 7.80, "Integrasi Basis Pengetahuan Semantik (M3), Mesin Inferensi CSP (M2), dan Mesin Perutean A* (M1)",
              fontsize=8.5, color="#CBD5E1")

    # Layer 1 (M3)
    draw_box(0.5, 5.10, 11.5, 2.20, rounding_size=0.10, facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.5)
    draw_box(0.7, 6.75, 11.1, 0.45, rounding_size=0.06, facecolor="#16A34A", edgecolor="#16A34A", linewidth=1)
    draw_text(6.25, 6.97, "LAPISAN 1: BASIS PENGETAHUAN KORPUS & VECTOR STORE CHROMADB (MILESTONE 3)",
              fontsize=9.5, fontweight="bold", color="#FFFFFF")

    draw_box(0.8, 5.25, 3.4, 1.35, facecolor="#FFFFFF", edgecolor="#86EFAC")
    draw_text(2.5, 6.35, "Korpus 28 Dokumen Otoritatif", fontsize=8.5, fontweight="bold", color="#166534")
    draw_text(2.5, 6.05, "• UU No. 13/2003 (Batas 8 Jam Naker)", fontsize=7.2, color="#1E293B")
    draw_text(2.5, 5.80, "• UU Pos 38/2009 & Permenkominfo", fontsize=7.2, color="#1E293B")
    draw_text(2.5, 5.55, "• S&K J&T, SiCepat, Tokopedia, K3", fontsize=7.2, color="#1E293B")

    draw_box(4.55, 5.25, 3.4, 1.35, facecolor="#FFFFFF", edgecolor="#86EFAC")
    draw_text(6.25, 6.35, "Pipeline Ingestion & Chunking", fontsize=8.5, fontweight="bold", color="#166534")
    draw_text(6.25, 6.05, "• Ekstraksi Multi-Format (PDF, DOCX)", fontsize=7.2, color="#1E293B")
    draw_text(6.25, 5.80, "• Recursive Chunking (500/100)", fontsize=7.2, color="#1E293B")
    draw_text(6.25, 5.55, "• Total 2.721 Chunks dengan Metadata", fontsize=7.2, color="#1E293B")

    draw_box(8.3, 5.25, 3.4, 1.35, facecolor="#FFFFFF", edgecolor="#86EFAC")
    draw_text(10.0, 6.35, "ChromaDB Persistent Search", fontsize=8.5, fontweight="bold", color="#166534")
    draw_text(10.0, 6.05, "• all-MiniLM-L6-v2 (384 Dimensi)", fontsize=7.2, color="#1E293B")
    draw_text(10.0, 5.80, "• HNSW Cosine Similarity Index", fontsize=7.2, color="#1E293B")
    draw_text(10.0, 5.55, "• Latensi Inferensi 23.8 ms (CPU)", fontsize=7.2, color="#1E293B")

    draw_v_arrow(6.25, 5.10, 4.40, label="Memvalidasi Batasan Legal, Volumetrik & K3", color="#0F172A", width=1.8)

    # Layer 2 (M2)
    draw_box(0.5, 2.70, 11.5, 1.70, rounding_size=0.10, facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.5)
    draw_box(0.7, 3.90, 11.1, 0.40, rounding_size=0.06, facecolor="#2563EB", edgecolor="#2563EB", linewidth=1)
    draw_text(6.25, 4.10, "LAPISAN 2: MESIN INFERENSI BATASAN ALOKASI ARMADA / CSP (MILESTONE 2)",
              fontsize=9.5, fontweight="bold", color="#FFFFFF")

    draw_box(0.8, 2.80, 5.2, 0.95, facecolor="#FFFFFF", edgecolor="#BFDBFE")
    draw_text(3.4, 3.52, "Propagasi Konsistensi AC-3", fontsize=8.2, fontweight="bold", color="#1E40AF")
    draw_text(3.4, 3.25, "Memangkas domain kurir yang melanggar muatan (30 vs 200 kg)", fontsize=7.2, color="#1E293B")
    draw_text(3.4, 3.00, "Menegakkan batas 8.0 jam kerja & syarat paket bulky secara apriori", fontsize=7.2, color="#1E293B")

    draw_box(6.5, 2.80, 5.2, 0.95, facecolor="#FFFFFF", edgecolor="#BFDBFE")
    draw_text(9.1, 3.52, "Backtracking Search Heuristik MRV & LCV", fontsize=8.2, fontweight="bold", color="#1E40AF")
    draw_text(9.1, 3.25, "Memilih paket paling terbatas terlebih dahulu (MRV)", fontsize=7.2, color="#1E293B")
    draw_text(9.1, 3.00, "Menugaskan kurir dengan fleksibilitas terbesar (LCV) -> 100% Solvable", fontsize=7.2, color="#1E293B")

    draw_v_arrow(6.25, 2.70, 2.00, label="Grup Alamat Pengiriman Legal per Kurir", color="#0F172A", width=1.8)

    # Layer 3 (M1)
    draw_box(0.5, 0.30, 11.5, 1.70, rounding_size=0.10, facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.5)
    draw_box(0.7, 1.50, 11.1, 0.40, rounding_size=0.06, facecolor="#D97706", edgecolor="#D97706", linewidth=1)
    draw_text(6.25, 1.70, "LAPISAN 3: MESIN PENCARIAN RUTE JALAN & EFISIENSI BBM / A* SEARCH (MILESTONE 1)",
              fontsize=9.5, fontweight="bold", color="#FFFFFF")

    draw_box(0.8, 0.40, 5.2, 0.95, facecolor="#FFFFFF", edgecolor="#FDE68A")
    draw_text(3.4, 1.12, "Topologi Jaringan Jalan OSM Laguboti-Balige", fontsize=8.2, fontweight="bold", color="#92400E")
    draw_text(3.4, 0.85, "1.436 simpul dan 3.258 busur jalan raya dan lokal", fontsize=7.2, color="#1E293B")
    draw_text(3.4, 0.60, "Model konsumsi BBM non-linear dipengaruhi elevasi dan kecepatan", fontsize=7.2, color="#1E293B")

    draw_box(6.5, 0.40, 5.2, 0.95, facecolor="#FFFFFF", edgecolor="#FDE68A")
    draw_text(9.1, 1.12, "A* Search Berbasis Heuristik Haversine", fontsize=8.2, fontweight="bold", color="#92400E")
    draw_text(9.1, 0.85, "Menghasilkan rute global terpendek yang terbukti optimal", fontsize=7.2, color="#1E293B")
    draw_text(9.1, 0.60, "Penghematan eksplorasi simpul hingga 54% dibandingkan UCS biasa", fontsize=7.2, color="#1E293B")

    output_file = FIGURES_DIR / "m3_arsitektur_tiga_lapisan.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Berhasil membuat: {output_file}")


def main():
    print("=" * 60)
    print("  MEMBUAT GAMBAR VISUALISASI LAPORAN MILESTONE 3")
    print("=" * 60)
    generate_benchmark_chart()
    generate_corpus_distribution()
    generate_3layer_architecture()
    print("[*] Selesai! Seluruh gambar tersimpan di reports/figures/")


if __name__ == "__main__":
    main()

