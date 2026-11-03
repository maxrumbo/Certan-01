"""
generate_architecture_diagram.py
--------------------------------
Membuat Diagram Alur Arsitektur Dua Lapisan Sistem Logistik:
Lapisan 1 (Inference CSP: AC-3 & Backtracking MRV) -> Lapisan 2 (Pathfinding A*)
Pola 'Cluster-First, Route-Second' untuk Gambar 1.1 Laporan.
Format: Bersih, akademis, kontras tinggi, aman untuk cetak hitam-putih (grayscale).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports" / "figures" / "arsitektur_dua_lapisan_csp_astar.png"

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12.5, 8.0), dpi=300)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Helper: Draw box (pad=0 ensures exact dimensions)
    def draw_box(x, y, w, h, rounding_size=0.1, 
                 facecolor="#FFFFFF", edgecolor="#1E293B", linewidth=1.2, zorder=2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding_size}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder
        )
        ax.add_patch(box)
        return box

    # Helper: Text
    def draw_text(x, y, text, fontsize=9.0, fontweight="normal", color="#0F172A", 
                  ha="center", va="center", zorder=4):
        return ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight, color=color,
                       ha=ha, va=va, zorder=zorder, family="sans-serif")

    # Helper: Horizontal Arrow
    def draw_h_arrow(x1, y1, x2, y2, label="", color="#1E293B", width=2.0):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            color=color,
            linewidth=width,
            zorder=5
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, y1 + 0.35
            ax.text(mx, my, label, fontsize=7.8, fontweight="bold", color="#0F172A",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.08", 
                              facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=0.8))

    # ==================== 1. HEADER UTAMA ====================
    draw_box(0.5, 7.35, 11.5, 0.70, rounding_size=0.08, facecolor="#0F172A", edgecolor="#0F172A", linewidth=1)
    draw_text(6.25, 7.78, "ARSITEKTUR DUA LAPISAN SISTEM LOGISTIK CERDAS", 
              fontsize=11.5, fontweight="bold", color="#FFFFFF")
    draw_text(6.25, 7.52, "Pola 'Cluster-First, Route-Second': Integrasi Mesin Inferensi CSP (Lapisan 1) dan Mesin Pencarian Rute A* (Lapisan 2)", 
              fontsize=8.5, fontweight="normal", color="#CBD5E1")

    # ==================== 2. TIGA KOLOM UTAMA ====================
    y_col = 2.15
    col_h = 4.95

    # ----- KOLOM 1: MASUKAN OPERASIONAL (x: 0.5 - 2.8, w=2.3) -----
    draw_box(0.5, y_col, 2.3, col_h, rounding_size=0.12, facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.5)
    # Header Kolom 1 (Terintegrasi Judul + Subjudul)
    draw_box(0.65, 6.30, 2.0, 0.65, rounding_size=0.08, facecolor="#334155", edgecolor="#334155", linewidth=1)
    draw_text(1.65, 6.72, "MASUKAN SISTEM", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    draw_text(1.65, 6.48, "Data Operasional Depot", fontsize=7.2, color="#CBD5E1")

    # Card 1.1: Data Paket
    draw_box(0.65, 4.35, 2.0, 1.80, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
    draw_text(1.65, 5.92, "Manifest Paket", fontsize=8.2, fontweight="bold", color="#0F172A")
    draw_text(1.65, 5.62, "• 50 Paket Riil (OSM)", fontsize=7.2, color="#334155")
    draw_text(1.65, 5.34, "• Berat Fisik (wi)", fontsize=7.2, color="#334155")
    draw_text(1.65, 5.06, "• Estimasi Waktu (ti)", fontsize=7.2, color="#334155")
    draw_text(1.65, 4.78, "• Penanda Bulky (bi)", fontsize=7.2, color="#334155")
    draw_text(1.65, 4.52, "• Koordinat Alamat", fontsize=7.2, color="#334155")

    # Card 1.2: Data Armada
    draw_box(0.65, 2.30, 2.0, 1.90, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
    draw_text(1.65, 3.95, "Spesifikasi Armada", fontsize=8.2, fontweight="bold", color="#0F172A")
    draw_text(1.65, 3.65, "• K1: Motor Bebek (15 kg)", fontsize=7.2, color="#334155")
    draw_text(1.65, 3.37, "• K2: Motor Matik (20 kg)", fontsize=7.2, color="#334155")
    draw_text(1.65, 3.10, "• K3: Mobil Box (200 kg)", fontsize=7.2, color="#334155")
    draw_text(1.65, 2.82, "• Jam Kerja Maks 8 Jam", fontsize=7.2, color="#334155")
    draw_text(1.65, 2.55, "• Kualifikasi Bulky Box", fontsize=7.2, color="#334155")

    # ----- KOLOM 2: LAPISAN 1 (CSP INFERENCE) (x: 3.5 - 7.6, w=4.1) -----
    draw_box(3.5, y_col, 4.1, col_h, rounding_size=0.12, facecolor="#F8FAFC", edgecolor="#1D4ED8", linewidth=1.8)
    # Header Kolom 2
    draw_box(3.65, 6.30, 3.8, 0.65, rounding_size=0.08, facecolor="#1D4ED8", edgecolor="#1D4ED8", linewidth=1)
    draw_text(5.55, 6.72, "LAPISAN 1: MESIN INFERENSI BATASAN (CSP)", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    draw_text(5.55, 6.48, "Tahap Alokasi Muatan & Partisi Armada (Cluster-First)", fontsize=7.2, color="#DBEAFE")

    # Card 2.1: Formulasi Matematis Triplet
    draw_box(3.65, 5.05, 3.8, 1.10, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.0)
    draw_text(5.55, 5.92, "Formulasi Matematis Triplet P = <X, D, C>", fontsize=8.0, fontweight="bold", color="#1E3A8A")
    draw_text(5.55, 5.66, "X = {P1, ..., Pn}   |   D(Pi) = {K1, ..., Km}", fontsize=7.5, fontweight="bold", color="#2563EB")
    draw_text(5.55, 5.42, "C1: Total Muatan <= Kapasitas Kendaraan (15/20/200 kg)", fontsize=7.0, color="#334155")
    draw_text(5.55, 5.20, "C2: Paket Bulky -> Mobil Box  |  C3: Total Waktu <= 8 Jam", fontsize=7.0, color="#334155")

    # Card 2.2: Algoritma Solver
    draw_box(3.65, 3.65, 3.8, 1.25, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.0)
    draw_text(5.55, 4.68, "Algoritma Pemecah Batasan (Constraint Solver)", fontsize=8.0, fontweight="bold", color="#1E3A8A")
    draw_text(5.55, 4.42, "1. Propagasi Batasan AC-3 (Arc Consistency Algorithm #3)", fontsize=7.2, color="#0F172A")
    draw_text(5.55, 4.22, "   • Pemangkasan apriori domain variabel yang tidak konsisten", fontsize=6.8, color="#475569")
    draw_text(5.55, 4.00, "2. Backtracking Search dengan Heuristik MRV + LCV", fontsize=7.2, color="#0F172A")
    draw_text(5.55, 3.80, "   • Fail-first principle & proteksi kapasitas armada besar", fontsize=6.8, color="#475569")

    # Card 2.3: Hasil Alokasi Terverifikasi
    draw_box(3.65, 2.30, 3.8, 1.20, rounding_size=0.08, facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.2)
    draw_text(5.55, 3.28, "Hasil Lapisan 1: Alokasi Legal 100% Terverifikasi", fontsize=8.0, fontweight="bold", color="#1E3A8A")
    draw_text(5.55, 3.04, "✓ Seluruh Hard Constraints Terpenuhi (Zero Violations)", fontsize=7.2, color="#1D4ED8")
    draw_text(5.55, 2.80, "✓ Terbentuk Klaster Paket Legal: K1 (15 kg), K2 (20 kg), K3 (200 kg)", fontsize=6.8, color="#0F172A")
    draw_text(5.55, 2.54, "[Partisi Paket Siap Diteruskan ke Modul Perutean]", fontsize=7.0, fontweight="bold", color="#1E40AF")

    # ----- KOLOM 3: LAPISAN 2 (A* ROUTING) (x: 8.3 - 12.0, w=3.7) -----
    draw_box(8.3, y_col, 3.7, col_h, rounding_size=0.12, facecolor="#F8FAFC", edgecolor="#15803D", linewidth=1.8)
    # Header Kolom 3
    draw_box(8.45, 6.30, 3.4, 0.65, rounding_size=0.08, facecolor="#15803D", edgecolor="#15803D", linewidth=1)
    draw_text(10.15, 6.72, "LAPISAN 2: MESIN PENCARIAN RUTE (A*)", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    draw_text(10.15, 6.48, "Tahap Penentuan Trajektori Jalan Nyata (Route-Second)", fontsize=7.2, color="#DCFCE7")

    # Card 3.1: Topologi Graf OSM
    draw_box(8.45, 5.05, 3.4, 1.10, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#86EFAC", linewidth=1.0)
    draw_text(10.15, 5.92, "Topologi Graf Jaringan Jalan (OSM)", fontsize=8.0, fontweight="bold", color="#14532D")
    draw_text(10.15, 5.66, "Peta Koridor Laguboti – Balige (Kab. Toba)", fontsize=7.3, fontweight="bold", color="#15803D")
    draw_text(10.15, 5.42, "12 Simpul Utama & 17 Ruas Jalan Riil Antar-Depot", fontsize=7.0, color="#334155")
    draw_text(10.15, 5.20, "Representasi Struktur Data Graf Adjasensi Berbobot", fontsize=6.8, color="#475569")

    # Card 3.2: Algoritma A*
    draw_box(8.45, 3.65, 3.4, 1.25, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#86EFAC", linewidth=1.0)
    draw_text(10.15, 4.68, "Algoritma Pathfinding Optimal (A* Search)", fontsize=8.0, fontweight="bold", color="#14532D")
    draw_text(10.15, 4.42, "1. Fungsi Evaluasi Biaya: f(n) = g(n) + h(n)", fontsize=7.2, color="#0F172A")
    draw_text(10.15, 4.22, "   • g(n): Biaya akumulasi jarak tempuh fisik riil", fontsize=6.8, color="#475569")
    draw_text(10.15, 4.00, "2. Heuristik Admisibel Euclidean: h(n) <= h*(n)", fontsize=7.2, color="#0F172A")
    draw_text(10.15, 3.80, "   • Menjamin penemuan jalur terpendek global", fontsize=6.8, color="#475569")

    # Card 3.3: Hasil Trajektori Optimal
    draw_box(8.45, 2.30, 3.4, 1.20, rounding_size=0.08, facecolor="#F0FDF4", edgecolor="#22C55E", linewidth=1.2)
    draw_text(10.15, 3.28, "Hasil Lapisan 2: Rute Fisik Terbaik per Kurir", fontsize=8.0, fontweight="bold", color="#14532D")
    draw_text(10.15, 3.04, "✓ Urutan Kunjungan Stop Pengiriman Paling Efisien", fontsize=7.2, color="#15803D")
    draw_text(10.15, 2.80, "✓ Penghematan Total Jarak Tempuh & Konsumsi BBM", fontsize=7.0, color="#334155")
    draw_text(10.15, 2.54, "[Navigasi Turn-by-Turn Siap Dilaksanakan Kurir]", fontsize=7.0, fontweight="bold", color="#15803D")

    # ==================== PANAH ANTAR KOLOM ====================
    # Masukan -> Lapisan 1
    draw_h_arrow(2.80, 4.80, 3.50, 4.80, label="Input Data", color="#1E293B", width=2.2)

    # Lapisan 1 -> Lapisan 2
    draw_h_arrow(7.60, 4.80, 8.30, 4.80, label="Paket per Kurir", color="#1D4ED8", width=2.2)

    # ==================== 3. BARIS BAWAH: LUARAN SISTEM ====================
    draw_box(0.5, 0.40, 11.5, 1.45, rounding_size=0.12, facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.5)
    # Header baris bawah
    draw_text(6.25, 1.62, "LUARAN AKHIR SISTEM (OPERASIONAL LAPANGAN)", fontsize=8.8, fontweight="bold", color="#0F172A")

    # 3 Kotak Hasil
    draw_box(0.7, 0.52, 3.5, 0.90, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
    draw_text(2.45, 1.20, "1. Manifest Tugas Kurir", fontsize=8.2, fontweight="bold", color="#1E293B")
    draw_text(2.45, 0.96, "Daftar muatan per paket sesuai kapasitas", fontsize=7.2, color="#475569")
    draw_text(2.45, 0.74, "resmi motor/mobil tanpa kelebihan beban.", fontsize=7.2, color="#475569")

    draw_box(4.5, 0.52, 3.5, 0.90, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
    draw_text(6.25, 1.20, "2. Peta & Navigasi Interaktif", fontsize=8.2, fontweight="bold", color="#1E293B")
    draw_text(6.25, 0.96, "Peta rute interaktif Folium HTML dan", fontsize=7.2, color="#475569")
    draw_text(6.25, 0.74, "tautan Google Maps per segmen jalan kurir.", fontsize=7.2, color="#475569")

    draw_box(8.3, 0.52, 3.5, 0.90, rounding_size=0.08, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
    draw_text(10.05, 1.20, "3. Chatbot Asisten Pengiriman", fontsize=8.2, fontweight="bold", color="#1E293B")
    draw_text(10.05, 0.96, "Layanan asisten kurir untuk cek rute,", fontsize=7.2, color="#475569")
    draw_text(10.05, 0.74, "urutan paket, dan estimasi waktu sampai.", fontsize=7.2, color="#475569")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Diagram sempurna berhasil dibuat: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_architecture_diagram()
