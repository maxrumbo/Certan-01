"""
generate_m2_visualizations.py
-----------------------------
Menghasilkan visualisasi grafis Milestone 2:
  1. reports/figures/m2_courier_allocation_map.png
       Peta topologi jaringan jalan koridor Laguboti - Balige yang menampilkan
       titik-titik pengiriman paket dengan kode visual warna kurir yang ditugaskan.
  2. reports/figures/m2_load_breakdown_chart.png
       Grafik batang perbandingan muatan riil vs kapasitas maksimum per kurir.

Cara menjalankan:
  uv run python scripts/generate_m2_visualizations.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from certan_01.osm_loader import LANDMARKS
from certan_01.solver import (
    CSPSolver,
    create_default_courier_fleet,
    load_real_ecommerce_packages,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_courier_allocation_map() -> None:
    """
    Membuat peta 2D sebaran paket dan alokasi armada kurir di koridor Laguboti - Balige.
    """
    fleet = create_default_courier_fleet(3)
    packages = load_real_ecommerce_packages(limit=15)

    solver = CSPSolver(packages, fleet)
    result = solver.solve()
    assert result is not None, "Solver harus menemukan solusi untuk 15 paket"

    # Palet warna per kurir (berbasis kontras hitam-putih / grayscale aman)
    courier_colors = {
        "K1_BEBEK": "#2563EB",  # Biru
        "K2_MATIK": "#16A34A",  # Hijau
        "K3_BOX1":  "#DC2626",  # Merah
    }
    courier_markers = {
        "K1_BEBEK": "o",
        "K2_MATIK": "^",
        "K3_BOX1":  "s",
    }

    hub_coord = LANDMARKS["hub_jne_laguboti"]  # (lat, lon) -> (y, x)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot Hub JNE Laguboti
    ax.scatter(
        [hub_coord[1]], [hub_coord[0]],
        c="#FBBF24", edgecolors="#B45309", s=280, marker="*", zorder=10,
        label="Depot Hub Logistik (JNE Laguboti)"
    )
    ax.text(
        hub_coord[1], hub_coord[0] + 0.003, "DEPOT HUB (JNE Laguboti)",
        fontsize=10, fontweight="bold", ha="center", color="#78350F",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#F59E0B", alpha=0.9)
    )

    # Plot landmark dan garis penugasan paket
    plotted_labels = set()
    for pkg in packages:
        dest_key = pkg.destination_node
        if dest_key in LANDMARKS:
            lat, lon = LANDMARKS[dest_key]
        else:
            # Fallback jika landmark tidak ada
            lat, lon = (hub_coord[0] + 0.01, hub_coord[1] + 0.01)

        c_id = result.assignment[pkg.package_id]
        c_color = courier_colors.get(c_id, "#4B5563")
        c_marker = courier_markers.get(c_id, "o")
        c_obj = solver.courier_map[c_id]

        label = f"{c_obj.name} ({c_obj.vehicle_type})"
        lbl = label if label not in plotted_labels else ""
        if lbl:
            plotted_labels.add(label)

        # Garis alokasi dari hub ke paket (garis putus-putus tipis)
        ax.plot(
            [hub_coord[1], lon], [hub_coord[0], lat],
            color=c_color, linestyle=":", linewidth=1.2, alpha=0.6, zorder=2
        )

        # Marker titik paket
        ax.scatter(
            [lon], [lat],
            c=c_color, edgecolors="#111827", s=110, marker=c_marker, zorder=5,
            label=lbl if lbl else None
        )

        # Label paket ID dan berat
        bulky_tag = " [BULKY]" if pkg.is_bulky else ""
        ax.annotate(
            f"{pkg.package_id} ({pkg.weight_kg:.1f}kg){bulky_tag}",
            xy=(lon, lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
            color="#1F2937",
        )

    ax.set_title(
        "Peta Alokasi Paket ke Armada Kurir (Koridor Laguboti - Balige)\n"
        "Mesin Inferensi CSP: Hasil Pembagian 15 Paket Riil E-Commerce",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Garis Bujur / Longitude (OpenStreetMap)", fontsize=10, labelpad=8)
    ax.set_ylabel("Garis Lintang / Latitude (OpenStreetMap)", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "m2_courier_allocation_map.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  [OK] Peta alokasi kurir tersimpan: {output_path}")


def generate_load_breakdown_chart() -> None:
    """
    Membuat diagram batang muatan riil vs kapasitas maksimum per kurir pada 50 paket.
    """
    fleet = create_default_courier_fleet(5)
    packages = load_real_ecommerce_packages(limit=50)

    solver = CSPSolver(packages, fleet)
    result = solver.solve()
    assert result is not None

    names = [c.name.split(" (")[0] for c in fleet]
    vehicles = [c.vehicle_type for c in fleet]
    loads = [result.courier_loads[c.courier_id] for c in fleet]
    capacities = [c.max_capacity_kg for c in fleet]
    hours = [result.courier_times[c.courier_id] for c in fleet]

    x = np.arange(len(fleet))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Bar muatan terisi vs kapasitas
    rects1 = ax1.bar(x - width/2, loads, width, label="Muatan Riil Terisi (kg)", color="#0284C7", edgecolor="#0369A1")
    rects2 = ax1.bar(x + width/2, capacities, width, label="Kapasitas Maksimal (kg)", color="#94A3B8", alpha=0.6, edgecolor="#475569")

    # Anotasi muatan di atas bar
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f} kg", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_title("Analisis Utilisasi Beban Armada pada Skenario 50 Paket Riil", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Nama Kurir & Jenis Kendaraan", fontsize=10, labelpad=8)
    ax1.set_ylabel("Berat Muatan (Kilogram)", fontsize=10, labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n({v})" for n, v in zip(names, vehicles)], fontsize=9)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    output_path = FIGURES_DIR / "m2_load_breakdown_chart.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  [OK] Diagram beban muatan tersimpan: {output_path}")


def generate_package_distribution_chart() -> None:
    """
    Membuat visualisasi karakteristik dataset 50 paket riil e-commerce:
    Histogram distribusi berat dan proporsi kategori paket (reguler vs bulky).
    """
    packages = load_real_ecommerce_packages(limit=50)
    weights = [p.weight_kg for p in packages]
    is_bulky = [p.is_bulky for p in packages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Subplot 1: Histogram Berat Paket
    n_bins = 12
    counts, bins, patches = ax1.hist(weights, bins=n_bins, color="#475569", edgecolor="#1E293B", alpha=0.85)
    ax1.axvline(15.0, color="#D97706", linestyle="--", linewidth=1.5, label="Batas Bebek (15 kg)")
    ax1.axvline(20.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Batas Matik (20 kg)")

    ax1.set_title("Distribusi Berat Paket Riil E-Commerce", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Berat Paket (Kilogram)", fontsize=9.5, labelpad=6)
    ax1.set_ylabel("Frekuensi Paket", fontsize=9.5, labelpad=6)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Proporsi Kategori Reguler vs Bulky
    bulky_count = sum(1 for b in is_bulky if b)
    regular_count = len(is_bulky) - bulky_count
    categories = [f"Reguler / Motor\n({regular_count} paket)", f"Bulky / Mobil Box\n({bulky_count} paket)"]
    sizes = [regular_count, bulky_count]
    colors = ["#334155", "#94A3B8"]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=categories, autopct="%1.1f%%",
        startangle=140, colors=colors, textprops=dict(color="#0F172A", fontsize=9.5),
        wedgeprops=dict(edgecolor="#0F172A", linewidth=1)
    )
    for at in autotexts:
        at.set_color("#FFFFFF")
        at.set_fontweight("bold")

    ax2.set_title("Proporsi Kualifikasi Kendaraan Armada", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    output_path = FIGURES_DIR / "m2_package_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  [OK] Diagram distribusi paket tersimpan: {output_path}")


def generate_workload_balance_chart() -> None:
    """
    Membuat diagram keseimbangan beban kerja (durasi jam kerja dan jumlah paket)
    tiap kurir serta kepatuhan batas 8.0 jam kerja sesuai UU Ketenagakerjaan.
    """
    fleet = create_default_courier_fleet(5)
    packages = load_real_ecommerce_packages(limit=50)

    solver = CSPSolver(packages, fleet)
    result = solver.solve()
    assert result is not None

    names = [c.name.split(" (")[0] for c in fleet]
    hours = [result.courier_times[c.courier_id] for c in fleet]
    pkg_counts = [
        sum(1 for cid in result.assignment.values() if cid == c.courier_id)
        for c in fleet
    ]

    x = np.arange(len(fleet))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.5), dpi=300, sharex=True)

    # Panel 1: Jam Kerja vs Regulasi
    bars1 = ax1.bar(x, hours, color="#334155", width=0.45, edgecolor="#0F172A")
    ax1.axhline(8.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Batas Legal UU (8.0 Jam)")
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f} j", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_title("Evaluasi Kepatuhan Regulasi Jam Kerja Maksimal (Maks. 8.0 Jam)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Durasi Kerja (Jam)", fontsize=9.5, labelpad=6)
    ax1.set_ylim(0, 10)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Jumlah Paket per Kurir
    bars2 = ax2.bar(x, pkg_counts, color="#64748B", width=0.45, edgecolor="#0F172A")
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{int(h)} pkt", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_title("Distribusi Jumlah Paket per Personel Armada", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Banyak Paket", fontsize=9.5, labelpad=6)
    ax2.set_xlabel("Personel Armada Kurir", fontsize=9.5, labelpad=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    output_path = FIGURES_DIR / "m2_courier_workload_balance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  [OK] Diagram beban kerja armada tersimpan: {output_path}")


def main() -> None:
    print("\n" + "=" * 65)
    print("  MEMBUAT VISUALISASI TAMBAHAN MILESTONE 2")
    print("=" * 65)
    generate_courier_allocation_map()
    generate_load_breakdown_chart()
    generate_package_distribution_chart()
    generate_workload_balance_chart()
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

