"""
run_sensitivity_benchmark.py
----------------------------
Skrip analisis sensitivitas dan evaluasi konvergensi komputasi Mesin Inferensi
CSP (CSPSolver) pada variasi ukuran masalah (5, 10, 20, 30, 50 paket e-commerce).

Luaran:
  1. Tabel metrik di terminal.
  2. Berkas data CSV : reports/sensitivity_results.csv
  3. Grafik ilmiah PNG : reports/figures/sensitivity_chart.png
                      (dan salinan di reports/sensitivity_chart.png)

Cara menjalankan:
  uv run python scripts/run_sensitivity_benchmark.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from certan_01.solver import (
    Courier,
    CSPSolver,
    create_default_courier_fleet,
    load_real_ecommerce_packages,
)

# ---------------------------------------------------------------------------
# Konfigurasi Path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT_PATH = REPORTS_DIR / "sensitivity_results.csv"
CHART_OUTPUT_PATH = FIGURES_DIR / "sensitivity_chart.png"
CHART_ROOT_COPY = REPORTS_DIR / "sensitivity_chart.png"


# ---------------------------------------------------------------------------
# Definisi Skenario Uji Sensitivitas
# ---------------------------------------------------------------------------

SCENARIOS = [
    {"label": "Skenario 5 Pkg",  "n_pkgs": 5,  "fleet_size": 2},
    {"label": "Skenario 10 Pkg", "n_pkgs": 10, "fleet_size": 3},
    {"label": "Skenario 20 Pkg", "n_pkgs": 20, "fleet_size": 3},
    {"label": "Skenario 30 Pkg", "n_pkgs": 30, "fleet_size": 5},
    {"label": "Skenario 50 Pkg", "n_pkgs": 50, "fleet_size": 5},
]

NUM_REPETITIONS = 7  # Pengulangan eksekusi untuk menghitung rerata waktu stabil


def run_benchmark() -> List[Dict]:
    print("\n" + "=" * 78)
    print("  EKSPERIMEN ANALISIS SENSITIVITAS & BENCHMARK KONVERGENSI CSP SOLVER")
    print("  Institut Teknologi Del | Sistem Cerdas (CERTAN-01 Milestone 2)")
    print("=" * 78)

    benchmark_rows: List[Dict] = []

    for sc in SCENARIOS:
        n = sc["n_pkgs"]
        fleet_size = sc["fleet_size"]
        label = sc["label"]

        fleet = create_default_courier_fleet(fleet_size)
        packages = load_real_ecommerce_packages(limit=n)

        total_weight = sum(p.weight_kg for p in packages)
        total_hours = sum(p.estimated_time_hrs for p in packages)
        bulky_count = sum(1 for p in packages if p.is_bulky)

        # Ukur stabilitas waktu komputasi
        execution_times: List[float] = []
        backtrack_counts: List[int] = []
        last_result = None

        for _ in range(NUM_REPETITIONS):
            solver = CSPSolver(packages, fleet)
            t0 = time.perf_counter()
            res = solver.solve()
            t_ms = (time.perf_counter() - t0) * 1000.0

            execution_times.append(t_ms)
            backtrack_counts.append(solver.metrics.backtrack_count)
            last_result = res

        avg_time = float(np.mean(execution_times))
        std_time = float(np.std(execution_times))
        avg_backtrack = int(np.mean(backtrack_counts))
        is_feasible = (last_result is not None)

        is_valid = False
        valid_msg = "N/A"
        if last_result is not None:
            is_valid, valid_msg = solver.verify_assignment(last_result.assignment)

        # Hitung utilitas muatan armada
        fleet_capacity = sum(c.max_capacity_kg for c in fleet)
        utilization_pct = (total_weight / fleet_capacity) * 100.0

        row = {
            "scenario": label,
            "packages_count": n,
            "fleet_size": fleet_size,
            "total_weight_kg": round(total_weight, 2),
            "bulky_packages": bulky_count,
            "total_hours": round(total_hours, 2),
            "status": "Feasible" if is_feasible else "Infeasible",
            "is_valid": is_valid,
            "avg_time_ms": round(avg_time, 3),
            "std_time_ms": round(std_time, 3),
            "backtrack_steps": avg_backtrack,
            "fleet_capacity_kg": fleet_capacity,
            "capacity_utilization_pct": round(utilization_pct, 1),
            "courier_loads": last_result.courier_loads if last_result else {},
            "courier_times": last_result.courier_times if last_result else {},
        }
        benchmark_rows.append(row)

        print(
            f"  [+] {label:<15} | N={n:2d} pkg | Fleet={fleet_size} | "
            f"W={total_weight:5.1f} kg | Time={avg_time:6.3f} ms (+/-{std_time:5.3f}) | "
            f"Backtracks={avg_backtrack} | Valid: {'YES' if is_valid else 'NO'}"
        )

    print("=" * 78 + "\n")
    return benchmark_rows


def export_to_csv(rows: List[Dict]) -> None:
    fieldnames = [
        "scenario",
        "packages_count",
        "fleet_size",
        "total_weight_kg",
        "bulky_packages",
        "total_hours",
        "status",
        "avg_time_ms",
        "std_time_ms",
        "backtrack_steps",
        "capacity_utilization_pct",
    ]

    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            clean_row = {k: r[k] for k in fieldnames}
            writer.writerow(clean_row)

    print(f"  [OK] Berkas CSV berhasil diekspor: {CSV_OUTPUT_PATH}")


def generate_sensitivity_plots(rows: List[Dict]) -> None:
    pkg_counts = [r["packages_count"] for r in rows]
    times = [r["avg_time_ms"] for r in rows]
    errors = [r["std_time_ms"] for r in rows]
    weights = [r["total_weight_kg"] for r in rows]
    utilizations = [r["capacity_utilization_pct"] for r in rows]

    # Setup visualisasi dengan tema profesional / jurnal ilmiah
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # -----------------------------------------------------------------------
    # Subplot 1: Konvergensi Waktu Komputasi vs Ukuran Masalah
    # -----------------------------------------------------------------------
    line_color = "#1E40AF"  # Royal Deep Blue
    accent_color = "#DC2626"  # Crimson Accent

    ax1.plot(
        pkg_counts,
        times,
        marker="o",
        markersize=8,
        color=line_color,
        linewidth=2.5,
        label="Waktu Eksekusi CSPSolver (ms)",
    )
    ax1.fill_between(
        pkg_counts,
        [max(0.0, t - e) for t, e in zip(times, errors)],
        [t + e for t, e in zip(times, errors)],
        alpha=0.2,
        color=line_color,
        label="Deviasi Standar (±1σ)",
    )

    # Anotasi angka waktu di setiap titik data
    for n, t in zip(pkg_counts, times):
        ax1.annotate(
            f"{t:.2f} ms",
            xy=(n, t),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2937",
        )

    ax1.set_title("Analisis Sensitivitas: Waktu Komputasi vs Ukuran Masalah", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Jumlah Paket Masukan (N)", fontsize=10.5, labelpad=8)
    ax1.set_ylabel("Waktu Eksekusi Solver (Milidetik)", fontsize=10.5, labelpad=8)
    ax1.set_xticks(pkg_counts)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # -----------------------------------------------------------------------
    # Subplot 2: Utilitas Muatan & Total Beban Armada
    # -----------------------------------------------------------------------
    bar_color = "#0D9488"  # Teal
    line_weight_color = "#D97706"  # Amber

    ax2_twin = ax2.twinx()

    bars = ax2.bar(
        [f"N={n}\n({r['fleet_size']} Kurir)" for n, r in zip(pkg_counts, rows)],
        utilizations,
        color=bar_color,
        width=0.45,
        alpha=0.85,
        label="Tingkat Utilitas Muatan (%)",
    )

    line2 = ax2_twin.plot(
        range(len(pkg_counts)),
        weights,
        color=line_weight_color,
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="-.",
        label="Total Muatan Barang (kg)",
    )

    # Anotasi persentase di atas bar
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#134E4A",
        )

    ax2.set_title("Efisiensi Muatan Armada & Total Bobot Logistik", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Skenario Uji (Jumlah Paket & Ukuran Armada)", fontsize=10.5, labelpad=8)
    ax2.set_ylabel("Tingkat Utilitas Armada (%)", fontsize=10.5, labelpad=8, color="#0D9488")
    ax2_twin.set_ylabel("Total Berat Paket (kg)", fontsize=10.5, labelpad=8, color="#D97706")
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Legend gabungan
    lines_labels_1 = [bars, line2[0]]
    labels_1 = ["Utilitas Armada (%)", "Total Berat (kg)"]
    ax2.legend(lines_labels_1, labels_1, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(CHART_OUTPUT_PATH, dpi=300)
    plt.savefig(CHART_ROOT_COPY, dpi=300)
    plt.close()

    print(f"  [OK] Grafik analisis sensitivitas tersimpan: {CHART_OUTPUT_PATH}")
    print(f"  [OK] Salinan di berkas laporan: {CHART_ROOT_COPY}")


def main() -> None:
    rows = run_benchmark()
    export_to_csv(rows)
    generate_sensitivity_plots(rows)
    print("\n  [SELESAI] Seluruh pengujian sensitivitas dan visualisasi berhasil dibuat!\n")


if __name__ == "__main__":
    main()
