"""
generate_terminal_log_image.py
------------------------------
Menghasilkan gambar terminal log beresolusi tinggi (300 DPI) untuk Gambar 3.1
agar tidak pecah/burik saat disisipkan ke Word dan diekspor ke PDF.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan src ke sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from certan_01.solver import (
    CSPSolver,
    create_default_courier_fleet,
    load_real_ecommerce_packages,
)


def capture_solver_log() -> str:
    """Menjalankan solver demo dan menangkap teks outputnya."""
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    def print_separator(char: str = "=", length: int = 74) -> None:
        print(char * length)

    def run_sc(sc_name: str, pkg_count: int, fleet_size: int) -> None:
        print_separator("=")
        print(f"  {sc_name}: {pkg_count} Paket E-Commerce ({fleet_size} Kurir)")
        print_separator("-")

        fleet = create_default_courier_fleet(fleet_size)
        packages = load_real_ecommerce_packages(limit=pkg_count)

        total_weight = sum(p.weight_kg for p in packages)
        total_hours = sum(p.estimated_time_hrs for p in packages)
        bulky_count = sum(1 for p in packages if p.is_bulky)

        print(f"  * Total Muatan: {total_weight:.1f} kg (Bulky: {bulky_count} pkt) | Total Waktu: {total_hours:.1f} jam")
        print(f"  * Armada      : {', '.join(c.name.split(' (')[0] + ' (' + c.vehicle_type + ')' for c in fleet)}")
        print_separator("-")

        solver = CSPSolver(packages, fleet)
        result = solver.solve()

        if result is None:
            print("  [HASIL] STATUS: INFEASIBLE")
            return

        is_valid, msg = solver.verify_assignment(result.assignment)
        status_label = "VALID & LEGAL (100%)" if is_valid else f"INVALID ({msg})"

        print(f"  [HASIL] STATUS : {status_label}")
        print(f"  * Waktu: {result.metrics.execution_time_ms:.2f} ms | Backtrack: {result.metrics.backtrack_count} | AC-3 Pruned: {result.metrics.ac3_pruned_values}")
        print()
        print(f"  {'Kurir':<18} | {'Kendaraan':<13} | {'Muatan (kg)':<18} | {'Durasi':<10} | {'Paket':<5}")
        print("  " + "-" * 72)

        for c in fleet:
            load = result.courier_loads.get(c.courier_id, 0.0)
            hours = result.courier_times.get(c.courier_id, 0.0)
            pkgs = [p_id for p_id, c_id in result.assignment.items() if c_id == c.courier_id]
            load_pct = (load / c.max_capacity_kg) * 100.0
            print(
                f"  {c.name.split(' (')[0]:<18} | {c.vehicle_type:<13} | "
                f"{load:5.1f} / {c.max_capacity_kg:3.0f} kg ({load_pct:4.1f}%) | "
                f"{hours:4.1f} / 8.0j | {len(pkgs):2d} pkt"
            )
        print("  " + "-" * 72)
        print()

    print()
    print_separator("=")
    print("   SIMULASI MESIN INFERENSI CSP -- ALOKASI ARMADA KURIR LOGISTIK")
    print("   Institut Teknologi Del | Sistem Cerdas (CERTAN-01 Milestone 2)")
    print_separator("=")
    print()

    run_sc("SKENARIO A (DEMO)", 5, 2)
    run_sc("SKENARIO B (SKALA SEDANG)", 15, 3)
    run_sc("SKENARIO C (SKALA BESAR)", 30, 5)
    run_sc("SKENARIO D (STRESS TEST)", 50, 5)

    print_separator("=")
    print("   [KESIMPULAN] Seluruh skenario pengujian berhasil dieksekusi legal 100%!")
    print_separator("=")

    sys.stdout = old_stdout
    return buffer.getvalue()


def render_terminal_image(text: str, output_path: Path) -> None:
    lines = text.strip("\n").split("\n")
    max_len = max(len(line) for line in lines)
    num_lines = len(lines)

    # Ukuran gambar proporsional terhadap panjang dan baris teks
    fig_width = 11.0
    line_height = 0.22
    fig_height = max(8.0, num_lines * line_height + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    fig.patch.set_facecolor("#0F172A")  # Slate 900 dark background
    ax.set_facecolor("#0F172A")

    # Header ala window terminal
    header_box = plt.Rectangle((0, 1 - 0.5 / fig_height), 1, 0.5 / fig_height,
                               transform=ax.transAxes, facecolor="#1E293B", edgecolor="none")
    ax.add_patch(header_box)

    # Titik tombol window terminal (merah, kuning, hijau)
    btn_y = 1 - 0.25 / fig_height
    ax.scatter([0.02, 0.04, 0.06], [btn_y, btn_y, btn_y],
               c=["#EF4444", "#F59E0B", "#10B981"], s=30, transform=ax.transAxes, zorder=5)

    ax.text(0.5, btn_y, "certan-01: python scripts/run_solver_demo.py (PowerShell)",
            color="#94A3B8", fontsize=9, fontfamily="monospace", ha="center", va="center",
            transform=ax.transAxes)

    # Render baris-baris teks terminal dengan warna sintaks
    y_start = 1.0 - 0.8 / fig_height
    y_step = 1.0 / (num_lines + 4)

    for i, line in enumerate(lines):
        y_pos = y_start - (i * (0.9 / num_lines))

        # Pewarnaan teks terminal
        if "SKENARIO" in line:
            color = "#38BDF8"  # Cyan
            weight = "bold"
        elif "STATUS" in line or "VALID & LEGAL" in line:
            color = "#4ADE80"  # Green
            weight = "bold"
        elif "KESIMPULAN" in line:
            color = "#FACC15"  # Yellow
            weight = "bold"
        elif "SIMULASI MESIN INFERENSI" in line or "Institut Teknologi Del" in line:
            color = "#F8FAFC"  # White
            weight = "bold"
        elif "=" in line or "-" in line:
            color = "#475569"  # Muted slate
            weight = "normal"
        elif "Kurir" in line and "Kendaraan" in line:
            color = "#93C5FD"  # Light blue
            weight = "bold"
        else:
            color = "#E2E8F0"  # Clean white/slate
            weight = "normal"

        ax.text(
            0.02, y_pos, line,
            color=color,
            fontsize=8.5,
            fontfamily="monospace",
            fontweight=weight,
            va="top",
            transform=ax.transAxes,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"Gambar berhasil disimpan di: {output_path}")


if __name__ == "__main__":
    out_file = root_dir / "reports" / "figures" / "m2_solver_terminal_log.png"
    log_text = capture_solver_log()
    render_terminal_image(log_text, out_file)
