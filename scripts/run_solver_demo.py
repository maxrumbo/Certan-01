"""
run_solver_demo.py
------------------
Skrip demonstrasi eksekusi Mesin Inferensi Batasan (CSPSolver) untuk alokasi
armada kurir pengiriman paket e-commerce di koridor Laguboti - Balige.

Menjalankan 4 skenario bertingkat:
  1. Skenario A: 5 Paket Riil   (Armada: 2 Motor)
  2. Skenario B: 15 Paket Riil  (Armada: 2 Motor + 1 Mobil Box)
  3. Skenario C: 30 Paket Riil  (Armada: 3 Motor + 2 Mobil Box)
  4. Skenario D: 50 Paket Riil  (Armada: 3 Motor + 2 Mobil Box - Full Stress Test)

Cara menjalankan:
  uv run python scripts/run_solver_demo.py
"""

from __future__ import annotations

import sys
from typing import List

from certan_01.solver import (
    Courier,
    CSPSolver,
    create_default_courier_fleet,
    load_real_ecommerce_packages,
)


def print_separator(char: str = "=", length: int = 76) -> None:
    print(char * length)


def run_scenario(scenario_name: str, pkg_count: int, fleet_size: int) -> None:
    print_separator("=")
    print(f"  {scenario_name}: {pkg_count} Paket E-Commerce ({fleet_size} Kurir)")
    print_separator("-")

    fleet = create_default_courier_fleet(fleet_size)
    packages = load_real_ecommerce_packages(limit=pkg_count)

    total_weight = sum(p.weight_kg for p in packages)
    total_hours = sum(p.estimated_time_hrs for p in packages)
    bulky_count = sum(1 for p in packages if p.is_bulky)

    print(f"  * Total Muatan Paket : {total_weight:.1f} kg (Bulky: {bulky_count} paket)")
    print(f"  * Total Estimasi Jam : {total_hours:.1f} jam")
    print(f"  * Armada Tersedia    : {', '.join(c.name for c in fleet)}")
    print_separator("-")

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    if result is None:
        print("  [HASIL] STATUS: INFEASIBLE (Tidak ada penugasan legal yang memenuhi batasan!)")
        print_separator("=")
        print()
        return

    is_valid, msg = solver.verify_assignment(result.assignment)
    status_label = "VALID & LEGAL (100%)" if is_valid else f"INVALID ({msg})"

    print(f"  [HASIL] STATUS      : {status_label}")
    print(f"  * Waktu Eksekusi    : {result.metrics.execution_time_ms:.2f} ms")
    print(f"  * Langkah Backtrack : {result.metrics.backtrack_count} langkah")
    print(f"  * Nilai Dipangkas   : {result.metrics.ac3_pruned_values} nilai (oleh AC-3)")
    print()
    print("  Rincian Alokasi per Kurir:")
    print("  " + "-" * 72)
    print(f"  {'Kurir':<22} | {'Kendaraan':<14} | {'Muatan (kg)':<14} | {'Durasi':<10} | {'Paket':<5}")
    print("  " + "-" * 72)

    for c in fleet:
        load = result.courier_loads.get(c.courier_id, 0.0)
        hours = result.courier_times.get(c.courier_id, 0.0)
        assigned_pkgs = [p_id for p_id, c_id in result.assignment.items() if c_id == c.courier_id]
        load_pct = (load / c.max_capacity_kg) * 100.0

        print(
            f"  {c.name:<22} | {c.vehicle_type:<14} | "
            f"{load:5.1f} / {c.max_capacity_kg:3.0f} kg ({load_pct:4.1f}%) | "
            f"{hours:4.1f} / 8.0 jam | {len(assigned_pkgs):2d} pkg"
        )

    print("  " + "-" * 72)
    print()


def main() -> None:
    print()
    print_separator("=")
    print("   SIMULASI MESIN INFERENSI CSP -- ALOKASI ARMADA KURIR LOGISTIK")
    print("   Institut Teknologi Del | Sistem Cerdas (CERTAN-01 Milestone 2)")
    print_separator("=")
    print()

    run_scenario("SKENARIO A (DEMO)", pkg_count=5, fleet_size=2)
    run_scenario("SKENARIO B (SKALA SEDANG)", pkg_count=15, fleet_size=3)
    run_scenario("SKENARIO C (SKALA BESAR)", pkg_count=30, fleet_size=5)
    run_scenario("SKENARIO D (STRES UJI REALISTIS)", pkg_count=50, fleet_size=5)

    print_separator("=")
    print("   Seluruh skenario pengujian berhasil dieksekusi dengan sukses!")
    print_separator("=")
    print()


if __name__ == "__main__":
    main()
