"""
run_dispatch_demo.py
--------------------
Demo integrasi M2 (CSP) -> M1 (A*) via orchestrator.py.

Menjalankan pipeline lengkap:
  1. CSP mengalokasikan 5 paket ke 3 kurir (Motor Bebek + Motor Matik + Mobil Box)
  2. A* menghitung rute BBM-optimal hub -> setiap tujuan paket

Cara menjalankan:
  uv run python scripts/run_dispatch_demo.py
"""

from certan_01.orchestrator import dispatch_and_route
from certan_01.solver import Courier, Package, create_default_courier_fleet

# -- Dataset Paket (campuran landmark + koordinat GPS) -----------------------
packages = [
    Package("P01", destination_node="it_del_sitoluama",   weight_kg=2.5,  is_bulky=False, estimated_time_hrs=0.4),
    Package("P02", destination_node="sma_unggul_del",     weight_kg=1.2,  is_bulky=False, estimated_time_hrs=0.4),
    Package("P03", destination_node=(2.3340, 99.0665),    weight_kg=3.8,  is_bulky=False, estimated_time_hrs=0.5),  # Mixue Balige via GPS
    Package("P04", destination_node="pasar_laguboti",     weight_kg=8.0,  is_bulky=False, estimated_time_hrs=0.6),
    Package("P05", destination_node="rs_hkbp_balige",     weight_kg=22.0, is_bulky=True,  estimated_time_hrs=0.8),  # Bulky -> hanya Mobil Box
]

# -- Armada Kurir -------------------------------------------------------------
fleet = create_default_courier_fleet(fleet_size=3)

# -- Pipeline Terintegrasi M2 -> M1 -------------------------------------------
plan = dispatch_and_route(
    packages=packages,
    couriers=fleet,
    hub_landmark="hub_jne_laguboti",
    vehicle="motor_matik_kurir",
    verbose=True,
)

# -- Output Ringkasan ---------------------------------------------------------
plan.print_summary(verbose=True)
print(f"Semua paket berhasil dirouting: {plan.all_packages_routed}")
