"""
test_solver.py
--------------
Suite pengujian unit otomatis untuk Mesin Inferensi Batasan (CSPSolver).
Menguji 4 kategori kasus wajib sesuai rubrik penilaian analitik:
  1. Kasus Normal: Semua paket berhasil dialokasikan secara legal.
  2. Kasus Over-Capacity (Ekstrem): Paket melebihi kapasitas armada -> Infeasible (None).
  3. Kasus Batas Tepat (Edge Case): Bobot paket pas di batas kapasitas kendaraan.
  4. Kasus Jam Kerja: Durasi paket melebihi batas jam kerja harian -> ditolak.
"""

import pytest

from certan_01.solver import (
    Courier,
    CSPSolver,
    Package,
    create_default_courier_fleet,
    generate_scenario_packages,
)


# ---------------------------------------------------------------------------
# Fixture Dasar
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_fleet():
    """Armada standar: Motor Bebek (15kg), Motor Matik (20kg), Mobil Box (200kg)."""
    return [
        Courier("K_BEBEK", "Budi", "Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
        Courier("K_MATIK", "Siti", "Motor Matik", max_capacity_kg=20.0, max_work_hours=8.0),
        Courier("K_BOX",   "Joko", "Mobil Box",   max_capacity_kg=200.0, max_work_hours=8.0),
    ]


# ---------------------------------------------------------------------------
# 1. Kasus Uji Normal (Feasible & Legal Allocation)
# ---------------------------------------------------------------------------

def test_normal_case_allocation(standard_fleet):
    """
    Kategori 1: Skenario operasional normal di mana seluruh paket
    berhasil dialokasikan tanpa pelanggaran batasan muatan maupun jam kerja.
    """
    packages = [
        Package("P01", "B", weight_kg=3.5, estimated_time_hrs=0.5),
        Package("P02", "C", weight_kg=5.0, estimated_time_hrs=0.5),
        Package("P03", "D", weight_kg=8.0, estimated_time_hrs=0.8),
        Package("P04", "E", weight_kg=12.0, estimated_time_hrs=1.0),
        Package("P05", "F", weight_kg=25.0, is_bulky=True, estimated_time_hrs=1.5),
    ]

    solver = CSPSolver(packages, standard_fleet)
    result = solver.solve()

    assert result is not None, "Solver harus menemukan solusi pada kasus normal"
    assert len(result.assignment) == len(packages)

    # Verifikasi formal
    is_valid, msg = solver.verify_assignment(result.assignment)
    assert is_valid, f"Verifikasi gagal: {msg}"

    # Paket bulky P05 wajib masuk ke Mobil Box
    assert result.assignment["P05"] == "K_BOX", "Paket bulky harus dialokasikan ke Mobil Box"


# ---------------------------------------------------------------------------
# 2. Kasus Uji Over-Capacity (Ekstrem / Infeasible)
# ---------------------------------------------------------------------------

def test_overcapacity_single_package(standard_fleet):
    """
    Kategori 2A: Terdapat paket dengan berat melebihi kapasitas kendaraan terbesar (250 kg > 200 kg).
    Solver harus menyatakan Infeasible (return None).
    """
    packages = [
        Package("P_GIANT", "B", weight_kg=250.0, is_bulky=True, estimated_time_hrs=2.0)
    ]

    solver = CSPSolver(packages, standard_fleet)
    result = solver.solve()

    assert result is None, "Solver harus return None jika ada paket melebihi batas armada terbesar"
    assert not solver.metrics.is_feasible


def test_overcapacity_fleet_saturation():
    """
    Kategori 2B: Akumulasi seluruh paket melebihi total daya tampung seluruh armada kurir yang tersedia.
    """
    fleet = [
        Courier("K1", "Rian", "Motor Bebek", max_capacity_kg=15.0),
        Courier("K2", "Dodi", "Motor Bebek", max_capacity_kg=15.0),
    ]
    # Total armada = 30 kg, muatan = 35 kg
    packages = [
        Package("P1", "B", weight_kg=12.0),
        Package("P2", "C", weight_kg=12.0),
        Package("P3", "D", weight_kg=11.0),
    ]

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is None, "Solver harus return None jika total muatan melampaui kapasitas seluruh kurir"


# ---------------------------------------------------------------------------
# 3. Kasus Uji Batas Tepat (Borderline / Edge Case)
# ---------------------------------------------------------------------------

def test_exact_boundary_capacity():
    """
    Kategori 3: Akumulasi berat paket tepat berada di batas limit kapasitas kendaraan (15.0 kg = 15.0 kg).
    Solver harus berhasil menemukan solusi legal.
    """
    fleet = [
        Courier("K_EXACT", "Andi", "Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0)
    ]
    packages = [
        Package("P1", "B", weight_kg=5.0, estimated_time_hrs=1.0),
        Package("P2", "C", weight_kg=6.0, estimated_time_hrs=1.0),
        Package("P3", "D", weight_kg=4.0, estimated_time_hrs=1.0),  # Total = 15.0 kg tepat
    ]

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is not None, "Solver harus berhasil jika total muatan pas di batas toleransi"
    assert result.courier_loads["K_EXACT"] == 15.0

    is_valid, _ = solver.verify_assignment(result.assignment)
    assert is_valid


# ---------------------------------------------------------------------------
# 4. Kasus Uji Pelanggaran Jam Kerja (Work-Hour Constraint)
# ---------------------------------------------------------------------------

def test_work_hour_overtime_rejected():
    """
    Kategori 4: Muatan tidak melebihi kapasitas, tetapi estimasi total waktu tempuh
    melebihi batas jam kerja harian maksimum (misal: total 9.0 jam > 8.0 jam).
    """
    fleet = [
        Courier("K_SOLO", "Tono", "Motor Matik", max_capacity_kg=20.0, max_work_hours=8.0)
    ]
    # Muatan hanya 10 kg (< 20 kg), tetapi waktu 3 x 3.0 = 9.0 jam (> 8.0 jam)
    packages = [
        Package("P1", "B", weight_kg=3.0, estimated_time_hrs=3.0),
        Package("P2", "C", weight_kg=3.0, estimated_time_hrs=3.0),
        Package("P3", "D", weight_kg=4.0, estimated_time_hrs=3.0),
    ]

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is None, "Solver harus menolak penugasan yang melanggar batas regulasi jam kerja"


# ---------------------------------------------------------------------------
# 5. Uji Kualifikasi Kendaraan (Bulky Constraint)
# ---------------------------------------------------------------------------

def test_bulky_package_vehicle_qualification():
    """
    Memastikan paket bertanda bulky tidak pernah dialokasikan ke kurir motor,
    bahkan jika beratnya ringan (misal 5 kg tapi bervolume besar).
    """
    fleet = [
        Courier("K_MOTOR", "Budi", "Motor Matik", max_capacity_kg=20.0, can_handle_bulky=False),
        Courier("K_MOBIL", "Joko", "Mobil Box",   max_capacity_kg=200.0, can_handle_bulky=True),
    ]
    packages = [
        Package("P_BULKY_LIGHT", "B", weight_kg=5.0, is_bulky=True, estimated_time_hrs=1.0)
    ]

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is not None
    assert result.assignment["P_BULKY_LIGHT"] == "K_MOBIL"


# ---------------------------------------------------------------------------
# 6. Uji Generator Skenario Skala Besar (Untuk Analisis Sensitivitas)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pkg_count, fleet_size", [(5, 2), (15, 3), (30, 5)])
def test_scenario_generator_and_metrics(pkg_count, fleet_size):
    """
    Menguji bahwa generator skenario menghasilkan paket yang valid
    dan solver mampu mencatat metrik waktu & backtrack secara konsisten.
    Sesuai panduan tugas: 5 paket (2 kurir), 15 paket (3 kurir), 30 paket (5 kurir).
    """
    fleet = create_default_courier_fleet(fleet_size)
    packages = generate_scenario_packages(pkg_count)

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is not None, f"Solver gagal menemukan solusi untuk skenario {pkg_count} paket ({fleet_size} kurir)"
    assert solver.metrics.execution_time_ms >= 0.0
    assert solver.metrics.total_packages == pkg_count
    assert solver.metrics.is_feasible is True

    is_valid, msg = solver.verify_assignment(result.assignment)
    assert is_valid, f"Solusi tidak valid: {msg}"


# ---------------------------------------------------------------------------
# 7. Uji Dataset Riil E-Commerce (Kaggle & Amazon Last-Mile Benchmark)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pkg_count, fleet_size", [(5, 2), (15, 3), (30, 5), (50, 5)])
def test_real_ecommerce_dataset_allocation(pkg_count, fleet_size):
    """
    Menguji performa solver pada dataset riil e-commerce koridor Laguboti-Balige
    (5, 15, 30, dan 50 paket) yang dimuat dari data/real_ecommerce_packages.json.
    """
    from certan_01.solver import load_real_ecommerce_packages

    fleet = create_default_courier_fleet(fleet_size)
    packages = load_real_ecommerce_packages(limit=pkg_count)

    assert len(packages) == pkg_count, f"Ekspektasi {pkg_count} paket terambil dari dataset riil"

    solver = CSPSolver(packages, fleet)
    result = solver.solve()

    assert result is not None, f"Solver harus menemukan solusi legal untuk {pkg_count} paket riil"
    assert len(result.assignment) == pkg_count

    # Verifikasi independen seluruh batasan fisik dan regulasi jam kerja
    is_valid, reason = solver.verify_assignment(result.assignment)
    assert is_valid, f"Pelanggaran batasan pada dataset riil {pkg_count} paket: {reason}"

    # Verifikasi khusus: seluruh paket bulky wajib masuk ke armada Mobil Box
    for p in packages:
        if p.is_bulky:
            assigned_courier = result.assignment[p.package_id]
            assert "box" in assigned_courier.lower(), (
                f"Paket bulky {p.package_id} teralokasi ke non-box: {assigned_courier}"
            )


