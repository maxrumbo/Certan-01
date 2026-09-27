"""
solver.py
---------
Mesin Inferensi Batasan (Constraint Satisfaction Problem Solver) untuk
alokasi armada kurir pengiriman paket e-commerce.

Komponen Utama:
  - Courier        : Model data kurir dan karakteristik kendaraannya.
  - Package        : Model data paket pengiriman dan karakteristik muatannya.
  - SolverMetrics  : Pencatat metrik konvergensi (waktu komputasi, langkah backtrack).
  - CSPSolver      : Solver CSP dengan algoritma AC-3 dan Backtracking MRV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model Data: Kurir & Paket
# ---------------------------------------------------------------------------

@dataclass
class Courier:
    """
    Model data kurir logistik dan spesifikasi armada kendaraannya.

    Attributes:
        courier_id      : Identifikasi unik kurir (misal: "K1", "K_BOX_1").
        name            : Nama kurir/pengemudi.
        vehicle_type    : Jenis kendaraan ("Motor Bebek", "Motor Matik", "Mobil Box").
        max_capacity_kg : Batas muatan aman maksimum dalam kilogram.
        max_work_hours  : Batas jam kerja harian resmi (standar 8.0 jam).
        can_handle_bulky: Apakah kendaraan mampu mengangkut paket besar/bulky.
    """

    courier_id: str
    name: str
    vehicle_type: str
    max_capacity_kg: float
    max_work_hours: float = 8.0
    can_handle_bulky: bool = False

    def __post_init__(self) -> None:
        # Mobil Box secara otomatis dapat membawa paket berukuran besar
        if "box" in self.vehicle_type.lower() or "mobil" in self.vehicle_type.lower():
            self.can_handle_bulky = True


@dataclass
class Package:
    """
    Model data paket pengiriman e-commerce.

    Attributes:
        package_id         : Identifikasi unik paket (misal: "P1", "P_EXP_05").
        destination_node   : Simpul lokasi tujuan pada graf jaringan jalan.
        weight_kg          : Berat paket dalam kilogram.
        is_bulky           : True jika paket berukuran besar (hanya bisa diangkut roda empat).
        estimated_time_hrs : Estimasi waktu tempuh & antar kurir untuk paket ini (jam).
    """

    package_id: str
    destination_node: str
    weight_kg: float
    is_bulky: bool = False
    estimated_time_hrs: float = 0.5


@dataclass
class SolverMetrics:
    """
    Pencatat metrik kinerja solver untuk analisis konvergensi & sensitivitas.
    """

    execution_time_ms: float = 0.0
    backtrack_count: int = 0
    assigned_count: int = 0
    total_packages: int = 0
    is_feasible: bool = False
    ac3_pruned_values: int = 0


@dataclass
class CSPAssignment:
    """
    Struktur data luaran hasil inferensi alokasi solver CSP.
    """

    assignment: Dict[str, str]  # {package_id: courier_id}
    courier_loads: Dict[str, float]  # {courier_id: total_kg}
    courier_times: Dict[str, float]  # {courier_id: total_hours}
    metrics: SolverMetrics


# ---------------------------------------------------------------------------
# Solver CSP (AC-3 + Backtracking MRV)
# ---------------------------------------------------------------------------

class CSPSolver:
    """
    Mesin Inferensi Batasan (CSP Solver) untuk alokasi paket ke armada kurir.

    Memodelkan formulasi CSP formal <X, D, C>:
      - X (Variabel) : Setiap paket pengiriman {P1, P2, ..., Pn}.
      - D (Domain)   : Himpunan kurir yang diizinkan untuk setiap paket.
      - C (Batasan)  : 
          1. Kapasitas Muatan (sum(weight) <= max_capacity_kg).
          2. Kualifikasi Kendaraan (bulky package hanya untuk Mobil Box).
          3. Jam Kerja Harian (sum(time) <= max_work_hours).
    """

    def __init__(
        self,
        packages: List[Package],
        couriers: List[Courier],
    ) -> None:
        self.packages = packages
        self.couriers = couriers

        self.package_map: Dict[str, Package] = {p.package_id: p for p in packages}
        self.courier_map: Dict[str, Courier] = {c.courier_id: c for c in couriers}

        self.variables: List[str] = [p.package_id for p in packages]

        # Inisialisasi domain D untuk setiap variabel X_i
        self.domains: Dict[str, List[str]] = {}
        for pkg in self.packages:
            possible_couriers: List[str] = []
            for c in self.couriers:
                # Filter kualifikasi kendaraan: paket bulky butuh kendaraan bulky
                if pkg.is_bulky and not c.can_handle_bulky:
                    continue
                # Filter muatan satuan: berat paket tunggal tidak boleh melebihi kapasitas kurir
                if pkg.weight_kg > c.max_capacity_kg:
                    continue
                possible_couriers.append(c.courier_id)
            self.domains[pkg.package_id] = possible_couriers

        self.metrics = SolverMetrics(total_packages=len(self.packages))

    # -----------------------------------------------------------------------
    # Validasi Batasan (Constraint Verification)
    # -----------------------------------------------------------------------

    def is_consistent(
        self,
        package_id: str,
        courier_id: str,
        current_assignment: Dict[str, str],
    ) -> bool:
        """
        Memeriksa apakah penugasan package_id -> courier_id konsisten terhadap
        seluruh batasan operasional (kapasitas berat & jam kerja).
        """
        target_courier = self.courier_map[courier_id]
        pkg_to_assign = self.package_map[package_id]

        # 1. Cek Kualifikasi Paket Bulky
        if pkg_to_assign.is_bulky and not target_courier.can_handle_bulky:
            return False

        # Hitung akumulasi muatan dan jam kerja kurir saat ini
        current_load = pkg_to_assign.weight_kg
        current_hours = pkg_to_assign.estimated_time_hrs

        for assigned_pkg_id, assigned_c_id in current_assignment.items():
            if assigned_c_id == courier_id:
                p = self.package_map[assigned_pkg_id]
                current_load += p.weight_kg
                current_hours += p.estimated_time_hrs

        # 2. Cek Batasan Kapasitas Muatan (Hard Constraint)
        if current_load > target_courier.max_capacity_kg:
            return False

        # 3. Cek Batasan Jam Kerja Harian (Hard Constraint)
        if current_hours > target_courier.max_work_hours:
            return False

        return True

    def verify_assignment(
        self,
        assignment: Dict[str, str],
    ) -> Tuple[bool, str]:
        """
        Memverifikasi kelayakan solusi akhir secara independen.
        Mengembalikan (True, 'Valid') atau (False, 'Alasan Pelanggaran').
        """
        if len(assignment) != len(self.variables):
            return False, f"Incomplete: {len(assignment)}/{len(self.variables)} paket teralokasi"

        load_tracker: Dict[str, float] = {c.courier_id: 0.0 for c in self.couriers}
        time_tracker: Dict[str, float] = {c.courier_id: 0.0 for c in self.couriers}

        for pkg_id, courier_id in assignment.items():
            if courier_id not in self.courier_map:
                return False, f"Kurir '{courier_id}' tidak terdaftar"

            pkg = self.package_map[pkg_id]
            courier = self.courier_map[courier_id]

            if pkg.is_bulky and not courier.can_handle_bulky:
                return False, f"Paket bulky '{pkg_id}' ditugaskan ke non-box '{courier_id}'"

            load_tracker[courier_id] += pkg.weight_kg
            time_tracker[courier_id] += pkg.estimated_time_hrs

        for c_id, total_load in load_tracker.items():
            max_cap = self.courier_map[c_id].max_capacity_kg
            if total_load > max_cap:
                return False, f"Kurir '{c_id}' kelebihan beban: {total_load:.1f}kg > {max_cap:.1f}kg"

        for c_id, total_time in time_tracker.items():
            max_hours = self.courier_map[c_id].max_work_hours
            if total_time > max_hours:
                return False, f"Kurir '{c_id}' lembur: {total_time:.1f}j > {max_hours:.1f}j"

        return True, "Solusi valid dan memenuhi seluruh batasan"

    # -----------------------------------------------------------------------
    # Propagasi Batasan AC-3 (Arc Consistency 3)
    # -----------------------------------------------------------------------

    def ac3(self) -> bool:
        """
        Menjalankan algoritma AC-3 untuk memangkas nilai domain kurir yang
        secara definitif tidak konsisten terhadap batasan global.
        Mengembalikan False jika terdeteksi domain kosong (Infeasible).
        """
        # Cek awal apakah ada paket yang domainnya kosong dari awal
        for var in self.variables:
            if len(self.domains[var]) == 0:
                return False

        # Inisialisasi antrean arc antara semua pasangan variabel
        queue: List[Tuple[str, str]] = []
        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if i != j:
                    queue.append((self.variables[i], self.variables[j]))

        while queue:
            xi, xj = queue.pop(0)
            if self._revise_arc(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                # Tambahkan tetangga kembali ke antrean
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return True

    def _revise_arc(self, xi: str, xj: str) -> bool:
        """
        Memeriksa apakah ada nilai di domain Xi yang tidak memiliki nilai pendukung di Xj.
        """
        revised = False
        pkg_i = self.package_map[xi]
        pkg_j = self.package_map[xj]

        couriers_to_remove: List[str] = []

        for c_id in self.domains[xi]:
            courier = self.courier_map[c_id]
            # Jika kedua paket ditugaskan ke kurir yang sama, cek apakah kapasitasnya cukup
            # Jika c_id adalah satu-satunya pilihan di domain Xj dan totalnya melebihi kapasitas,
            # maka c_id tidak konsisten untuk Xi.
            if len(self.domains[xj]) == 1 and self.domains[xj][0] == c_id:
                if (pkg_i.weight_kg + pkg_j.weight_kg) > courier.max_capacity_kg:
                    couriers_to_remove.append(c_id)
                elif (pkg_i.estimated_time_hrs + pkg_j.estimated_time_hrs) > courier.max_work_hours:
                    couriers_to_remove.append(c_id)

        for c_id in couriers_to_remove:
            self.domains[xi].remove(c_id)
            self.metrics.ac3_pruned_values += 1
            revised = True

        return revised

    # -----------------------------------------------------------------------
    # Heuristik MRV & LCV
    # -----------------------------------------------------------------------

    def select_unassigned_variable(
        self,
        assignment: Dict[str, str],
        domains: Dict[str, List[str]],
    ) -> str:
        """
        Heuristik MRV (Minimum Remaining Values):
        Memilih paket yang belum ditugaskan dengan pilihan kurir legal paling sedikit.
        Tie-breaker: Degree Heuristic (paket dengan bobot terbesar diprioritaskan).
        """
        unassigned = [v for v in self.variables if v not in assignment]

        def mrv_key(var: str) -> Tuple[int, float]:
            legal_values_count = sum(
                1 for c_id in domains[var] if self.is_consistent(var, c_id, assignment)
            )
            # Nilai balik: (sisa kurir legal, -berat paket untuk memprioritaskan paket berat)
            return (legal_values_count, -self.package_map[var].weight_kg)

        return min(unassigned, key=mrv_key)

    def order_domain_values(
        self,
        var: str,
        assignment: Dict[str, str],
        domains: Dict[str, List[str]],
    ) -> List[str]:
        """
        Heuristik LCV (Least Constraining Value) berorientasi tipe armada:
        - Paket non-bulky memprioritaskan kurir motor (Bebek/Matik) untuk menghemat ruang Mobil Box bagi paket besar.
        - Paket bulky otomatis hanya bisa memilih Mobil Box.
        - Di antara kurir dengan tipe yang sama, prioritaskan kurir dengan sisa kapasitas terbanyak.
        """
        pkg = self.package_map[var]

        def lcv_key(courier_id: str) -> Tuple[int, float]:
            c = self.courier_map[courier_id]
            used_weight = sum(
                self.package_map[p_id].weight_kg
                for p_id, assigned_c in assignment.items()
                if assigned_c == courier_id
            )
            rem_capacity = c.max_capacity_kg - used_weight

            # Prioritas 0: Kendaraan reguler untuk paket reguler (menjaga box tetap bebas)
            # Prioritas 1: Mobil Box jika terpaksa atau untuk paket bulky
            tier = 1 if (c.can_handle_bulky and not pkg.is_bulky) else 0

            return (tier, -rem_capacity)

        return sorted(domains[var], key=lcv_key)

    # -----------------------------------------------------------------------
    # Pencarian Backtracking MRV
    # -----------------------------------------------------------------------

    def solve(self, use_ac3_preprocess: bool = True) -> Optional[CSPAssignment]:
        """
        Menjalankan solver CSP lengkap:
        1. Pre-check global capacity & work hours invariant.
        2. Preprocessing propagasi AC-3.
        3. Systematic Backtracking Search dengan pemandu MRV dan LCV.

        Returns:
            CSPAssignment jika solusi ditemukan, atau None jika Infeasible.
        """
        start_time = time.perf_counter()
        self.metrics.backtrack_count = 0

        # Tahap 0: Global Invariant Pre-check (O(N) pruning)
        total_pkg_weight = sum(p.weight_kg for p in self.packages)
        total_fleet_cap = sum(c.max_capacity_kg for c in self.couriers)
        if total_pkg_weight > total_fleet_cap:
            self.metrics.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self.metrics.is_feasible = False
            return None

        total_pkg_hours = sum(p.estimated_time_hrs for p in self.packages)
        total_fleet_hours = sum(c.max_work_hours for c in self.couriers)
        if total_pkg_hours > total_fleet_hours:
            self.metrics.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self.metrics.is_feasible = False
            return None

        bulky_weight = sum(p.weight_kg for p in self.packages if p.is_bulky)
        box_capacity = sum(c.max_capacity_kg for c in self.couriers if c.can_handle_bulky)
        if bulky_weight > box_capacity:
            self.metrics.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self.metrics.is_feasible = False
            return None

        # Tahap 1: Preprocessing AC-3
        if use_ac3_preprocess:
            if not self.ac3():
                self.metrics.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
                self.metrics.is_feasible = False
                return None

        # Tahap 2: Backtracking Search
        initial_domains = {v: list(self.domains[v]) for v in self.variables}
        solution = self._backtrack({}, initial_domains)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.metrics.execution_time_ms = elapsed_ms

        if solution is None:
            self.metrics.is_feasible = False
            return None

        self.metrics.is_feasible = True
        self.metrics.assigned_count = len(solution)

        # Hitung statistik beban & jam kurir
        loads: Dict[str, float] = {c.courier_id: 0.0 for c in self.couriers}
        times: Dict[str, float] = {c.courier_id: 0.0 for c in self.couriers}
        for p_id, c_id in solution.items():
            p = self.package_map[p_id]
            loads[c_id] += p.weight_kg
            times[c_id] += p.estimated_time_hrs

        return CSPAssignment(
            assignment=solution,
            courier_loads=loads,
            courier_times=times,
            metrics=self.metrics,
        )

    def _backtrack(
        self,
        assignment: Dict[str, str],
        domains: Dict[str, List[str]],
    ) -> Optional[Dict[str, str]]:
        """Fungsi rekursif backtracking dengan heuristik MRV."""
        # Basis terminasi: seluruh variabel telah teralokasi
        if len(assignment) == len(self.variables):
            return assignment

        var = self.select_unassigned_variable(assignment, domains)

        for value in self.order_domain_values(var, assignment, domains):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value

                result = self._backtrack(assignment, domains)
                if result is not None:
                    return result

                # Backtrack jika cabang gagal
                del assignment[var]
                self.metrics.backtrack_count += 1

        return None


# ---------------------------------------------------------------------------
# Helper Generator Skenario Dataset Baku (Untuk Pengujian & Sensitivitas)
# ---------------------------------------------------------------------------

def create_default_courier_fleet(fleet_size: int = 3) -> List[Courier]:
    """
    Mengembalikan armada kurir standar sesuai domain bisnis proyek.
    
    Args:
        fleet_size: 
            - 2: 2 kurir (1 Motor Bebek, 1 Motor Matik) -> untuk demo / skala kecil (5 paket)
            - 3: 3 kurir (1 Motor Bebek, 1 Motor Matik, 1 Mobil Box) -> untuk skala sedang (15 paket)
            - 5: 5 kurir (1 Motor Bebek, 2 Motor Matik, 2 Mobil Box) -> untuk skala besar (30 & 50 paket)
    """
    base_fleet = [
        Courier(courier_id="K1_BEBEK", name="Budi (Motor Bebek)", vehicle_type="Motor Bebek", max_capacity_kg=15.0),
        Courier(courier_id="K2_MATIK", name="Siti (Motor Matik)", vehicle_type="Motor Matik", max_capacity_kg=20.0),
        Courier(courier_id="K3_BOX1",  name="Joko (Mobil Box 1)",  vehicle_type="Mobil Box",   max_capacity_kg=200.0),
        Courier(courier_id="K4_MATIK", name="Dewi (Motor Matik 2)", vehicle_type="Motor Matik", max_capacity_kg=20.0),
        Courier(courier_id="K5_BOX2",  name="Hendra (Mobil Box 2)", vehicle_type="Mobil Box",   max_capacity_kg=200.0),
    ]
    return base_fleet[:fleet_size]


def generate_scenario_packages(num_packages: int = 5) -> List[Package]:
    """
    Menghasilkan dataset paket representatif dengan ukuran bervariasi
    (5, 10, 15, 20, 30, 50 paket) untuk analisis sensitivitas.
    """
    destinations = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    packages: List[Package] = []

    for i in range(1, num_packages + 1):
        pkg_id = f"P{i:02d}"
        dest = destinations[(i - 1) % len(destinations)]

        # Paket besar setiap kelipatan 7
        is_bulky = (i % 7 == 0)
        weight = 25.0 if is_bulky else round(1.5 + ((i * 1.3) % 7.5), 1)
        est_time = 0.8 if is_bulky else 0.4

        packages.append(
            Package(
                package_id=pkg_id,
                destination_node=dest,
                weight_kg=weight,
                is_bulky=is_bulky,
                estimated_time_hrs=est_time,
            )
        )

    return packages


def load_real_ecommerce_packages(limit: Optional[int] = None) -> List[Package]:
    """
    Memuat dataset riil pengiriman paket e-commerce dari data/real_ecommerce_packages.json
    (berbasis distribusi bobot dan atribut paket Kaggle & Amazon Last-Mile Benchmark).
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    json_path = root / "data" / "real_ecommerce_packages.json"

    if not json_path.exists():
        return generate_scenario_packages(limit or 5)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_pkgs = data.get("packages", [])
    if limit is not None:
        raw_pkgs = raw_pkgs[:limit]

    return [
        Package(
            package_id=p["package_id"],
            destination_node=p["destination_landmark"],
            weight_kg=float(p["weight_kg"]),
            is_bulky=bool(p.get("is_bulky", False)),
            estimated_time_hrs=float(p.get("estimated_service_time_hrs", 0.4)),
        )
        for p in raw_pkgs
    ]

