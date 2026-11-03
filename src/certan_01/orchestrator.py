"""
orchestrator.py
---------------
Modul Orkestrator Sistem Logistik Cerdas CERTAN-01.

Bertindak sebagai JEMBATAN antara dua lapisan sistem:
  - Lapisan 1 (M2 -- CSP Solver)    : Alokasi paket ke kurir (siapa bawa apa?)
  - Lapisan 2 (M1 -- A* Pathfinder) : Routing BBM-optimal per kurir (lewat jalan mana?)

Desain ini memastikan setiap Milestone baru (M3, M4, M5) dapat diintegrasikan
sebagai lapisan tambahan tanpa mengubah lapisan yang sudah ada (*open-closed principle*).

Alur Pipeline Lengkap:
    [Paket + Armada]
         |
         v  Lapisan 1 (M2)
    CSPSolver.solve()
         |  output: {pkg_id -> courier_id}
         v  Lapisan 2 (M1)
    untuk setiap kurir K_j:
      untuk setiap paket P_i milik K_j:
        A* (hub_depot -> P_i.destination_node)
         |
         v
    DispatchPlan  <- output terintegrasi

Cara Penggunaan:
    from certan_01.orchestrator import dispatch_and_route
    from certan_01.solver import Package, create_default_courier_fleet

    packages = [
        Package("P1", destination_node="it_del_sitoluama", weight_kg=2.5),
        Package("P2", destination_node=(2.3340, 99.0665),  weight_kg=5.0),  # koordinat GPS
    ]
    fleet = create_default_courier_fleet(fleet_size=3)
    plan  = dispatch_and_route(packages, fleet, hub_landmark="hub_jne_laguboti")
    plan.print_summary()
"""

from __future__ import annotations

import math
import sys as _sys

from dataclasses import dataclass, field
from typing import Any

from certan_01.data import build_laguboti_road_network
from certan_01.heuristics import euclidean_heuristic
from certan_01.osm_loader import LANDMARK_LABELS, LANDMARKS, parse_coordinates
from certan_01.search import SearchResult, astar_search
from certan_01.solver import (
    CSPAssignment,
    CSPSolver,
    Courier,
    Package,
)


# ---------------------------------------------------------------------------
# Tipe Data Output
# ---------------------------------------------------------------------------

@dataclass
class CourierRoute:
    """
    Rute hasil kalkulasi A* untuk satu kurir beserta daftar paket yang dibawanya.

    Attributes:
        courier         : Objek kurir beserta spesifikasi kendaraannya.
        packages        : Daftar paket yang dialokasikan ke kurir ini (hasil CSP).
        routes          : Mapping package_id -> SearchResult A* (rute per paket).
        total_fuel_cost : Akumulasi biaya BBM seluruh rute kurir ini (Rp).
        total_distance  : Akumulasi jarak tempuh seluruh rute kurir ini (km).
        courier_load    : Total muatan aktual kurir (kg).
        courier_time    : Total estimasi jam kerja aktual kurir.
        tour_sequence   : Urutan lokasi tur pengantaran yang dikunjungi kurir.
        tour_legs       : Daftar SearchResult tiap leg dalam tur pengantaran.
        is_multi_stop   : True jika rute dirangkai sebagai tur multi-stop (TSP).
    """
    courier: Courier
    packages: list[Package]
    routes: dict[str, SearchResult]
    total_fuel_cost: float = 0.0
    total_distance: float = 0.0
    courier_load: float = 0.0
    courier_time: float = 0.0
    tour_sequence: list[str] = field(default_factory=list)
    tour_legs: list[SearchResult] = field(default_factory=list)
    is_multi_stop: bool = False


@dataclass
class DispatchPlan:
    """
    Output terintegrasi dari pipeline M2 -> M1.

    Berisi hasil alokasi CSP dan rute A* untuk seluruh armada kurir,
    serta metadata kinerja sistem.

    Dirancang sebagai struktur data terbuka untuk M3+ -- cukup tambah field
    atau wrap DispatchPlan sebagai input ke lapisan berikutnya.
    """
    courier_routes: list[CourierRoute] = field(default_factory=list)
    csp_assignment: CSPAssignment | None = None
    is_feasible: bool = True
    infeasible_reason: str = ""
    hub_landmark: Any = "hub_jne_laguboti"
    vehicle: str = "motor_matik_kurir"

    @property
    def all_packages_routed(self) -> bool:
        """True jika seluruh paket berhasil mendapat rute A*."""
        for cr in self.courier_routes:
            for pkg in cr.packages:
                res = cr.routes.get(pkg.package_id)
                if not res or not res.found:
                    return False
        return True

    @property
    def total_fleet_fuel_cost(self) -> float:
        """Total biaya BBM seluruh armada (Rp)."""
        return sum(cr.total_fuel_cost for cr in self.courier_routes)

    @property
    def total_fleet_distance(self) -> float:
        """Total jarak tempuh seluruh armada (km)."""
        return sum(cr.total_distance for cr in self.courier_routes)

    def print_summary(self, verbose: bool = True) -> None:
        """Cetak ringkasan hasil dispatch ke terminal."""
        sep = "=" * 72
        print(f"\n{sep}")
        print("  DISPATCH PLAN -- HASIL INTEGRASI CSP (M2) + A* (M1)")
        print(sep)

        if not self.is_feasible:
            print(f"  [INFEASIBLE] {self.infeasible_reason}")
            print(sep)
            return

        if self.csp_assignment:
            m = self.csp_assignment.metrics
            print(f"  CSP Solver   : {m.execution_time_ms:.2f} ms | "
                  f"Backtrack: {m.backtrack_count} | "
                  f"AC-3 Pruned: {m.ac3_pruned_values} nilai")

        print(f"  Total BBM    : Rp {self.total_fleet_fuel_cost:,.0f}  |  "
              f"Total Jarak: {self.total_fleet_distance:.2f} km")
        print(sep)

        for cr in self.courier_routes:
            if not cr.packages:
                continue
            print(f"\n  [{cr.courier.courier_id}] {cr.courier.name} "
                  f"({cr.courier.vehicle_type})")
            print(f"  {'-' * 68}")
            print(f"  Paket    : {len(cr.packages)} paket | "
                  f"Muatan: {cr.courier_load:.1f}/{cr.courier.max_capacity_kg:.0f} kg | "
                  f"Jam: {cr.courier_time:.1f}/{cr.courier.max_work_hours:.1f} jam")
            print(f"  BBM      : Rp {cr.total_fuel_cost:,.0f} | "
                  f"Jarak: {cr.total_distance:.2f} km")

            if verbose:
                if cr.is_multi_stop and len(cr.packages) > 1 and cr.tour_sequence:
                    tour_str = " -> ".join(cr.tour_sequence)
                    print(f"  Tur Rute : {tour_str}")
                for pkg in cr.packages:
                    res = cr.routes.get(pkg.package_id)
                    dest_label = _format_destination(pkg.destination_node)
                    if res and res.found:
                        path_str = " -> ".join(res.path_names[:3])
                        if len(res.path_names) > 3:
                            path_str += " -> ..."
                        print(f"    * {pkg.package_id} -> {dest_label}")
                        print(f"        Rute  : {path_str}")
                        print(f"        BBM   : Rp {res.total_cost:,.0f} | "
                              f"Jarak: {res.total_distance:.2f} km")
                    else:
                        print(f"    * {pkg.package_id} -> {dest_label}  "
                              f"[RUTE TIDAK DITEMUKAN]")

        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Helper Internal
# ---------------------------------------------------------------------------

def _format_destination(destination_node: "str | tuple[float, float]") -> str:
    """Format destination_node ke label ramah pengguna."""
    if isinstance(destination_node, (tuple, list)):
        return f"GPS ({destination_node[0]:.4f}, {destination_node[1]:.4f})"
    if destination_node in LANDMARK_LABELS:
        return LANDMARK_LABELS[destination_node]
    return str(destination_node)


def _map_vehicle_profile(vehicle_type: str) -> str:
    """
    Petakan tipe kendaraan Courier (domain M2) ke profil kendaraan osm_loader (domain M1).

    Ini adalah titik integrasi kritis antara dua domain yang sebelumnya terpisah.
    Courier.vehicle_type (teks bebas) -> key di vehicle_fuel_model.json.
    """
    vt = vehicle_type.lower()
    if "bebek" in vt:
        return "motor_bebek_kurir"
    if "matik" in vt or "matic" in vt:
        return "motor_matik_kurir"
    if "sport" in vt or "kopling" in vt or "premium" in vt:
        return "motor_premium_kurir"
    if "diesel" in vt or "solar" in vt or "l300" in vt:
        return "mobil_box_diesel"
    if "box" in vt or "mobil" in vt or "pick" in vt:
        return "mobil_box_kurir"
    return "motor_matik_kurir"


# ---------------------------------------------------------------------------
# Resolusi Destination (helper publik untuk M3+)
# ---------------------------------------------------------------------------

def resolve_package_destination(
    destination_node: "str | tuple[float, float]",
) -> tuple[float, float]:
    """
    Resolve destination_node Package ke koordinat (lat, lon).

    Mendukung:
    - tuple (lat, lon)   -> langsung dikembalikan
    - str (landmark key) -> lookup LANDMARKS dict
    - str koordinat      -> di-parse ("2.3832, 99.1486")

    Raises:
        ValueError: Jika destination tidak dapat di-resolve ke koordinat valid.
    """
    if isinstance(destination_node, (tuple, list)) and len(destination_node) == 2:
        return float(destination_node[0]), float(destination_node[1])

    if isinstance(destination_node, str):
        if destination_node in LANDMARKS:
            return LANDMARKS[destination_node]
        parsed = parse_coordinates(destination_node)
        if parsed:
            return parsed

    raise ValueError(
        f"destination_node '{destination_node}' tidak dapat di-resolve. "
        f"Gunakan landmark key yang valid atau format koordinat '(lat, lon)'."
    )


# ---------------------------------------------------------------------------
# Algoritma Optimasi Urutan Pengantaran Paket (TSP: Nearest Neighbor + 2-Opt)
# ---------------------------------------------------------------------------

def _haversine_coords_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Hitung jarak Great-Circle antara dua pasangan koordinat (lat, lon) dalam km."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return 6371.0088 * c


def optimize_delivery_sequence(
    hub_coords: tuple[float, float],
    packages: list[Package],
) -> list[Package]:
    """
    Optimasi urutan pengantaran paket (Traveling Salesperson Problem / TSP)
    menggunakan algoritma Nearest-Neighbor yang diperhalus dengan 2-Opt local search.

    Menghasilkan urutan kunjungan yang meminimalkan jarak jelajah kurir.
    """
    if len(packages) <= 1:
        return list(packages)

    # 1. Resolusi koordinat semua paket
    pkg_coords: dict[str, tuple[float, float]] = {}
    for p in packages:
        try:
            pkg_coords[p.package_id] = resolve_package_destination(p.destination_node)
        except Exception:
            pkg_coords[p.package_id] = (2.3667, 99.1247)

    # 2. Greedy Nearest Neighbor Tour Construction
    unvisited = list(packages)
    ordered: list[Package] = []
    current_pt = hub_coords

    while unvisited:
        best_idx = 0
        best_dist = float("inf")
        for i, p in enumerate(unvisited):
            d = _haversine_coords_km(current_pt, pkg_coords[p.package_id])
            if d < best_dist:
                best_dist = d
                best_idx = i

        chosen = unvisited.pop(best_idx)
        ordered.append(chosen)
        current_pt = pkg_coords[chosen.package_id]

    # 3. 2-Opt Local Search Improvement (untuk N <= 30)
    if 2 < len(ordered) <= 30:
        def tour_dist(seq: list[Package]) -> float:
            total = _haversine_coords_km(hub_coords, pkg_coords[seq[0].package_id])
            for idx in range(len(seq) - 1):
                total += _haversine_coords_km(pkg_coords[seq[idx].package_id], pkg_coords[seq[idx + 1].package_id])
            return total

        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            best_d = tour_dist(ordered)
            for i in range(len(ordered) - 1):
                for j in range(i + 1, len(ordered)):
                    new_seq = ordered[:i] + ordered[i:j + 1][::-1] + ordered[j + 1:]
                    d_new = tour_dist(new_seq)
                    if d_new < best_d - 1e-4:
                        ordered = new_seq
                        best_d = d_new
                        improved = True
                        break
                if improved:
                    break

    return ordered


# ---------------------------------------------------------------------------
# Fungsi Utama Orchestrator
# ---------------------------------------------------------------------------

def dispatch_and_route(
    packages: list[Package],
    couriers: list[Courier],
    hub_landmark: "str | tuple[float, float]" = "hub_jne_laguboti",
    vehicle: str = "motor_matik_kurir",
    verbose: bool = True,
    multi_stop: bool = True,
    return_to_hub: bool = False,
) -> DispatchPlan:
    """
    Pipeline terintegrasi M2 -> M1: alokasi CSP lalu routing A* per kurir.

    Lapisan 1 (M2): CSPSolver mengalokasikan paket ke kurir berdasarkan
    kapasitas, kualifikasi kendaraan, dan batas jam kerja (hard constraints).

    Lapisan 2 (M1): Untuk setiap kurir, A* menghitung rute BBM-optimal.
    Jika kurir membawa > 1 paket dan multi_stop=True, urutan kunjungan dioptimasi
    via TSP (Nearest Neighbor + 2-Opt) dan dirangkai leg-demi-leg.

    Titik Ekstensi untuk M3+:
    - DispatchPlan bisa dijadikan input lapisan berikutnya (mis. penjadwalan waktu).
    - Tambahkan parameter baru ke fungsi ini tanpa mengubah M1/M2 yang sudah ada.

    Args:
        packages      : Daftar paket yang harus diantar hari ini.
        couriers      : Daftar armada kurir yang tersedia.
        hub_landmark  : Titik asal kurir -- landmark key atau koordinat GPS.
        vehicle       : Profil kendaraan fallback jika tidak terpetakan dari kurir.
        verbose       : Jika True, cetak log progres selama eksekusi.
        multi_stop    : Jika True, susun tur pengantaran terintegrasi (TSP) per kurir.
        return_to_hub : Jika True, tambahkan leg kembali ke Hub depot di akhir tur.

    Returns:
        DispatchPlan: Hasil terintegrasi. Jika CSP infeasible, is_feasible=False.
    """
    # Pastikan output UTF-8 di terminal Windows
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8")

    plan = DispatchPlan(hub_landmark=hub_landmark, vehicle=vehicle)

    # ------------------------------------------------------------------
    # Lapisan 1 -- M2: CSP Solver (Alokasi Paket ke Kurir)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[M2-CSP] Memulai alokasi {len(packages)} paket "
              f"ke {len(couriers)} kurir...")

    solver  = CSPSolver(packages, couriers)
    csp_out = solver.solve()
    plan.csp_assignment = csp_out

    if csp_out is None:
        plan.is_feasible      = False
        plan.infeasible_reason = (
            "CSP tidak menemukan alokasi legal: total muatan, kapasitas kendaraan, "
            "atau jam kerja tidak mencukupi untuk semua paket."
        )
        if verbose:
            print(f"[M2-CSP] INFEASIBLE -- {plan.infeasible_reason}")
        return plan

    if verbose:
        m = csp_out.metrics
        print(f"[M2-CSP] Selesai: {m.execution_time_ms:.2f} ms | "
              f"Backtrack: {m.backtrack_count} | "
              f"AC-3 pruned: {m.ac3_pruned_values} nilai")

    # Kelompokkan paket per kurir dari hasil assignment CSP
    pkg_map: dict[str, Package]  = {p.package_id: p for p in packages}
    grouped: dict[str, list[Package]] = {c.courier_id: [] for c in couriers}
    for pkg_id, courier_id in csp_out.assignment.items():
        grouped[courier_id].append(pkg_map[pkg_id])

    # ------------------------------------------------------------------
    # Lapisan 2 -- M1: A* Routing per Kurir
    # ------------------------------------------------------------------
    if verbose:
        active_couriers = sum(1 for c in couriers if grouped.get(c.courier_id))
        print(f"\n[M1-A*] Memulai routing A* untuk {active_couriers} kurir aktif...")

    # Resolusi koordinat hub
    try:
        hub_coords = resolve_package_destination(hub_landmark)
    except Exception:
        hub_coords = (2.3667, 99.1247)

    for courier in couriers:
        assigned_pkgs = grouped.get(courier.courier_id, [])
        cr = CourierRoute(
            courier=courier,
            packages=assigned_pkgs,
            routes={},
            courier_load=sum(p.weight_kg for p in assigned_pkgs),
            courier_time=sum(p.estimated_time_hrs for p in assigned_pkgs),
        )

        if not assigned_pkgs:
            plan.courier_routes.append(cr)
            continue

        veh_profile = _map_vehicle_profile(courier.vehicle_type)

        if verbose:
            print(f"\n  [{courier.courier_id}] {courier.name} "
                  f"-- {len(assigned_pkgs)} paket ({veh_profile})")

        # Kasus 1: Multi-Stop TSP Routing (Paket > 1 dan multi_stop aktif)
        if multi_stop and len(assigned_pkgs) > 1:
            ordered_pkgs = optimize_delivery_sequence(hub_coords, assigned_pkgs)
            cr.packages = ordered_pkgs
            cr.is_multi_stop = True
            current_origin: Any = hub_landmark
            cr.tour_sequence = [_format_destination(hub_landmark)]

            for pkg in ordered_pkgs:
                dest_label = _format_destination(pkg.destination_node)
                if verbose:
                    orig_label = _format_destination(current_origin)
                    print(f"    [A* Leg] {orig_label} -> {dest_label} ({pkg.package_id})...", end=" ")

                try:
                    graph = build_laguboti_road_network(
                        hub_landmark=current_origin,
                        goal_landmark=pkg.destination_node,
                        vehicle=veh_profile,
                    )
                    hub_node  = graph.hub_node()
                    goal_node = graph.goal_node()

                    if hub_node is None or goal_node is None:
                        if verbose:
                            print("[WARN: simpul hub/goal tidak ditemukan]")
                        continue

                    min_cost = graph.min_cost_per_km
                    def heuristic(node, g=goal_node):
                        return euclidean_heuristic(node, g, c_min_per_km=min_cost)

                    result = astar_search(
                        graph, hub_node.node_id, goal_node.node_id, heuristic
                    )
                    cr.routes[pkg.package_id] = result
                    cr.tour_legs.append(result)

                    if result.found:
                        cr.total_fuel_cost += result.total_cost
                        cr.total_distance  += result.total_distance
                        if verbose:
                            print(f"Rp {result.total_cost:,.0f} | "
                                  f"{result.total_distance:.2f} km")
                    else:
                        if verbose:
                            print("[WARN: tidak ada rute]")

                    current_origin = pkg.destination_node
                    cr.tour_sequence.append(dest_label)

                except Exception as exc:
                    if verbose:
                        print(f"[ERROR: {exc}]")

            if return_to_hub:
                try:
                    graph_ret = build_laguboti_road_network(
                        hub_landmark=current_origin,
                        goal_landmark=hub_landmark,
                        vehicle=veh_profile,
                    )
                    hn = graph_ret.hub_node()
                    gn = graph_ret.goal_node()
                    if hn and gn:
                        min_c = graph_ret.min_cost_per_km
                        ret_res = astar_search(
                            graph_ret, hn.node_id, gn.node_id,
                            lambda n, g=gn: euclidean_heuristic(n, g, c_min_per_km=min_c)
                        )
                        if ret_res.found:
                            cr.tour_legs.append(ret_res)
                            cr.total_fuel_cost += ret_res.total_cost
                            cr.total_distance += ret_res.total_distance
                            cr.tour_sequence.append(_format_destination(hub_landmark))
                except Exception:
                    pass

        # Kasus 2: Single-Stop atau mode direct
        else:
            cr.tour_sequence = [_format_destination(hub_landmark)]
            for pkg in assigned_pkgs:
                dest_label = _format_destination(pkg.destination_node)
                if verbose:
                    print(f"    [A*] {pkg.package_id} -> {dest_label}...", end=" ")

                try:
                    graph = build_laguboti_road_network(
                        hub_landmark=hub_landmark,
                        goal_landmark=pkg.destination_node,
                        vehicle=veh_profile,
                    )
                    hub_node  = graph.hub_node()
                    goal_node = graph.goal_node()

                    if hub_node is None or goal_node is None:
                        if verbose:
                            print("[WARN: simpul hub/goal tidak ditemukan]")
                        continue

                    min_cost = graph.min_cost_per_km
                    def heuristic(node, g=goal_node):
                        return euclidean_heuristic(node, g, c_min_per_km=min_cost)

                    result = astar_search(
                        graph, hub_node.node_id, goal_node.node_id, heuristic
                    )
                    cr.routes[pkg.package_id] = result
                    cr.tour_legs.append(result)

                    if result.found:
                        cr.total_fuel_cost += result.total_cost
                        cr.total_distance  += result.total_distance
                        if verbose:
                            print(f"Rp {result.total_cost:,.0f} | "
                                  f"{result.total_distance:.2f} km")
                    else:
                        if verbose:
                            print("[WARN: tidak ada rute]")

                    cr.tour_sequence.append(dest_label)

                except Exception as exc:
                    if verbose:
                        print(f"[ERROR: {exc}]")

        plan.courier_routes.append(cr)

    if verbose:
        print(f"\n[Orkestrator] Selesai. "
              f"Total BBM: Rp {plan.total_fleet_fuel_cost:,.0f} | "
              f"Jarak: {plan.total_fleet_distance:.2f} km")

    return plan
