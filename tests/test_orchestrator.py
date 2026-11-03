"""
tests/test_orchestrator.py
--------------------------
Unit test untuk modul orchestrator.py (integrasi M2 CSP Solver -> M1 A* Routing).
"""

from __future__ import annotations

import pytest

from certan_01.orchestrator import (
    CourierRoute,
    DispatchPlan,
    _format_destination,
    _haversine_coords_km,
    _map_vehicle_profile,
    dispatch_and_route,
    optimize_delivery_sequence,
    resolve_package_destination,
)
from certan_01.solver import Courier, Package, create_default_courier_fleet


# ---------------------------------------------------------------------------
# Test Resolusi Tujuan & Profil Kendaraan
# ---------------------------------------------------------------------------

def test_resolve_package_destination_tuple():
    coords = (2.3832, 99.1486)
    res = resolve_package_destination(coords)
    assert res == coords


def test_resolve_package_destination_landmark():
    res = resolve_package_destination("it_del_sitoluama")
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert abs(res[0] - 2.3832) < 0.01


def test_resolve_package_destination_string_coords():
    res = resolve_package_destination("2.3832, 99.1486")
    assert res == (2.3832, 99.1486)


def test_resolve_package_destination_invalid():
    with pytest.raises(ValueError, match="tidak dapat di-resolve"):
        resolve_package_destination("tempat_tidak_dikenal_xyz")


def test_map_vehicle_profile():
    assert _map_vehicle_profile("Motor Bebek") == "motor_bebek_kurir"
    assert _map_vehicle_profile("Motor Matik Kurir") == "motor_matik_kurir"
    assert _map_vehicle_profile("Motor Matic") == "motor_matik_kurir"
    assert _map_vehicle_profile("Mobil Box Kurir") == "mobil_box_kurir"
    assert _map_vehicle_profile("Mobil Box Diesel L300") == "mobil_box_diesel"
    assert _map_vehicle_profile("Kendaraan Umum") == "motor_matik_kurir"


def test_format_destination():
    assert "GPS" in _format_destination((2.3340, 99.0665))
    assert "Institut Teknologi Del" in _format_destination("it_del_sitoluama")
    assert _format_destination("custom_dest") == "custom_dest"


# ---------------------------------------------------------------------------
# Test Pipeline dispatch_and_route
# ---------------------------------------------------------------------------

def test_dispatch_and_route_success():
    packages = [
        Package("P01", destination_node="it_del_sitoluama", weight_kg=2.0, estimated_time_hrs=0.3),
        Package("P02", destination_node="pasar_laguboti", weight_kg=3.0, estimated_time_hrs=0.3),
    ]
    fleet = [
        Courier("K1", name="Budi", vehicle_type="Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
        Courier("K2", name="Siti", vehicle_type="Motor Matik", max_capacity_kg=20.0, max_work_hours=8.0),
    ]

    plan: DispatchPlan = dispatch_and_route(
        packages=packages,
        couriers=fleet,
        hub_landmark="hub_jne_laguboti",
        verbose=False,
    )

    assert plan.is_feasible is True
    assert plan.csp_assignment is not None
    assert plan.all_packages_routed is True
    assert plan.total_fleet_fuel_cost > 0
    assert plan.total_fleet_distance > 0
    assert len(plan.courier_routes) == 2


def test_dispatch_and_route_with_gps_coords():
    packages = [
        Package("P_GPS", destination_node=(2.3340, 99.0665), weight_kg=1.5, estimated_time_hrs=0.4),
    ]
    fleet = [
        Courier("K1", name="Budi", vehicle_type="Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
    ]

    plan = dispatch_and_route(packages, fleet, hub_landmark="hub_jne_laguboti", verbose=False)
    assert plan.is_feasible is True
    assert plan.all_packages_routed is True

    route = plan.courier_routes[0].routes["P_GPS"]
    assert route.found is True
    assert route.total_cost > 0


def test_dispatch_and_route_bulky_package():
    # Paket bulky hanya boleh dibawa mobil box (bukan motor)
    packages = [
        Package("P_BULKY", destination_node="rs_hkbp_balige", weight_kg=25.0, is_bulky=True, estimated_time_hrs=0.5),
        Package("P_KECIL", destination_node="it_del_sitoluama", weight_kg=2.0, is_bulky=False, estimated_time_hrs=0.3),
    ]
    fleet = [
        Courier("K1_MOTOR", name="Budi", vehicle_type="Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
        Courier("K2_MOBIL", name="Joko", vehicle_type="Mobil Box Kurir", max_capacity_kg=200.0, max_work_hours=8.0),
    ]

    plan = dispatch_and_route(packages, fleet, verbose=False)
    assert plan.is_feasible is True

    # Temukan rute untuk kurir mobil box
    mobil_route = next(cr for cr in plan.courier_routes if cr.courier.courier_id == "K2_MOBIL")
    pkg_ids = [p.package_id for p in mobil_route.packages]
    assert "P_BULKY" in pkg_ids


def test_dispatch_and_route_infeasible_fleet():
    # Paket melebihi kapasitas seluruh armada
    packages = [
        Package("P_OVER", destination_node="it_del_sitoluama", weight_kg=500.0, estimated_time_hrs=0.5),
    ]
    fleet = [
        Courier("K1", name="Budi", vehicle_type="Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
    ]

    plan = dispatch_and_route(packages, fleet, verbose=False)
    assert plan.is_feasible is False
    assert len(plan.infeasible_reason) > 0


def test_dispatch_plan_print_summary(capsys):
    packages = [
        Package("P01", destination_node="it_del_sitoluama", weight_kg=2.0, estimated_time_hrs=0.3),
    ]
    fleet = [
        Courier("K1", name="Budi", vehicle_type="Motor Bebek", max_capacity_kg=15.0, max_work_hours=8.0),
    ]

    plan = dispatch_and_route(packages, fleet, verbose=False)
    plan.print_summary(verbose=True)

    captured = capsys.readouterr().out
    assert "DISPATCH PLAN" in captured
    assert "Budi" in captured
    assert "P01" in captured


def test_optimize_delivery_sequence():
    hub = (2.3667, 99.1247)  # JNE Laguboti
    pkgs = [
        Package("P_FAR", destination_node="rs_hkbp_balige", weight_kg=2.0),       # Balige (jauh)
        Package("P_NEAR", destination_node="pasar_laguboti", weight_kg=1.0),      # Pasar Laguboti (sangat dekat)
        Package("P_MID", destination_node="it_del_sitoluama", weight_kg=2.5),    # Sitoluama (menengah)
    ]

    ordered = optimize_delivery_sequence(hub, pkgs)
    assert len(ordered) == 3
    # 2-Opt menghasilkan tur optimal koridor searah (IT Del -> Pasar -> Balige atau sebaliknya)
    # untuk menghindari backtracking bolak-balik antara ujung utara dan selatan
    order_ids = [p.package_id for p in ordered]
    assert order_ids in (["P_MID", "P_NEAR", "P_FAR"], ["P_FAR", "P_NEAR", "P_MID"])


def test_dispatch_and_route_multi_stop():
    # Kurir tunggal membawa 3 paket di koridor Laguboti
    pkgs = [
        Package("P1", destination_node="it_del_sitoluama", weight_kg=2.0, estimated_time_hrs=0.3),
        Package("P2", destination_node="pasar_laguboti", weight_kg=3.0, estimated_time_hrs=0.3),
        Package("P3", destination_node="sma_unggul_del", weight_kg=1.5, estimated_time_hrs=0.3),
    ]
    fleet = [
        Courier("K_MULTI", name="Tigor", vehicle_type="Motor Bebek", max_capacity_kg=30.0, max_work_hours=8.0),
    ]

    plan = dispatch_and_route(
        packages=pkgs,
        couriers=fleet,
        hub_landmark="hub_jne_laguboti",
        verbose=False,
        multi_stop=True,
    )

    assert plan.is_feasible is True
    assert plan.all_packages_routed is True
    cr = plan.courier_routes[0]
    assert cr.is_multi_stop is True
    assert len(cr.tour_sequence) >= 4  # Hub -> Stop1 -> Stop2 -> Stop3
    assert len(cr.tour_legs) == 3
    assert cr.total_fuel_cost > 0
    assert cr.total_distance > 0
    for p in pkgs:
        assert p.package_id in cr.routes
        assert cr.routes[p.package_id].found is True
