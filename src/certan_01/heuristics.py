"""
heuristics.py
-------------
Fungsi heuristik h(n) untuk algoritma A* pada sistem optimasi rute kurir.

Fungsi heuristik yang digunakan adalah **Euclidean Distance Berbobot Biaya BBM Minimum**:

    h(n) = sqrt((x_n - x_goal)^2 + (y_n - y_goal)^2) * C_min_per_km

Sifat yang terjamin:
  - ADMISSIBLE  : h(n) <= h*(n)
      Jarak garis lurus tidak pernah melebihi jarak jaringan jalan riil, dan
      C_min_per_km adalah batas bawah biaya nyata per km. Kombinasi keduanya
      memastikan estimasi TIDAK PERNAH melebihi biaya riil sesungguhnya.

  - CONSISTENT  : h(n) <= C(n, a, n') + h(n')
      Berlaku karena Euclidean Distance memenuhi pertidaksamaan segitiga:
      ||p_n - p_goal|| <= ||p_n - p_n'|| + ||p_n' - p_goal||
      dan setelah dikalikan C_min_per_km (konstanta positif yang sama),
      ketaksamaan tetap berlaku.

Referensi:
  Russell, S. J. & Norvig, P. (2022). *Artificial Intelligence: A Modern
  Approach* (4th ed.), Chapter 3 — Solving Problems by Searching.
"""

from __future__ import annotations

import math

from certan_01.models import Node


# ---------------------------------------------------------------------------
# Konstanta Biaya BBM Minimum
# ---------------------------------------------------------------------------

# Batas bawah (lower bound) biaya operasional kurir per kilometer
# untuk jaringan perkotaan sintetis Milestone 1.
C_MIN_PER_KM: float = 1_500.0  # Rp/km (biaya bensin minimum estimasi M1)

# Batas bawah biaya operasional kurir riil per km pada koridor Toba (motor bebek)
C_MIN_PER_KM_TOBA: float = 182.0  # Rp/km (Honda Revo/Supra 100cc)


# ---------------------------------------------------------------------------
# Fungsi Heuristik Utama
# ---------------------------------------------------------------------------

def haversine_distance_km(node1: Node, node2: Node) -> float:
    """
    Hitung jarak ortodromik (Great-Circle / Haversine) antara dua simpul GPS dalam km.
    Terbukti matematis selalu <= jarak jaringan jalan riil pada permukaan bumi (admissible).
    """
    lat1 = math.radians(node1.y)
    lon1 = math.radians(node1.x)
    lat2 = math.radians(node2.y)
    lon2 = math.radians(node2.x)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return 6371.0088 * c


def euclidean_heuristic(
    current: Node,
    goal: Node,
    c_min_per_km: float | None = None,
) -> float:
    """
    Menghitung estimasi biaya minimum dari simpul saat ini ke simpul tujuan.

    Mendukung dua topologi:
      - Bidang datar sintetis M1: jarak kartesius * C_MIN_PER_KM (1500 Rp/km).
      - Jaringan jalan nyata OpenStreetMap (Toba): jarak Great-Circle (km) * biaya BBM dasar.

    Args:
        current      : Simpul jaringan jalan yang sedang dievaluasi.
        goal         : Simpul tujuan akhir pengiriman paket.
        c_min_per_km : Batas bawah biaya operasional per km (jika None, diatur otomatis
                       berdasarkan topologi simpul).

    Returns:
        Nilai estimasi biaya (float) dari simpul current menuju goal.
        Nilai ini TIDAK PERNAH melebihi biaya riil sebenarnya (admissible).
    """
    dist_km: float = current.euclidean_distance_to(goal)
    is_geographic = current.is_geo or goal.is_geo or (abs(current.x) > 40.0 and abs(current.y) < 40.0)

    if c_min_per_km is not None and c_min_per_km != C_MIN_PER_KM:
        effective_c_min = c_min_per_km
    elif is_geographic:
        # Koridor jalan nyata Toba: motor bebek adalah profil armada paling hemat (Rp 182/km)
        effective_c_min = C_MIN_PER_KM_TOBA
    else:
        # Jaringan sintetis M1
        effective_c_min = C_MIN_PER_KM

    return dist_km * effective_c_min


def zero_heuristic(current: Node, goal: Node, **_kwargs) -> float:
    """
    Heuristik nol — menjadikan A* setara dengan Uniform Cost Search (UCS).

    Berguna untuk verifikasi: hasil A* dengan heuristik nol harus identik
    dengan hasil UCS pada graf yang sama.

    Args:
        current : Simpul saat ini (tidak digunakan, untuk signature seragam).
        goal    : Simpul tujuan (tidak digunakan, untuk signature seragam).

    Returns:
        Selalu 0.0
    """
    return 0.0
