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

# Batas bawah (lower bound) biaya operasional kurir per kilometer.
# Diturunkan dari biaya bensin termurah yang mungkin ditemui pada setiap
# ruas jalan dalam jaringan (tanpa biaya tambahan).
# Nilai ini digunakan sebagai pembobot heuristik agar h(n) terjamin admissible.
C_MIN_PER_KM: float = 1_500.0  # Rp/km (biaya bensin minimum estimasi)


# ---------------------------------------------------------------------------
# Fungsi Heuristik Utama
# ---------------------------------------------------------------------------

def euclidean_heuristic(current: Node, goal: Node, c_min_per_km: float = C_MIN_PER_KM) -> float:
    """
    Menghitung estimasi biaya minimum dari simpul saat ini ke simpul tujuan.

    Formula:
        h(n) = sqrt( (x_n - x_goal)^2 + (y_n - y_goal)^2 ) * c_min_per_km

    Args:
        current      : Simpul jaringan jalan yang sedang dievaluasi.
        goal         : Simpul tujuan akhir pengiriman paket.
        c_min_per_km : Batas bawah biaya operasional per km (default = C_MIN_PER_KM).

    Returns:
        Nilai estimasi biaya (float) dari simpul current menuju goal.
        Nilai ini TIDAK PERNAH melebihi biaya riil sebenarnya (admissible).

    Examples:
        >>> from certan_01.models import Node
        >>> hub  = Node("H", "Hub", 0.0, 0.0)
        >>> goal = Node("G", "Tujuan", 3.0, 4.0)
        >>> euclidean_heuristic(hub, goal)
        7500.0  # sqrt(9+16) * 1500 = 5.0 * 1500 = 7500.0
    """
    euclidean_dist_km: float = current.euclidean_distance_to(goal)
    return euclidean_dist_km * c_min_per_km


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
