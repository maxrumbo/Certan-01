"""
data.py
-------
Dataset representatif jaringan jalan perkotaan untuk simulasi optimasi
rute kurir pengiriman paket e-commerce.

Skenario:
  Kurir dari Hub Logistik "Depo Utara" harus mengantarkan paket ke berbagai
  alamat pelanggan di area perkotaan. Jaringan jalan dimodelkan sebagai
  graf tak-berarah berbobot dengan 12 simpul persimpangan/lokasi.

Satuan:
  - Koordinat (x, y) dalam kilometer, relatif terhadap pusat kota.
  - Biaya BBM per km dalam Rupiah (Rp/km).
  - Biaya tambahan (parkir, hambatan) dalam Rupiah (Rp).

Peta Konseptual Jaringan Jalan:
  (setiap ruas adalah jalan yang bisa dilalui dua arah)

  [HUB] Depo Utara (A) ─────── Simpang Pasar (B) ──── Jl. Merdeka Tengah (C)
           │                         │                         │
      Jl. Veteran (D) ──── Bundaran Polres (E) ──── Jl. Diponegoro (F)
           │                         │                         │
      Jl. Sudirman (G) ──── Simpang Malioboro (H) ── Jl. Ahmad Yani (I)
                                     │
                          [GOAL] Perumahan Griya (J) ── Jl. Hasanuddin (K) ── Taman Kota (L)
"""

from __future__ import annotations

from certan_01.models import Graph, Node


def build_urban_road_network(
    hub_id: str = "A",
    goal_id: str = "J",
) -> Graph:
    """
    Membangun dan mengembalikan objek Graph yang merepresentasikan
    jaringan jalan perkotaan dengan 12 simpul dan 17 ruas jalan.

    Args:
        hub_id  : ID simpul yang ditandai sebagai Hub logistik kurir.
        goal_id : ID simpul yang ditandai sebagai tujuan pengiriman paket.

    Returns:
        Graph: Objek graf jaringan jalan siap digunakan untuk pencarian.
    """
    graph = Graph()

    # ------------------------------------------------------------------
    # Simpul-Simpul Jaringan Jalan
    # Format: (node_id, nama_lokasi, x_km, y_km)
    # ------------------------------------------------------------------
    node_data = [
        ("A", "Depo Logistik Utara",        0.0,  6.0),  # Hub kurir
        ("B", "Simpang Pasar Lama",          3.0,  6.0),
        ("C", "Jl. Merdeka Tengah",          6.0,  6.0),
        ("D", "Jl. Veteran Selatan",         0.0,  3.0),
        ("E", "Bundaran Polres",             3.0,  3.0),
        ("F", "Jl. Diponegoro Timur",        6.0,  3.0),
        ("G", "Jl. Sudirman Barat",          0.0,  0.0),
        ("H", "Simpang Malioboro",           3.0,  0.0),
        ("I", "Jl. Ahmad Yani Selatan",      6.0,  0.0),
        ("J", "Perumahan Griya Indah",       4.5, -2.0),  # Tujuan pengiriman
        ("K", "Jl. Hasanuddin Raya",         7.5, -1.0),
        ("L", "Taman Kota Selatan",          9.0,  1.5),
    ]

    for nid, name, x, y in node_data:
        is_hub  = (nid == hub_id)
        is_goal = (nid == goal_id)
        graph.add_node(Node(nid, name, x, y, is_hub=is_hub, is_goal=is_goal))

    # ------------------------------------------------------------------
    # Ruas-Ruas Jalan
    # Format: (from_id, to_id, distance_km, fuel_cost_per_km, extra_cost_rp)
    #
    # fuel_cost_per_km bervariasi sesuai kondisi jalan:
    #   - Jalan arteri lancar : Rp 1.500/km
    #   - Jalan padat/macet   : Rp 1.800–2.000/km (konsumsi BBM lebih tinggi)
    #   - Jalan berbayar/tol  : + extra_cost_rp
    # ------------------------------------------------------------------
    edge_data = [
        # Baris atas (y=6): A - B - C
        ("A", "B", 3.0, 1_500, 0),
        ("B", "C", 3.0, 1_600, 0),

        # Kolom kiri: A - D - G
        ("A", "D", 3.0, 1_500, 0),
        ("D", "G", 3.0, 1_600, 0),

        # Baris tengah (y=3): D - E - F
        ("D", "E", 3.0, 1_800, 0),       # Jalan padat
        ("E", "F", 3.0, 1_700, 0),

        # Kolom tengah: B - E - H
        ("B", "E", 3.0, 1_600, 0),
        ("E", "H", 3.0, 1_700, 0),

        # Kolom kanan: C - F
        ("C", "F", 3.0, 1_500, 0),

        # Baris bawah (y=0): G - H - I
        ("G", "H", 3.0, 1_700, 0),
        ("H", "I", 3.0, 1_600, 0),

        # Koneksi F - I (kolom kanan bawah)
        ("F", "I", 3.0, 1_500, 0),

        # Koneksi ke area perumahan (tujuan)
        ("H", "J", 2.5, 1_600, 1_000),   # Biaya parkir Rp 1.000
        ("I", "J", 2.7, 1_500, 0),
        ("I", "K", 2.0, 1_500, 0),

        # Ruas sisi selatan
        ("J", "K", 3.0, 1_600, 0),
        ("K", "L", 2.5, 1_800, 2_000),   # Jalan berbayar (tambahan Rp 2.000)
    ]

    for from_id, to_id, dist, fuel, extra in edge_data:
        graph.add_edge(from_id, to_id, dist, fuel, extra)

    return graph


def list_available_scenarios() -> list[dict]:
    """
    Kembalikan daftar skenario pengiriman paket yang telah didefinisikan
    untuk digunakan pada mode demonstrasi CLI.

    Returns:
        List dict dengan kunci 'label', 'hub_id', 'goal_id', 'description'.
    """
    return [
        {
            "label"      : "Skenario 1: Depo Utara → Perumahan Griya",
            "hub_id"     : "A",
            "goal_id"    : "J",
            "description": "Pengiriman dari Depo Logistik Utara ke Perumahan Griya Indah.",
        },
        {
            "label"      : "Skenario 2: Depo Utara → Taman Kota Selatan",
            "hub_id"     : "A",
            "goal_id"    : "L",
            "description": "Pengiriman dari Depo Logistik Utara ke Taman Kota Selatan.",
        },
        {
            "label"      : "Skenario 3: Depo Utara → Jl. Ahmad Yani Selatan",
            "hub_id"     : "A",
            "goal_id"    : "I",
            "description": "Pengiriman dari Depo Logistik Utara ke Jl. Ahmad Yani Selatan.",
        },
        {
            "label"      : "Skenario 4: Jl. Veteran → Perumahan Griya",
            "hub_id"     : "D",
            "goal_id"    : "J",
            "description": "Pengiriman dari Jl. Veteran ke Perumahan Griya Indah.",
        },
    ]
