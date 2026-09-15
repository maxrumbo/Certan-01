"""
models.py
---------
Representasi data struktur graf jaringan jalan perkotaan untuk simulasi
optimasi rute kurir pengiriman paket.

Kelas-kelas utama:
  - Node       : Simpul pada jaringan jalan (persimpangan, hub, atau alamat pelanggan).
  - Edge       : Ruas jalan berbobot yang menghubungkan dua simpul.
  - Graph      : Graf tak-berarah berbobot yang merepresentasikan jaringan jalan.
  - SearchNode : Bungkus simpul untuk penelusuran dengan pelacak biaya & jalur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Node: Simpul jaringan jalan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """
    Merepresentasikan satu titik pada jaringan jalan perkotaan.

    Attributes:
        node_id  : Identifikasi unik simpul (misal: "A", "B", "Hub").
        name     : Nama deskriptif lokasi (misal: "Simpang Pos", "Jl. Pemuda 12").
        x        : Koordinat sumbu-X (longitude estimasi dalam km, titik acuan = 0).
        y        : Koordinat sumbu-Y (latitude estimasi dalam km, titik acuan = 0).
        is_hub   : True jika simpul ini merupakan depot/hub logistik kurir.
        is_goal  : True jika simpul ini merupakan alamat tujuan pengiriman paket.
    """

    node_id: str
    name: str
    x: float
    y: float
    is_hub: bool = False
    is_goal: bool = False

    def euclidean_distance_to(self, other: "Node") -> float:
        """Hitung jarak garis lurus (Euclidean) ke simpul lain dalam satuan km."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self) -> str:
        tag = " [HUB]" if self.is_hub else (" [TUJUAN]" if self.is_goal else "")
        return f"Node({self.node_id}: {self.name}{tag})"


# ---------------------------------------------------------------------------
# Edge: Ruas jalan berbobot
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """
    Merepresentasikan satu ruas jalan yang menghubungkan dua simpul.

    Attributes:
        from_node       : Simpul asal ruas jalan.
        to_node         : Simpul tujuan ruas jalan.
        distance_km     : Panjang ruas jalan dalam kilometer.
        fuel_cost_per_km: Biaya bensin riil per kilometer pada ruas ini (Rp/km).
        extra_cost      : Biaya tambahan tetap pada ruas ini (misal: parkir, tol) dalam Rp.
    """

    from_node: Node
    to_node: Node
    distance_km: float
    fuel_cost_per_km: float
    extra_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Biaya riil total menelusuri ruas jalan ini (Step Cost Function C)."""
        return self.distance_km * self.fuel_cost_per_km + self.extra_cost

    def __repr__(self) -> str:
        return (
            f"Edge({self.from_node.node_id} → {self.to_node.node_id} | "
            f"{self.distance_km:.2f} km | Rp {self.total_cost:,.0f})"
        )


# ---------------------------------------------------------------------------
# Graph: Graf jaringan jalan perkotaan
# ---------------------------------------------------------------------------

class Graph:
    """
    Graf tak-berarah berbobot yang merepresentasikan jaringan jalan perkotaan.

    Setiap ruas jalan diasumsikan bisa dilalui dua arah (undirected) dengan
    biaya yang sama pada kedua arah.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        # adjacency list: node_id -> list of Edge
        self._adjacency: dict[str, list[Edge]] = {}

    # ------------------------------------------------------------------
    # Builder Methods
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Tambahkan simpul ke graf."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        distance_km: float,
        fuel_cost_per_km: float,
        extra_cost: float = 0.0,
    ) -> None:
        """
        Tambahkan ruas jalan tak-berarah antara dua simpul.

        Raises:
            KeyError: Jika salah satu node_id belum terdaftar.
        """
        from_node = self._nodes[from_id]
        to_node = self._nodes[to_id]

        edge_forward = Edge(from_node, to_node, distance_km, fuel_cost_per_km, extra_cost)
        edge_backward = Edge(to_node, from_node, distance_km, fuel_cost_per_km, extra_cost)

        self._adjacency[from_id].append(edge_forward)
        self._adjacency[to_id].append(edge_backward)

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Node:
        """Ambil simpul berdasarkan ID-nya."""
        return self._nodes[node_id]

    def get_neighbors(self, node_id: str) -> list[Edge]:
        """Kembalikan semua ruas jalan yang berawal dari simpul ini."""
        return self._adjacency.get(node_id, [])

    def get_all_nodes(self) -> list[Node]:
        """Kembalikan seluruh simpul dalam graf."""
        return list(self._nodes.values())

    def hub_node(self) -> Optional[Node]:
        """Kembalikan simpul hub logistik (titik awal kurir), jika ada."""
        for node in self._nodes.values():
            if node.is_hub:
                return node
        return None

    def goal_node(self) -> Optional[Node]:
        """Kembalikan simpul tujuan pengiriman paket, jika ada."""
        for node in self._nodes.values():
            if node.is_goal:
                return node
        return None

    def summary(self) -> str:
        """Ringkasan singkat struktur graf."""
        total_edges = sum(len(edges) for edges in self._adjacency.values()) // 2
        return (
            f"Graf Jaringan Jalan: {len(self._nodes)} simpul, {total_edges} ruas jalan"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# SearchNode: Bungkus untuk keperluan penelusuran berbasis priority queue
# ---------------------------------------------------------------------------

@dataclass(order=True)
class SearchNode:
    """
    Bungkus simpul yang digunakan selama penelusuran (UCS / A*).

    Disimpan dalam antrean prioritas (heapq) berdasarkan nilai prioritas.
    Menggunakan @dataclass(order=True) sehingga perbandingan otomatis
    menggunakan urutan field (prioritas dulu, baru sisanya diabaikan).

    Attributes:
        priority  : Nilai prioritas antrean (g(n) untuk UCS, g(n)+h(n) untuk A*).
        node      : Simpul jaringan jalan yang direpresentasikan.
        g_cost    : Biaya akumulatif riil dari simpul awal ke simpul ini (g(n)).
        path      : Daftar node_id yang menyusun jalur dari simpul awal ke simpul ini.
    """

    priority: float
    node: Node = field(compare=False)
    g_cost: float = field(compare=False)
    path: list[str] = field(default_factory=list, compare=False)
