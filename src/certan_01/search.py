"""
search.py
---------
Implementasi algoritma pencarian berbasis graf untuk optimasi rute kurir
pengiriman paket perkotaan.

Algoritma yang tersedia:
  1. Uniform Cost Search (UCS)
     - Menelusuri graf dengan memprioritaskan simpul berdasarkan g(n):
       total biaya akumulatif riil dari simpul awal ke simpul saat ini.
     - Menjamin jalur dengan total biaya minimum (optimal).
     - Kompleksitas: O(b^(1+C*/ε)) di mana C* = biaya solusi optimal,
       ε = biaya langkah minimum, b = faktor percabangan.

  2. A* Search
     - Menelusuri graf dengan memprioritaskan berdasarkan f(n) = g(n) + h(n):
       kombinasi biaya riil akumulatif dan estimasi biaya ke tujuan.
     - Dengan heuristik admissible & consistent, A* menjamin solusi optimal
       dan lebih efisien (mengeksplorasi lebih sedikit simpul) daripada UCS.

Kedua algoritma menggunakan GRAPH-SEARCH (explored set) untuk menghindari
eksplorasi simpul yang sama lebih dari satu kali.

Referensi:
  Russell, S. J. & Norvig, P. (2022). *Artificial Intelligence: A Modern
  Approach* (4th ed.), Chapter 3.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Callable, Optional

from certan_01.models import Graph, Node, SearchNode


# ---------------------------------------------------------------------------
# SearchResult: Wadah hasil penelusuran
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    Menyimpan seluruh informasi hasil penelusuran algoritma pencarian.

    Attributes:
        algorithm       : Nama algoritma yang digunakan ("UCS" atau "A*").
        found           : True jika jalur ke tujuan berhasil ditemukan.
        path_ids        : Urutan node_id dari simpul awal ke simpul tujuan.
        path_names      : Urutan nama deskriptif lokasi (untuk ditampilkan).
        total_cost      : Total biaya operasional jalur yang ditemukan (Rp).
        total_distance  : Total jarak tempuh jalur yang ditemukan (km).
        nodes_expanded  : Jumlah simpul yang dieksplorasi selama penelusuran.
        runtime_ms      : Waktu eksekusi pencarian dalam milidetik.
    """

    algorithm: str
    found: bool
    path_ids: list[str]
    path_names: list[str]
    total_cost: float
    total_distance: float
    nodes_expanded: int
    runtime_ms: float

    def summary(self) -> str:
        """Kembalikan ringkasan hasil dalam bentuk string terformat."""
        if not self.found:
            return f"[{self.algorithm}] ❌ Jalur tidak ditemukan."
        path_str = " → ".join(self.path_names)
        return (
            f"\n{'='*60}\n"
            f"  Algoritma      : {self.algorithm}\n"
            f"  Status         : ✅ Jalur Ditemukan\n"
            f"  Rute           : {path_str}\n"
            f"  Total Jarak    : {self.total_distance:.2f} km\n"
            f"  Total Biaya    : Rp {self.total_cost:,.0f}\n"
            f"  Simpul Dijelajah: {self.nodes_expanded}\n"
            f"  Waktu Eksekusi : {self.runtime_ms:.4f} ms\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# Helper internal: rekonstruksi jarak total dari jalur yang ditemukan
# ---------------------------------------------------------------------------

def _calculate_path_distance(graph: Graph, path_ids: list[str]) -> float:
    """Hitung total jarak tempuh (km) berdasarkan urutan simpul dalam jalur."""
    total_distance = 0.0
    for i in range(len(path_ids) - 1):
        current_id = path_ids[i]
        next_id = path_ids[i + 1]
        for edge in graph.get_neighbors(current_id):
            if edge.to_node.node_id == next_id:
                total_distance += edge.distance_km
                break
    return total_distance


# ---------------------------------------------------------------------------
# 1. Uniform Cost Search (UCS)
# ---------------------------------------------------------------------------

def uniform_cost_search(
    graph: Graph,
    start_id: str,
    goal_id: str,
) -> SearchResult:
    """
    Uniform Cost Search (UCS) pada graf jaringan jalan perkotaan.

    Algoritma menelusuri simpul-simpul secara bertahap, selalu memilih simpul
    dengan akumulasi biaya g(n) terkecil dari antrean prioritas.

    Implementasi menggunakan GRAPH-SEARCH dengan *explored set* untuk
    mencegah kunjungan ulang ke simpul yang sama.

    Args:
        graph    : Graf jaringan jalan perkotaan.
        start_id : ID simpul awal (Hub logistik kurir).
        goal_id  : ID simpul tujuan (alamat pelanggan).

    Returns:
        SearchResult: Hasil penelusuran lengkap.
    """
    start_time = time.perf_counter()
    nodes_expanded: int = 0

    start_node: Node = graph.get_node(start_id)
    goal_node: Node = graph.get_node(goal_id)

    # Inisialisasi antrean prioritas: (g_cost, SearchNode)
    # Prioritas = g(n) = akumulasi biaya riil
    initial = SearchNode(
        priority=0.0,
        node=start_node,
        g_cost=0.0,
        path=[start_id],
    )
    frontier: list[SearchNode] = [initial]
    heapq.heapify(frontier)

    # Explored set: simpul yang telah dieksplorasi (node_id)
    explored: set[str] = set()

    while frontier:
        current: SearchNode = heapq.heappop(frontier)

        # Jika simpul sudah pernah dieksplorasi, lewati
        if current.node.node_id in explored:
            continue

        explored.add(current.node.node_id)
        nodes_expanded += 1

        # Goal test
        if current.node.node_id == goal_id:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            path_names = [graph.get_node(nid).name for nid in current.path]
            total_dist = _calculate_path_distance(graph, current.path)
            return SearchResult(
                algorithm="Uniform Cost Search (UCS)",
                found=True,
                path_ids=current.path,
                path_names=path_names,
                total_cost=current.g_cost,
                total_distance=total_dist,
                nodes_expanded=nodes_expanded,
                runtime_ms=elapsed_ms,
            )

        # Ekspansi: tambahkan semua tetangga yang belum dijelajahi ke frontier
        for edge in graph.get_neighbors(current.node.node_id):
            neighbor_id = edge.to_node.node_id
            if neighbor_id not in explored:
                new_g = current.g_cost + edge.total_cost
                new_path = current.path + [neighbor_id]
                child = SearchNode(
                    priority=new_g,       # UCS: prioritas = g(n) saja
                    node=edge.to_node,
                    g_cost=new_g,
                    path=new_path,
                )
                heapq.heappush(frontier, child)

    # Jika frontier habis dan tujuan tidak ditemukan
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        algorithm="Uniform Cost Search (UCS)",
        found=False,
        path_ids=[],
        path_names=[],
        total_cost=float("inf"),
        total_distance=float("inf"),
        nodes_expanded=nodes_expanded,
        runtime_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# 2. A* Search
# ---------------------------------------------------------------------------

def astar_search(
    graph: Graph,
    start_id: str,
    goal_id: str,
    heuristic_fn: Callable[[Node, Node], float],
) -> SearchResult:
    """
    A* Search pada graf jaringan jalan perkotaan.

    Algoritma menelusuri simpul-simpul dengan memprioritaskan berdasarkan
    nilai f(n) = g(n) + h(n):
      - g(n) : Biaya akumulatif riil dari simpul awal ke simpul n.
      - h(n) : Estimasi heuristik biaya dari simpul n ke tujuan.

    Dengan heuristik admissible dan consistent, A* menjamin:
      1. Solusi optimal (biaya minimum).
      2. Efisiensi lebih tinggi dari UCS (lebih sedikit simpul dieksplorasi).

    Implementasi menggunakan GRAPH-SEARCH dengan *explored set*.

    Args:
        graph         : Graf jaringan jalan perkotaan.
        start_id      : ID simpul awal (Hub logistik kurir).
        goal_id       : ID simpul tujuan (alamat pelanggan).
        heuristic_fn  : Fungsi heuristik h(n) dengan signature
                        (current: Node, goal: Node) -> float.

    Returns:
        SearchResult: Hasil penelusuran lengkap.
    """
    start_time = time.perf_counter()
    nodes_expanded: int = 0

    start_node: Node = graph.get_node(start_id)
    goal_node: Node = graph.get_node(goal_id)

    # Hitung h(start)
    h_start = heuristic_fn(start_node, goal_node)

    # Inisialisasi antrean prioritas
    # Prioritas = f(n) = g(n) + h(n)
    initial = SearchNode(
        priority=h_start,   # f(start) = 0 + h(start)
        node=start_node,
        g_cost=0.0,
        path=[start_id],
    )
    frontier: list[SearchNode] = [initial]
    heapq.heapify(frontier)

    # Explored set
    explored: set[str] = set()

    while frontier:
        current: SearchNode = heapq.heappop(frontier)

        if current.node.node_id in explored:
            continue

        explored.add(current.node.node_id)
        nodes_expanded += 1

        # Goal test
        if current.node.node_id == goal_id:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            path_names = [graph.get_node(nid).name for nid in current.path]
            total_dist = _calculate_path_distance(graph, current.path)
            return SearchResult(
                algorithm="A* Search",
                found=True,
                path_ids=current.path,
                path_names=path_names,
                total_cost=current.g_cost,
                total_distance=total_dist,
                nodes_expanded=nodes_expanded,
                runtime_ms=elapsed_ms,
            )

        # Ekspansi
        for edge in graph.get_neighbors(current.node.node_id):
            neighbor_id = edge.to_node.node_id
            if neighbor_id not in explored:
                new_g = current.g_cost + edge.total_cost
                h_val = heuristic_fn(edge.to_node, goal_node)
                new_f = new_g + h_val          # f(n) = g(n) + h(n)
                new_path = current.path + [neighbor_id]
                child = SearchNode(
                    priority=new_f,            # A*: prioritas = f(n)
                    node=edge.to_node,
                    g_cost=new_g,
                    path=new_path,
                )
                heapq.heappush(frontier, child)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        algorithm="A* Search",
        found=False,
        path_ids=[],
        path_names=[],
        total_cost=float("inf"),
        total_distance=float("inf"),
        nodes_expanded=nodes_expanded,
        runtime_ms=elapsed_ms,
    )
