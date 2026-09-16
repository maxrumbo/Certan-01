"""
test_search.py
--------------
Suite pengujian unit otomatis menggunakan pytest untuk memverifikasi
kebenaran dan optimalitas algoritma UCS dan A*.

Skenario Pengujian:
  1. Optimalitas UCS & A* pada graf sederhana (hasil harus identik).
  2. Admissibility heuristik (h(n) tidak pernah melebihi biaya riil).
  3. UCS vs A* pada graf jaringan jalan perkotaan (semua skenario demo).
  4. Penanganan kasus tepi: simpul terisolasi, asal = tujuan.
  5. Konsistensi heuristik (triangle inequality).

Cara menjalankan:
  uv run pytest tests/ -v
  uv run pytest tests/ -v --tb=short
"""

from __future__ import annotations

import math

import pytest

from certan_01.data import build_urban_road_network, list_available_scenarios
from certan_01.heuristics import C_MIN_PER_KM, euclidean_heuristic, zero_heuristic
from certan_01.models import Graph, Node
from certan_01.search import astar_search, uniform_cost_search


# ---------------------------------------------------------------------------
# Fixtures: Graf Sederhana (untuk pengujian terkontrol)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> Graph:
    """
    Graf sederhana 5 simpul untuk pengujian terkontrol.

    Topologi (semua ruas dua arah):

      S ─(3km, 1500/km)─ A ─(2km, 1500/km)─ G
      │                  │
     (4km,1600/km)      (5km,2000/km)
      │                  │
      B ─(2km, 1500/km)─ C

    Biaya optimal S→G:
      Jalur S-A-G : (3*1500) + (2*1500)        = 4500 + 3000 = 7500
      Jalur S-B-C-A-G: (4*1600)+(2*1500)+(5*2000)+(2*1500)  = besar, bukan optimal
      Maka rute optimal = S → A → G dengan total_cost = 7500 Rp
    """
    g = Graph()
    nodes = [
        Node("S", "Start Hub",    0.0, 0.0, is_hub=True),
        Node("A", "Simpang A",    3.0, 0.0),
        Node("B", "Simpang B",    0.0, -4.0),
        Node("C", "Simpang C",    3.0, -4.0),
        Node("G", "Tujuan",       5.0, 0.0, is_goal=True),
    ]
    for node in nodes:
        g.add_node(node)

    g.add_edge("S", "A", 3.0, 1_500)
    g.add_edge("S", "B", 4.0, 1_600)
    g.add_edge("A", "G", 2.0, 1_500)
    g.add_edge("A", "C", 5.0, 2_000)
    g.add_edge("B", "C", 2.0, 1_500)

    return g


@pytest.fixture
def urban_graph() -> Graph:
    """Graf jaringan jalan perkotaan lengkap (Skenario 1: A → J)."""
    return build_urban_road_network(hub_id="A", goal_id="J")


# ---------------------------------------------------------------------------
# 1. Pengujian Optimalitas pada Graf Sederhana
# ---------------------------------------------------------------------------

class TestSimpleGraphOptimality:
    """Verifikasi UCS dan A* menghasilkan biaya optimal identik."""

    EXPECTED_OPTIMAL_COST = 7_500.0   # Rp (jalur S-A-G, lihat fixture)
    EXPECTED_OPTIMAL_PATH = ["S", "A", "G"]

    def test_ucs_finds_solution(self, simple_graph: Graph) -> None:
        result = uniform_cost_search(simple_graph, "S", "G")
        assert result.found, "UCS harus menemukan jalur pada graf yang terhubung."

    def test_astar_finds_solution(self, simple_graph: Graph) -> None:
        result = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert result.found, "A* harus menemukan jalur pada graf yang terhubung."

    def test_ucs_optimal_cost(self, simple_graph: Graph) -> None:
        result = uniform_cost_search(simple_graph, "S", "G")
        assert math.isclose(result.total_cost, self.EXPECTED_OPTIMAL_COST, rel_tol=1e-6), (
            f"UCS optimal cost seharusnya Rp {self.EXPECTED_OPTIMAL_COST:,.0f}, "
            f"didapat Rp {result.total_cost:,.0f}"
        )

    def test_astar_optimal_cost(self, simple_graph: Graph) -> None:
        result = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert math.isclose(result.total_cost, self.EXPECTED_OPTIMAL_COST, rel_tol=1e-6), (
            f"A* optimal cost seharusnya Rp {self.EXPECTED_OPTIMAL_COST:,.0f}, "
            f"didapat Rp {result.total_cost:,.0f}"
        )

    def test_ucs_and_astar_costs_identical(self, simple_graph: Graph) -> None:
        """Biaya solusi UCS dan A* harus selalu identik (jaminan optimalitas)."""
        ucs  = uniform_cost_search(simple_graph, "S", "G")
        star = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert math.isclose(ucs.total_cost, star.total_cost, rel_tol=1e-6), (
            f"UCS cost ({ucs.total_cost}) ≠ A* cost ({star.total_cost})"
        )

    def test_ucs_optimal_path(self, simple_graph: Graph) -> None:
        result = uniform_cost_search(simple_graph, "S", "G")
        assert result.path_ids == self.EXPECTED_OPTIMAL_PATH, (
            f"UCS path seharusnya {self.EXPECTED_OPTIMAL_PATH}, didapat {result.path_ids}"
        )

    def test_astar_optimal_path(self, simple_graph: Graph) -> None:
        result = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert result.path_ids == self.EXPECTED_OPTIMAL_PATH, (
            f"A* path seharusnya {self.EXPECTED_OPTIMAL_PATH}, didapat {result.path_ids}"
        )

    def test_astar_expands_leq_ucs_nodes(self, simple_graph: Graph) -> None:
        """A* tidak boleh mengeksplorasi lebih banyak simpul daripada UCS."""
        ucs  = uniform_cost_search(simple_graph, "S", "G")
        star = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert star.nodes_expanded <= ucs.nodes_expanded, (
            f"A* ({star.nodes_expanded}) harus ≤ UCS ({ucs.nodes_expanded}) nodes expanded."
        )


# ---------------------------------------------------------------------------
# 2. Pengujian Admissibility Heuristik
# ---------------------------------------------------------------------------

class TestHeuristicAdmissibility:
    """
    Verifikasi bahwa h(n) <= h*(n) untuk setiap simpul n.

    Cara verifikasi empiris:
      Untuk setiap simpul n, hitung h(n). Lalu jalankan UCS dari n ke goal
      untuk mendapatkan biaya riil h*(n). Pastikan h(n) <= h*(n).
    """

    def test_heuristic_admissible_all_nodes_to_goal(self, urban_graph: Graph) -> None:
        """h(n) tidak pernah melebihi biaya riil UCS dari n ke goal."""
        goal_node = urban_graph.goal_node()
        assert goal_node is not None, "Graf harus memiliki simpul tujuan."

        for node in urban_graph.get_all_nodes():
            if node.node_id == goal_node.node_id:
                continue  # h*(goal) = 0

            h_n = euclidean_heuristic(node, goal_node)
            ucs = uniform_cost_search(urban_graph, node.node_id, goal_node.node_id)

            if not ucs.found:
                # Jika tidak terhubung, lewati
                continue

            h_star = ucs.total_cost
            assert h_n <= h_star + 1e-6, (
                f"Admissibility GAGAL pada simpul {node.node_id}: "
                f"h(n)={h_n:.2f} > h*(n)={h_star:.2f}"
            )

    def test_zero_heuristic_equals_ucs(self, simple_graph: Graph) -> None:
        """
        A* dengan heuristik nol harus menghasilkan biaya identik dengan UCS.
        (Karena zero heuristic menjadikan A* setara UCS secara perilaku.)
        """
        ucs  = uniform_cost_search(simple_graph, "S", "G")
        star = astar_search(simple_graph, "S", "G", zero_heuristic)
        assert math.isclose(ucs.total_cost, star.total_cost, rel_tol=1e-6), (
            "A* dengan heuristik nol harus identik dengan UCS."
        )

    def test_heuristic_at_goal_is_zero(self, urban_graph: Graph) -> None:
        """h(goal) harus bernilai 0 (simpul tujuan tidak perlu estimasi lebih lanjut)."""
        goal = urban_graph.goal_node()
        h_val = euclidean_heuristic(goal, goal)
        assert math.isclose(h_val, 0.0, abs_tol=1e-9), (
            f"h(goal) harus 0.0, didapat {h_val}"
        )

    def test_heuristic_non_negative(self, urban_graph: Graph) -> None:
        """h(n) harus selalu bernilai >= 0."""
        goal = urban_graph.goal_node()
        for node in urban_graph.get_all_nodes():
            h_val = euclidean_heuristic(node, goal)
            assert h_val >= 0.0, (
                f"h({node.node_id}) bernilai negatif: {h_val}"
            )


# ---------------------------------------------------------------------------
# 3. Pengujian pada Semua Skenario Demo (Urban Graph)
# ---------------------------------------------------------------------------

class TestAllDemoScenarios:
    """Pastikan UCS dan A* optimal dan konsisten pada semua skenario demo."""

    @pytest.mark.parametrize(
        "scenario",
        list_available_scenarios(),
        ids=[s["label"][:40] for s in list_available_scenarios()],
    )
    def test_scenario_ucs_finds_solution(self, scenario: dict) -> None:
        graph = build_urban_road_network(scenario["hub_id"], scenario["goal_id"])
        result = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        assert result.found, f"UCS gagal: {scenario['label']}"

    @pytest.mark.parametrize(
        "scenario",
        list_available_scenarios(),
        ids=[s["label"][:40] for s in list_available_scenarios()],
    )
    def test_scenario_astar_finds_solution(self, scenario: dict) -> None:
        graph = build_urban_road_network(scenario["hub_id"], scenario["goal_id"])
        result = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"], euclidean_heuristic
        )
        assert result.found, f"A* gagal: {scenario['label']}"

    @pytest.mark.parametrize(
        "scenario",
        list_available_scenarios(),
        ids=[s["label"][:40] for s in list_available_scenarios()],
    )
    def test_scenario_ucs_astar_costs_match(self, scenario: dict) -> None:
        """Biaya UCS dan A* harus identik di semua skenario."""
        graph = build_urban_road_network(scenario["hub_id"], scenario["goal_id"])
        ucs  = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        star = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"], euclidean_heuristic
        )
        assert ucs.found and star.found
        assert math.isclose(ucs.total_cost, star.total_cost, rel_tol=1e-6), (
            f"Biaya berbeda pada {scenario['label']}: "
            f"UCS={ucs.total_cost:.0f}, A*={star.total_cost:.0f}"
        )

    @pytest.mark.parametrize(
        "scenario",
        list_available_scenarios(),
        ids=[s["label"][:40] for s in list_available_scenarios()],
    )
    def test_scenario_astar_leq_ucs_expanded(self, scenario: dict) -> None:
        """A* tidak boleh mengeksplorasi lebih banyak simpul dari UCS."""
        graph = build_urban_road_network(scenario["hub_id"], scenario["goal_id"])
        ucs  = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        star = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"], euclidean_heuristic
        )
        assert star.nodes_expanded <= ucs.nodes_expanded, (
            f"[{scenario['label']}] A* ({star.nodes_expanded}) > UCS ({ucs.nodes_expanded})"
        )


# ---------------------------------------------------------------------------
# 4. Pengujian Kasus Tepi
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verifikasi penanganan input dan kondisi tidak biasa."""

    def test_unreachable_goal(self) -> None:
        """UCS dan A* harus mengembalikan found=False jika tujuan tidak terhubung."""
        g = Graph()
        g.add_node(Node("X", "Titik X", 0.0, 0.0, is_hub=True))
        g.add_node(Node("Y", "Titik Y", 5.0, 5.0, is_goal=True))
        # Tidak ada edge → Y tidak terjangkau dari X

        ucs  = uniform_cost_search(g, "X", "Y")
        star = astar_search(g, "X", "Y", euclidean_heuristic)

        assert not ucs.found,  "UCS harus mengembalikan found=False untuk graf tak terhubung."
        assert not star.found, "A* harus mengembalikan found=False untuk graf tak terhubung."

    def test_single_node_start_equals_goal(self) -> None:
        """
        Jika start == goal, algoritma seharusnya langsung menemukan solusi
        dengan biaya 0 dan jalur hanya berisi satu simpul.
        """
        g = Graph()
        g.add_node(Node("A", "Titik A", 0.0, 0.0, is_hub=True, is_goal=True))

        ucs  = uniform_cost_search(g, "A", "A")
        star = astar_search(g, "A", "A", euclidean_heuristic)

        assert ucs.found,  "UCS: Start=Goal harus ditemukan langsung."
        assert star.found, "A*: Start=Goal harus ditemukan langsung."
        assert math.isclose(ucs.total_cost,  0.0, abs_tol=1e-9)
        assert math.isclose(star.total_cost, 0.0, abs_tol=1e-9)
        assert ucs.path_ids  == ["A"]
        assert star.path_ids == ["A"]

    def test_direct_single_edge_path(self) -> None:
        """Pada jalur satu langkah, biaya harus sama persis dengan edge cost."""
        g = Graph()
        g.add_node(Node("S", "Start", 0.0, 0.0, is_hub=True))
        g.add_node(Node("G", "Goal",  3.0, 0.0, is_goal=True))
        g.add_edge("S", "G", 3.0, 1_500, 500)  # total = 3*1500 + 500 = 5000

        ucs  = uniform_cost_search(g, "S", "G")
        star = astar_search(g, "S", "G", euclidean_heuristic)

        assert math.isclose(ucs.total_cost,  5_000.0, abs_tol=1e-6)
        assert math.isclose(star.total_cost, 5_000.0, abs_tol=1e-6)
        assert ucs.path_ids  == ["S", "G"]
        assert star.path_ids == ["S", "G"]

    def test_runtime_is_positive(self, simple_graph: Graph) -> None:
        """Runtime harus bernilai positif untuk setiap eksekusi."""
        ucs  = uniform_cost_search(simple_graph, "S", "G")
        star = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert ucs.runtime_ms  >= 0.0
        assert star.runtime_ms >= 0.0

    def test_nodes_expanded_at_least_one(self, simple_graph: Graph) -> None:
        """Minimal satu simpul harus dieksplorasi (simpul awal)."""
        ucs  = uniform_cost_search(simple_graph, "S", "G")
        star = astar_search(simple_graph, "S", "G", euclidean_heuristic)
        assert ucs.nodes_expanded  >= 1
        assert star.nodes_expanded >= 1


# ---------------------------------------------------------------------------
# 5. Pengujian Konsistensi Heuristik (Triangle Inequality)
# ---------------------------------------------------------------------------

class TestHeuristicConsistency:
    """
    Verifikasi sifat Consistency: h(n) <= C(n, a, n') + h(n')
    untuk setiap ruas jalan (n → n') dalam jaringan.
    """

    def test_consistency_all_edges(self, urban_graph: Graph) -> None:
        """Periksa konsistensi pada setiap ruas jalan dalam jaringan."""
        goal = urban_graph.goal_node()
        assert goal is not None

        violations = []
        for node in urban_graph.get_all_nodes():
            h_n = euclidean_heuristic(node, goal)
            for edge in urban_graph.get_neighbors(node.node_id):
                neighbor = edge.to_node
                h_n_prime = euclidean_heuristic(neighbor, goal)
                step_cost = edge.total_cost
                # Konsistensi: h(n) <= C(n,a,n') + h(n')
                if h_n > step_cost + h_n_prime + 1e-6:
                    violations.append(
                        f"{node.node_id}→{neighbor.node_id}: "
                        f"h({node.node_id})={h_n:.2f} > "
                        f"C={step_cost:.2f} + h({neighbor.node_id})={h_n_prime:.2f}"
                    )

        assert not violations, (
            f"Pelanggaran Konsistensi Heuristik ditemukan:\n"
            + "\n".join(violations)
        )
