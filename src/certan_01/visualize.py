"""
visualize.py
------------
Modul generator visualisasi grafis untuk simulasi optimasi rute kurir.

Menghasilkan tiga jenis gambar PNG di folder `reports/figures/`:

  1. road_network_topology.png
       Peta 2D seluruh jaringan jalan perkotaan: simpul (nodes) dengan
       koordinat (x, y), ruas jalan (edges) dengan label jarak & biaya,
       serta penanda khusus untuk Hub dan simpul Tujuan.

  2. route_solution_{skenario}.png
       Peta yang menyorot rute optimal hasil algoritma UCS dan A*
       (ditampilkan berdampingan) dari simpul asal ke simpul tujuan.

  3. performance_benchmark_chart.png
       Grafik batang (bar chart) perbandingan metrik kinerja UCS vs A*
       pada seluruh skenario: Simpul Dijelajahi & Runtime (ms).

Cara menjalankan (dari root proyek):
  uv run certan-01 --visualize

Atau dari Python:
  from certan_01.visualize import generate_all_figures
  generate_all_figures()

Referensi warna:
  Hub (awal)   : Biru (#2563EB)
  Goal (tujuan): Hijau (#16A34A)
  Node biasa   : Abu-abu muda (#94A3B8)
  Rute UCS     : Oranye (#F97316)
  Rute A*      : Ungu (#7C3AED)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # Backend non-interaktif agar aman di lingkungan tanpa display

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from certan_01.data import build_urban_road_network, list_available_scenarios
from certan_01.heuristics import euclidean_heuristic
from certan_01.models import Graph, Node
from certan_01.search import SearchResult, astar_search, uniform_cost_search


# ---------------------------------------------------------------------------
# Konstanta Visual
# ---------------------------------------------------------------------------

FIGURES_DIR = Path("reports") / "figures"

# Palet warna utama
COLOR_HUB       = "#2563EB"   # Biru
COLOR_GOAL      = "#16A34A"   # Hijau
COLOR_NODE      = "#CBD5E1"   # Abu-abu muda
COLOR_EDGE      = "#64748B"   # Abu-abu gelap
COLOR_UCS       = "#F97316"   # Oranye
COLOR_ASTAR     = "#7C3AED"   # Ungu
COLOR_BG        = "#0F172A"   # Dark navy (latar belakang)
COLOR_PANEL     = "#1E293B"   # Dark slate (panel)
COLOR_TEXT      = "#F1F5F9"   # Putih terang
COLOR_SUBTEXT   = "#94A3B8"   # Abu-abu terang

# Font
FONT_FAMILY = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Helpers: Gaya Plot Global
# ---------------------------------------------------------------------------

def _apply_dark_style(fig: plt.Figure, ax: plt.Axes | list[plt.Axes]) -> None:
    """Terapkan tema gelap premium pada figure dan axes."""
    fig.patch.set_facecolor(COLOR_BG)
    axes = ax if isinstance(ax, list) else [ax]
    for a in axes:
        a.set_facecolor(COLOR_PANEL)
        a.tick_params(colors=COLOR_SUBTEXT, labelsize=8)
        for spine in a.spines.values():
            spine.set_edgecolor(COLOR_EDGE)


def _draw_graph_base(
    ax: plt.Axes,
    graph: Graph,
    highlight_edges: set[tuple[str, str]] | None = None,
    highlight_color: str = COLOR_UCS,
    show_edge_labels: bool = True,
) -> dict[str, tuple[float, float]]:
    """
    Gambar jaringan jalan dasar pada axes yang diberikan.

    Args:
        ax              : Axes matplotlib target.
        graph           : Objek Graf jaringan jalan.
        highlight_edges : Set pasangan (from_id, to_id) yang akan disorot.
        highlight_color : Warna garis ruas yang disorot.
        show_edge_labels: Tampilkan label biaya/jarak pada ruas.

    Returns:
        Dict node_id -> (x, y) posisi masing-masing simpul.
    """
    nodes = graph.get_all_nodes()
    pos: dict[str, tuple[float, float]] = {n.node_id: (n.x, n.y) for n in nodes}

    highlight_edges = highlight_edges or set()
    drawn_edges: set[frozenset] = set()  # Hindari gambar ruas duplikat (undirected)

    # --- Gambar ruas jalan ---
    for node in nodes:
        for edge in graph.get_neighbors(node.node_id):
            edge_key = frozenset({node.node_id, edge.to_node.node_id})
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            x_vals = [pos[node.node_id][0], pos[edge.to_node.node_id][0]]
            y_vals = [pos[node.node_id][1], pos[edge.to_node.node_id][1]]

            is_highlighted = (
                (node.node_id, edge.to_node.node_id) in highlight_edges
                or (edge.to_node.node_id, node.node_id) in highlight_edges
            )

            if is_highlighted:
                # Ruas disorot: garis tebal berwarna
                ax.plot(x_vals, y_vals, color=highlight_color,
                        linewidth=3.5, solid_capstyle="round", zorder=2,
                        alpha=0.9)
                # Glow effect: garis lebih tebal + transparan di belakang
                ax.plot(x_vals, y_vals, color=highlight_color,
                        linewidth=7, solid_capstyle="round", zorder=1,
                        alpha=0.2)
            else:
                ax.plot(x_vals, y_vals, color=COLOR_EDGE,
                        linewidth=1.2, alpha=0.5, zorder=1)

            # Label ruas (jarak & biaya)
            if show_edge_labels and not is_highlighted:
                mid_x = (x_vals[0] + x_vals[1]) / 2
                mid_y = (y_vals[0] + y_vals[1]) / 2
                label_txt = f"{edge.distance_km:.1f}km\nRp{edge.total_cost/1000:.1f}k"
                ax.text(mid_x, mid_y, label_txt,
                        fontsize=5.5, color=COLOR_SUBTEXT, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=COLOR_PANEL,
                                  alpha=0.7, edgecolor="none"),
                        zorder=3)

    # --- Gambar simpul ---
    for node in nodes:
        x, y = pos[node.node_id]

        if node.is_hub:
            color, size, zorder = COLOR_HUB, 220, 5
        elif node.is_goal:
            color, size, zorder = COLOR_GOAL, 220, 5
        else:
            color, size, zorder = COLOR_NODE, 120, 4

        ax.scatter(x, y, s=size, color=color, zorder=zorder,
                   edgecolors=COLOR_BG, linewidths=1.5)

        # Label simpul
        label = f"{node.node_id}\n{node.name}"
        offset_y = 0.5 if y >= 0 else -0.5

        # Sesuaikan posisi label agar tidak tumpang tindih
        ha = "center"
        if node.node_id in ("A", "D", "G"):
            ha = "right"
            ax.text(x - 0.3, y + offset_y, label,
                    fontsize=6.5, color=COLOR_TEXT, ha=ha, va="bottom",
                    fontweight="bold" if (node.is_hub or node.is_goal) else "normal",
                    zorder=6)
        elif node.node_id in ("C", "F", "I", "K", "L"):
            ha = "left"
            ax.text(x + 0.3, y + offset_y, label,
                    fontsize=6.5, color=COLOR_TEXT, ha=ha, va="bottom",
                    fontweight="bold" if (node.is_hub or node.is_goal) else "normal",
                    zorder=6)
        else:
            ax.text(x, y + offset_y, label,
                    fontsize=6.5, color=COLOR_TEXT, ha="center", va="bottom",
                    fontweight="bold" if (node.is_hub or node.is_goal) else "normal",
                    zorder=6)

    return pos


def _path_to_edge_set(path_ids: list[str]) -> set[tuple[str, str]]:
    """Konversi daftar ID simpul dalam jalur ke set pasangan edge."""
    return {(path_ids[i], path_ids[i + 1]) for i in range(len(path_ids) - 1)}


# ---------------------------------------------------------------------------
# Gambar 1: Topologi Jaringan Jalan
# ---------------------------------------------------------------------------

def plot_road_network_topology(
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Gambar peta 2D seluruh jaringan jalan perkotaan beserta simpul,
    ruas jalan, label biaya, dan keterangan jenis simpul.

    Args:
        output_dir: Direktori tujuan file PNG.

    Returns:
        Path file gambar yang berhasil disimpan.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "road_network_topology.png"

    graph = build_urban_road_network(hub_id="A", goal_id="J")

    fig, ax = plt.subplots(figsize=(13, 9))
    _apply_dark_style(fig, ax)

    _draw_graph_base(ax, graph, show_edge_labels=True)

    # Judul & label sumbu
    ax.set_title(
        "Topologi Jaringan Jalan Perkotaan — Jaringan Kurir CERTAN-01",
        fontsize=13, fontweight="bold", color=COLOR_TEXT,
        pad=16, fontfamily=FONT_FAMILY,
    )
    ax.set_xlabel("Koordinat X (km)", fontsize=9, color=COLOR_SUBTEXT)
    ax.set_ylabel("Koordinat Y (km)", fontsize=9, color=COLOR_SUBTEXT)
    ax.tick_params(colors=COLOR_SUBTEXT)

    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HUB,  label="Hub Logistik (Depo Utara)"),
        mpatches.Patch(facecolor=COLOR_GOAL, label="Simpul Tujuan Pengiriman"),
        mpatches.Patch(facecolor=COLOR_NODE, label="Persimpangan / Ruas Jalan"),
        Line2D([0], [0], color=COLOR_EDGE, linewidth=1.5, label="Ruas Jalan Berbobot"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right", fontsize=8,
        facecolor=COLOR_BG, edgecolor=COLOR_EDGE,
        labelcolor=COLOR_TEXT,
    )

    ax.set_xlim(-2, 11.5)
    ax.set_ylim(-4, 8)
    ax.grid(True, color=COLOR_EDGE, alpha=0.15, linestyle="--", linewidth=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Gambar 2: Perbandingan Rute UCS vs A* (per skenario)
# ---------------------------------------------------------------------------

def plot_route_solution_comparison(
    hub_id: str = "A",
    goal_id: str = "J",
    scenario_label: str = "Skenario 1",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Gambar peta berdampingan yang menyorot rute optimal UCS (kiri)
    dan A* (kanan) dari simpul Hub ke simpul Tujuan.

    Args:
        hub_id        : ID simpul asal (Hub logistik).
        goal_id       : ID simpul tujuan (alamat pelanggan).
        scenario_label: Label skenario untuk judul gambar & nama file.
        output_dir    : Direktori tujuan file PNG.

    Returns:
        Path file gambar yang berhasil disimpan.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = scenario_label.lower().replace(" ", "_").replace(":", "").replace("→", "to")
    output_path = output_dir / f"route_solution_{safe_label}.png"

    graph = build_urban_road_network(hub_id=hub_id, goal_id=goal_id)
    ucs_result   = uniform_cost_search(graph, hub_id, goal_id)
    astar_result = astar_search(graph, hub_id, goal_id, heuristic_fn=euclidean_heuristic)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _apply_dark_style(fig, axes.tolist())

    configs = [
        (axes[0], ucs_result,   COLOR_UCS,   "Uniform Cost Search (UCS)"),
        (axes[1], astar_result, COLOR_ASTAR,  "A* Search"),
    ]

    for ax, result, color, algo_name in configs:
        edge_set = _path_to_edge_set(result.path_ids) if result.found else set()
        _draw_graph_base(ax, graph,
                         highlight_edges=edge_set,
                         highlight_color=color,
                         show_edge_labels=False)

        # Subtitle per panel
        if result.found:
            subtitle = (
                f"Rute: {' → '.join(result.path_ids)}\n"
                f"Biaya: Rp {result.total_cost:,.0f}  |  "
                f"Jarak: {result.total_distance:.2f} km  |  "
                f"Simpul Dijelajahi: {result.nodes_expanded}"
            )
        else:
            subtitle = "Jalur tidak ditemukan."

        ax.set_title(
            f"{algo_name}",
            fontsize=11, fontweight="bold", color=color, pad=10,
        )
        ax.text(0.5, -0.08, subtitle,
                transform=ax.transAxes,
                fontsize=7.5, color=COLOR_SUBTEXT, ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_BG,
                          alpha=0.8, edgecolor=COLOR_EDGE))

        ax.set_xlabel("Koordinat X (km)", fontsize=8, color=COLOR_SUBTEXT)
        ax.set_ylabel("Koordinat Y (km)", fontsize=8, color=COLOR_SUBTEXT)
        ax.set_xlim(-2, 11.5)
        ax.set_ylim(-5, 8.5)
        ax.grid(True, color=COLOR_EDGE, alpha=0.15, linestyle="--", linewidth=0.6)

        # Legenda mini
        legend_els = [
            mpatches.Patch(facecolor=COLOR_HUB,  label=f"Hub [{hub_id}]"),
            mpatches.Patch(facecolor=COLOR_GOAL, label=f"Tujuan [{goal_id}]"),
            Line2D([0], [0], color=color, linewidth=2.5, label="Rute Terpilih"),
        ]
        ax.legend(handles=legend_els, fontsize=7.5, loc="lower right",
                  facecolor=COLOR_BG, edgecolor=COLOR_EDGE, labelcolor=COLOR_TEXT)

    fig.suptitle(
        f"Perbandingan Rute Kurir — {scenario_label}  |  [{hub_id}] → [{goal_id}]",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Gambar 3: Grafik Batang Perbandingan Kinerja (Bar Chart)
# ---------------------------------------------------------------------------

def plot_performance_benchmark_chart(
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Gambar grafik batang (bar chart) perbandingan metrik kinerja UCS vs A*
    pada seluruh skenario yang tersedia.

    Metrik yang divisualisasikan:
      - Simpul Dijelajahi (nodes expanded)
      - Runtime (ms)

    Args:
        output_dir: Direktori tujuan file PNG.

    Returns:
        Path file gambar yang berhasil disimpan.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "performance_benchmark_chart.png"

    scenarios = list_available_scenarios()

    # Kumpulkan data metrik
    labels: list[str] = []
    ucs_nodes:    list[int]   = []
    astar_nodes:  list[int]   = []
    ucs_runtimes: list[float] = []
    astar_runtimes: list[float] = []

    for scenario in scenarios:
        graph = build_urban_road_network(
            hub_id=scenario["hub_id"], goal_id=scenario["goal_id"]
        )
        ucs   = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        astar = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"],
            heuristic_fn=euclidean_heuristic,
        )
        # Label skenario singkat: potong "Skenario N: " jadi "S.N"
        short = scenario["label"].split(":")[0].replace("Skenario", "S.")
        labels.append(short.strip())
        ucs_nodes.append(ucs.nodes_expanded)
        astar_nodes.append(astar.nodes_expanded)
        ucs_runtimes.append(ucs.runtime_ms)
        astar_runtimes.append(astar.runtime_ms)

    n = len(labels)
    x = list(range(n))
    bar_w = 0.35

    fig, (ax_nodes, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_style(fig, [ax_nodes, ax_time])

    def _draw_bars(
        ax: plt.Axes,
        ucs_vals: list,
        astar_vals: list,
        ylabel: str,
        title: str,
        unit: str = "",
    ) -> None:
        bars_ucs   = ax.bar([xi - bar_w/2 for xi in x], ucs_vals,   width=bar_w,
                            color=COLOR_UCS,   label="UCS",  alpha=0.9,
                            edgecolor=COLOR_BG, linewidth=0.8)
        bars_astar = ax.bar([xi + bar_w/2 for xi in x], astar_vals, width=bar_w,
                            color=COLOR_ASTAR, label="A*",   alpha=0.9,
                            edgecolor=COLOR_BG, linewidth=0.8)

        # Nilai di atas batang
        for bar in bars_ucs:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(ucs_vals + astar_vals) * 0.02,
                    f"{h:.3f}{unit}" if unit else str(int(h)),
                    ha="center", va="bottom", fontsize=8, color=COLOR_UCS, fontweight="bold")
        for bar in bars_astar:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(ucs_vals + astar_vals) * 0.02,
                    f"{h:.3f}{unit}" if unit else str(int(h)),
                    ha="center", va="bottom", fontsize=8, color=COLOR_ASTAR, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color=COLOR_TEXT)
        ax.set_ylabel(ylabel, fontsize=9, color=COLOR_SUBTEXT)
        ax.set_title(title, fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=12)
        ax.set_ylim(0, max(ucs_vals + astar_vals) * 1.45)
        ax.legend(fontsize=9, facecolor=COLOR_BG, edgecolor=COLOR_EDGE,
                  labelcolor=COLOR_TEXT)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f" if unit else "%d"))
        ax.grid(axis="y", color=COLOR_EDGE, alpha=0.2, linestyle="--", linewidth=0.6)

    _draw_bars(
        ax_nodes, ucs_nodes, astar_nodes,
        ylabel="Jumlah Simpul",
        title="Simpul Dijelajahi (Nodes Expanded)",
    )
    _draw_bars(
        ax_time, ucs_runtimes, astar_runtimes,
        ylabel="Waktu (ms)",
        title="Runtime Pencarian",
        unit="ms",
    )

    fig.suptitle(
        "Perbandingan Kinerja Algoritma: UCS vs A*  —  CERTAN-01 Milestone 1",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Entry Point: Generate Semua Gambar
# ---------------------------------------------------------------------------

def generate_all_figures(output_dir: Path = FIGURES_DIR) -> list[Path]:
    """
    Generate seluruh gambar visualisasi dan simpan ke `output_dir`.

    Gambar yang dihasilkan:
      1. road_network_topology.png          — Peta topologi jaringan jalan
      2. route_solution_skenario_N.png      — Rute UCS vs A* per skenario (4 gambar)
      3. performance_benchmark_chart.png    — Grafik batang komparasi metrik

    Args:
        output_dir: Direktori tujuan. Default: reports/figures/.

    Returns:
        List Path dari semua file gambar yang berhasil disimpan.
    """
    generated: list[Path] = []

    print("[VIZ] Membuat gambar 1/6: Topologi Jaringan Jalan...")
    p = plot_road_network_topology(output_dir)
    generated.append(p)
    print(f"      ✓ {p}")

    scenarios = list_available_scenarios()
    for idx, scenario in enumerate(scenarios, start=2):
        label = scenario["label"]
        print(f"[VIZ] Membuat gambar {idx}/6: Rute — {label}...")
        p = plot_route_solution_comparison(
            hub_id=scenario["hub_id"],
            goal_id=scenario["goal_id"],
            scenario_label=label,
            output_dir=output_dir,
        )
        generated.append(p)
        print(f"      ✓ {p}")

    print("[VIZ] Membuat gambar 6/6: Grafik Benchmark...")
    p = plot_performance_benchmark_chart(output_dir)
    generated.append(p)
    print(f"      ✓ {p}")

    print(f"\n[VIZ] Selesai! {len(generated)} gambar tersimpan di: {output_dir.resolve()}")
    return generated
