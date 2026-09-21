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

import folium

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


# ---------------------------------------------------------------------------
# Folium: Peta Interaktif Rute Laguboti
# ---------------------------------------------------------------------------

def generate_laguboti_folium_map(
    graph: Graph,
    ucs_result: SearchResult,
    astar_result: SearchResult,
    scenario_name: str = "laguboti",
    output_dir: Path = FIGURES_DIR,
    vehicle_label: str | None = None,
    dist_result: SearchResult | None = None,
    dist_fuel_cost: float | None = None,
) -> Path:
    """
    Hasilkan peta interaktif HTML yang menampilkan rute optimal di atas
    jaringan jalan nyata Kecamatan Laguboti menggunakan Folium + Esri tiles.

    Peta menampilkan:
      - Seluruh ruas jalan jaringan nyata (garis abu-abu tipis).
      - Rute Rekomendasi Optimal (A*, ungu solid tebal).
      - Rute Pembanding Jarak Terpendek (oranye putus-putus, jika ada).
      - Rute Komparasi UCS (opsional dalam layer control).
      - Marker Hub (biru) dan Goal (hijau).
      - Panel Rekomendasi Sistem berisi info profil kendaraan kurir,
        biaya BBM, perbandingan penghematan, dan efisiensi pencarian.

    Args:
        graph         : Graf jaringan jalan Laguboti yang digunakan pencarian.
        ucs_result    : Hasil pencarian UCS (berbasis biaya BBM).
        astar_result  : Hasil pencarian A* (berbasis biaya BBM).
        scenario_name : Nama skenario untuk penamaan file output.
        output_dir    : Direktori output untuk menyimpan file HTML.
        vehicle_label : Label deskripsi kendaraan kurir dan spesifikasi BBM.
        dist_result   : Hasil pencarian rute terpendek secara jarak (murni).
        dist_fuel_cost: Biaya BBM riil jika melewati rute jarak terpendek.

    Returns:
        Path: Path file HTML yang dihasilkan.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cari hub dan goal node untuk centering peta
    hub_node  = graph.hub_node()
    goal_node = graph.goal_node()

    # Center peta di titik tengah antara hub dan goal
    if hub_node and goal_node:
        center_lat = (hub_node.y + goal_node.y) / 2
        center_lon = (hub_node.x + goal_node.x) / 2
    elif hub_node:
        center_lat, center_lon = hub_node.y, hub_node.x
    else:
        center_lat, center_lon = 2.3750, 99.1370   # Laguboti default

    # Buat peta dengan Esri World Street Map (free, no API key)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS"
    )

    # ---- Layer 1: Seluruh ruas jalan (background tipis) ----
    all_roads_layer = folium.FeatureGroup(name="🗺️ Jaringan Jalan Laguboti (OSM)", show=True)

    plotted_edges: set[tuple[str, str]] = set()
    for node in graph.get_all_nodes():
        for edge in graph.get_neighbors(node.node_id):
            pair = (min(edge.from_node.node_id, edge.to_node.node_id),
                    max(edge.from_node.node_id, edge.to_node.node_id))
            if pair in plotted_edges:
                continue
            plotted_edges.add(pair)
            folium.PolyLine(
                locations=[
                    [edge.from_node.y, edge.from_node.x],
                    [edge.to_node.y,   edge.to_node.x],
                ],
                color="#94A3B8",
                weight=1.5,
                opacity=0.35,
                tooltip=f"{edge.from_node.name} → {edge.to_node.name} | "
                        f"{edge.distance_km:.3f} km | Rp {edge.total_cost:,.0f}"
            ).add_to(all_roads_layer)
    all_roads_layer.add_to(m)

    # ---- Layer 2: Rute Pembanding Jarak Terpendek (Putus-putus) ----
    if dist_result and dist_result.found and len(dist_result.path_ids) > 1:
        f_cost = dist_fuel_cost if dist_fuel_cost is not None else 0.0
        dist_layer = folium.FeatureGroup(
            name=f"📏 Rute Jarak Terpendek | {dist_result.total_distance:.2f} km (BBM: Rp {f_cost:,.0f})",
            show=True
        )
        dist_coords = []
        for nid in dist_result.path_ids:
            node = graph.get_node(nid)
            dist_coords.append([node.y, node.x])
        folium.PolyLine(
            locations=dist_coords,
            color="#EA580C",
            weight=4,
            dash_array="6, 8",
            opacity=0.85,
            tooltip=f"Rute Jarak Terpendek | {dist_result.total_distance:.2f} km | "
                    f"Biaya BBM Riil: Rp {f_cost:,.0f} | "
                    f"Simpul diekspansi: {dist_result.nodes_expanded}"
        ).add_to(dist_layer)
        dist_layer.add_to(m)

    # ---- Layer 3: Rute Komparasi Algoritma UCS (BBM) ----
    if ucs_result.found and len(ucs_result.path_ids) > 1:
        ucs_layer = folium.FeatureGroup(
            name=f"🔬 Algoritma UCS (BBM) | Rp {ucs_result.total_cost:,.0f} | {ucs_result.total_distance:.2f} km",
            show=False   # Tidak aktif default agar tidak menumpuk, bisa diaktifkan di layer control
        )
        ucs_coords = []
        for nid in ucs_result.path_ids:
            node = graph.get_node(nid)
            ucs_coords.append([node.y, node.x])
        folium.PolyLine(
            locations=ucs_coords,
            color=COLOR_UCS,
            weight=5,
            opacity=0.8,
            tooltip=f"UCS (BBM) | Biaya: Rp {ucs_result.total_cost:,.0f} | "
                    f"Jarak: {ucs_result.total_distance:.2f} km | "
                    f"Simpul: {ucs_result.nodes_expanded}"
        ).add_to(ucs_layer)
        ucs_layer.add_to(m)

    # ---- Layer 4: 🏆 Rute Rekomendasi Optimal (A*) ----
    if astar_result.found and len(astar_result.path_ids) > 1:
        astar_layer = folium.FeatureGroup(
            name=f"🏆 Rekomendasi Optimal (A*) | Rp {astar_result.total_cost:,.0f} | {astar_result.total_distance:.2f} km",
            show=True
        )
        astar_coords = []
        for nid in astar_result.path_ids:
            node = graph.get_node(nid)
            # Geser sedikit jika garis tumpang tindih
            astar_coords.append([node.y + 0.00002, node.x + 0.00002])
        folium.PolyLine(
            locations=astar_coords,
            color="#7C3AED",
            weight=6,
            opacity=0.95,
            tooltip=f"🏆 Rekomendasi Optimal (A*) | Biaya: Rp {astar_result.total_cost:,.0f} | "
                    f"Jarak: {astar_result.total_distance:.2f} km | "
                    f"Simpul diekspansi: {astar_result.nodes_expanded}"
        ).add_to(astar_layer)
        astar_layer.add_to(m)

    # ---- Marker: Hub & Goal ----
    if hub_node:
        folium.Marker(
            location=[hub_node.y, hub_node.x],
            popup=folium.Popup(
                f"<div style='font-family:sans-serif; min-width:180px;'>"
                f"<b style='color:#1E40AF;'>[HUB KURIR]</b><br>"
                f"<b>{hub_node.name}</b><br>"
                f"<span style='font-size:11px; color:#64748B;'>Titik Awal Pengiriman</span><br>"
                f"<span style='font-size:10px; color:#94A3B8;'>({hub_node.y:.5f}, {hub_node.x:.5f})</span>"
                f"</div>",
                max_width=250
            ),
            tooltip=f"[HUB] {hub_node.name}",
            icon=folium.Icon(color="blue", icon="home", prefix="fa")
        ).add_to(m)

    if goal_node:
        folium.Marker(
            location=[goal_node.y, goal_node.x],
            popup=folium.Popup(
                f"<div style='font-family:sans-serif; min-width:180px;'>"
                f"<b style='color:#059669;'>[TUJUAN PAKET]</b><br>"
                f"<b>{goal_node.name}</b><br>"
                f"<span style='font-size:11px; color:#64748B;'>Alamat Pengantaran</span><br>"
                f"<span style='font-size:10px; color:#94A3B8;'>({goal_node.y:.5f}, {goal_node.x:.5f})</span>"
                f"</div>",
                max_width=250
            ),
            tooltip=f"[TUJUAN] {goal_node.name}",
            icon=folium.Icon(color="green", icon="package", prefix="fa")
        ).add_to(m)

    # ---- Komputasi Metrik untuk Panel Info ----
    veh_text = vehicle_label if vehicle_label else "Sepeda Motor Matik Kurir (Honda BeAT 110cc) | 50.0 km/L | Pertalite Rp 10,000/L"

    efficiency_text = ""
    if ucs_result.found and astar_result.found and ucs_result.nodes_expanded > 0:
        saved_nodes = ucs_result.nodes_expanded - astar_result.nodes_expanded
        saved_pct = (saved_nodes / ucs_result.nodes_expanded) * 100
        efficiency_text = (
            f"⚡ <b>Efisiensi Pencarian:</b> A* hemat <b>{saved_pct:.1f}%</b> ekspansi simpul "
            f"({astar_result.nodes_expanded} vs {ucs_result.nodes_expanded} simpul UCS)"
        )

    comparison_html = ""
    if dist_result and dist_result.found and dist_fuel_cost is not None:
        savings_rp = dist_fuel_cost - astar_result.total_cost
        extra_km = astar_result.total_distance - dist_result.total_distance
        if abs(extra_km) < 0.01 and abs(savings_rp) < 1.0:
            comparison_html = (
                "<div style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); "
                "border-radius:6px; padding:6px 10px; margin-top:8px; font-size:11px; color:#A7F3D0;'>"
                "✨ <b>Status Rute:</b> Rute terpendek sekaligus rute paling hemat BBM!"
                "</div>"
            )
        elif savings_rp > 0:
            savings_pct = (savings_rp / dist_fuel_cost * 100) if dist_fuel_cost > 0 else 0.0
            comparison_html = (
                f"<div style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); "
                f"border-radius:6px; padding:6px 10px; margin-top:8px; font-size:11px; color:#A7F3D0;'>"
                f"💰 <b>Hemat BBM:</b> Rp {savings_rp:,.0f} ({savings_pct:.1f}%) dibanding rute terpendek<br>"
                f"<span style='color:#94A3B8; font-size:10px;'>Trade-off: +{extra_km:.2f} km (melewati jalan lebih lancar & irit)</span>"
                f"</div>"
            )
        else:
            comparison_html = (
                f"<div style='background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); "
                f"border-radius:6px; padding:6px 10px; margin-top:8px; font-size:11px; color:#BFDBFE;'>"
                f"ℹ️ <b>Rute Optimal:</b> {astar_result.total_distance:.2f} km | Rp {astar_result.total_cost:,.0f}"
                f"</div>"
            )

    legend_html = f"""
    <div style="
        position: fixed; top: 12px; right: 12px; z-index: 9999;
        background: rgba(15, 23, 42, 0.94); backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
        padding: 16px; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC; width: 320px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
    ">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <span style="background: linear-gradient(135deg, #10B981, #059669); color:#FFFFFF;
                     font-size:10px; font-weight:700; padding:3px 9px; border-radius:20px;
                     text-transform:uppercase; letter-spacing:0.5px;">
          🏆 Rekomendasi Optimal
        </span>
        <span style="font-size:11px; color:#94A3B8; font-weight:600;">Milestone 1</span>
      </div>

      <h3 style="margin:0 0 4px 0; font-size:14px; font-weight:700; color:#FFFFFF;">
        {scenario_name}
      </h3>

      <!-- Box Kendaraan -->
      <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.06);
                  border-radius: 8px; padding: 8px 10px; margin: 8px 0; font-size:11px; line-height:1.4;">
        <span style="color:#38BDF8; font-weight:600;">🛵 Profil Kendaraan:</span><br>
        <span style="color:#CBD5E1;">{veh_text}</span>
      </div>

      <!-- Ringkasan Hasil Rute Terbaik -->
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin:8px 0;">
        <div style="background:rgba(124, 58, 237, 0.15); border:1px solid rgba(124, 58, 237, 0.4);
                    border-radius:8px; padding:8px; text-align:center;">
          <div style="font-size:10px; color:#C4B5FD; text-transform:uppercase;">Estimasi BBM</div>
          <div style="font-size:15px; font-weight:700; color:#FFFFFF; margin-top:2px;">
            Rp {astar_result.total_cost:,.0f}
          </div>
        </div>
        <div style="background:rgba(59, 130, 246, 0.15); border:1px solid rgba(59, 130, 246, 0.4);
                    border-radius:8px; padding:8px; text-align:center;">
          <div style="font-size:10px; color:#93C5FD; text-transform:uppercase;">Total Jarak</div>
          <div style="font-size:15px; font-weight:700; color:#FFFFFF; margin-top:2px;">
            {astar_result.total_distance:.2f} km
          </div>
        </div>
      </div>

      <!-- Perbandingan Penghematan -->
      {comparison_html}

      <!-- Efisiensi Algoritma -->
      <div style="font-size:10.5px; color:#94A3B8; margin-top:8px; border-top:1px solid #334155; padding-top:8px; line-height:1.4;">
        {efficiency_text}
      </div>

      <!-- Keterangan Garis di Peta -->
      <div style="margin-top:10px; padding-top:8px; border-top:1px solid #334155; font-size:11px;">
        <div style="margin-bottom:4px;">
          <span style="background:#7C3AED; width:16px; height:4px; display:inline-block; vertical-align:middle; border-radius:2px;"></span>
          <span style="margin-left:6px; color:#E2E8F0;"><b>Ungu Solid:</b> Rute Rekomendasi (A*)</span>
        </div>
        <div>
          <span style="background:#EA580C; width:16px; height:0px; border-top:3px dashed #EA580C; display:inline-block; vertical-align:middle;"></span>
          <span style="margin-left:6px; color:#E2E8F0;"><b>Oranye Putus:</b> Rute Terpendek</span>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer control di pojok kiri bawah agar tidak menutupi panel info
    folium.LayerControl(position="bottomleft", collapsed=False).add_to(m)

    # Bersihkan nama file dari karakter yang tidak valid di Windows
    safe_name = scenario_name.lower()
    for ch in (" ", "->", "→", ":", "/", "\\", "*", "?", "\"", "<", ">", "|"):
        safe_name = safe_name.replace(ch, "_")
    # Hapus underscore berulang dan trailing underscore
    import re as _re
    safe_name = _re.sub(r"_+", "_", safe_name).strip("_")
    out_path = output_dir / f"laguboti_rute_{safe_name}.html"
    m.save(str(out_path))

    return out_path
