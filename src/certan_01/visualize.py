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

import json
import math
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
from certan_01.osm_loader import LANDMARK_LABELS, LANDMARKS, get_landmark_info
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

    Fitur Utama:
      - Desain UI Dashboard Dark Slate Modern (tanpa gradien aneh, tanpa emoji).
      - Panel Kontrol Interaktif dengan Tab: Ringkasan, Navigasi Rute, Landmark, Benchmark.
      - Tombol Aksi Cepat: Fokus Rute, Ke Titik Asal, Ke Titik Tujuan, Reset Peta.
      - Minimasi / Perluasan Panel (Collapsible floating card).
      - Desain Landmark Komprehensif: Pin DivIcon dengan warna kategori solid,
        popup detail lengkap (alamat, deskripsi, catatan logistik kurir, salin GPS).
      - Interaktivitas Antar-Elemen: Klik waypoint atau landmark di panel otomatis
        memusatkan peta (flyTo), membuat efek pulse, dan membuka popup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    hub_node = graph.hub_node()
    goal_node = graph.goal_node()

    if hub_node and goal_node:
        center_lat = (hub_node.y + goal_node.y) / 2
        center_lon = (hub_node.x + goal_node.x) / 2
    elif hub_node:
        center_lat, center_lon = hub_node.y, hub_node.x
    else:
        center_lat, center_lon = 2.3750, 99.1370

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
    )
    map_var_name = m.get_name()

    # Base Tile Layers (Tanpa Emoji)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS",
        name="Peta Jalan Esri (Topografi & Kota)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Peta Standar)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, iCubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Citra Satelit Esri (Foto Udara)",
        overlay=False,
        control=True,
    ).add_to(m)

    # ---- Layer 1: Seluruh ruas jalan OSM (Background Tipis) ----
    all_roads_layer = folium.FeatureGroup(name="Jaringan Jalan Nyata (OSM)", show=True)
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
                    [edge.to_node.y, edge.to_node.x],
                ],
                color="#64748B",
                weight=1.5,
                opacity=0.35,
                tooltip=f"{edge.from_node.name} &rarr; {edge.to_node.name} | "
                        f"{edge.distance_km:.3f} km | Rp {edge.total_cost:,.0f}"
            ).add_to(all_roads_layer)
    all_roads_layer.add_to(m)

    # ---- Layer 2: Rute Pembanding Jarak Terpendek ----
    if dist_result and dist_result.found and len(dist_result.path_ids) > 1:
        f_cost = dist_fuel_cost if dist_fuel_cost is not None else 0.0
        dist_layer = folium.FeatureGroup(
            name=f"Rute Jarak Terpendek | {dist_result.total_distance:.2f} km (BBM: Rp {f_cost:,.0f})",
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

    # ---- Layer 3: Rute Komparasi Algoritma UCS ----
    if ucs_result.found and len(ucs_result.path_ids) > 1:
        ucs_layer = folium.FeatureGroup(
            name=f"Algoritma UCS (BBM) | Rp {ucs_result.total_cost:,.0f} | {ucs_result.total_distance:.2f} km",
            show=False
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
                    f"Simpul diekspansi: {ucs_result.nodes_expanded}"
        ).add_to(ucs_layer)
        ucs_layer.add_to(m)

    # ---- Layer 4: Rute Rekomendasi Optimal (A*) ----
    astar_coords: list[list[float]] = []
    if astar_result.found and len(astar_result.path_ids) > 1:
        astar_layer = folium.FeatureGroup(
            name=f"Rute Rekomendasi Optimal (A*) | Rp {astar_result.total_cost:,.0f} | {astar_result.total_distance:.2f} km",
            show=True
        )
        for nid in astar_result.path_ids:
            node = graph.get_node(nid)
            astar_coords.append([node.y, node.x])

        folium.PolyLine(
            locations=astar_coords,
            color="#7C3AED",
            weight=6,
            opacity=0.95,
            tooltip=f"Rute Rekomendasi Optimal (A*) | Biaya: Rp {astar_result.total_cost:,.0f} | "
                    f"Jarak: {astar_result.total_distance:.2f} km | "
                    f"Simpul diekspansi: {astar_result.nodes_expanded}"
        ).add_to(astar_layer)
        astar_layer.add_to(m)

    # ---- Layer 5: Landmark & Fasilitas Wilayah ----
    landmarks_layer = folium.FeatureGroup(name="Landmark & Fasilitas Wilayah", show=True)
    active_points = set()
    if hub_node:
        active_points.add((round(hub_node.y, 4), round(hub_node.x, 4)))
    if goal_node:
        active_points.add((round(goal_node.y, 4), round(goal_node.x, 4)))

    corridor_landmarks: list[dict[str, Any]] = []

    for lm_key, (l_lat, l_lon) in LANDMARKS.items():
        if lm_key.startswith("jl_") or lm_key.startswith("trans_") or lm_key.startswith("jembatan_"):
            continue
        if (round(l_lat, 4), round(l_lon, 4)) in active_points:
            continue

        info = get_landmark_info(lm_key)
        clean_name = info["name"].replace("'", "\\'")

        # Hitung jarak kasar landmark ke titik pusat rute
        dist_to_center = math.sqrt((l_lat - center_lat) ** 2 + (l_lon - center_lon) ** 2) * 111.0

        corridor_landmarks.append({
            "key": lm_key,
            "name": info["name"],
            "category": info["category"],
            "cat_key": info["cat_key"],
            "icon": info["icon"],
            "color": info["color"],
            "lat": l_lat,
            "lon": l_lon,
            "address": info["address"],
            "description": info["description"],
            "courier_note": info["courier_note"],
            "dist_km": dist_to_center,
        })

        # Marker HTML Custom DivIcon
        marker_html = f"""
        <div class="certan-map-pin" style="--pin-color: {info['color']};" title="{info['name']}">
          <div class="certan-pin-head">
            <i class="{info['icon']}"></i>
          </div>
          <div class="certan-pin-needle"></div>
        </div>
        """

        popup_html = f"""
        <div class="certan-popup-card">
          <div class="certan-popup-header" style="background: {info['color']};">
            <div class="certan-popup-badge"><i class="{info['icon']}"></i> {info['category']}</div>
            <div class="certan-popup-coords">{l_lat:.5f}, {l_lon:.5f}</div>
          </div>
          <div class="certan-popup-body">
            <h4 class="certan-popup-title">{info['name']}</h4>
            <div class="certan-popup-address"><i class="fa-solid fa-location-dot"></i> {info['address']}</div>
            <div class="certan-popup-desc">{info['description']}</div>
            <div class="certan-popup-courier-box">
              <div class="certan-popup-courier-title"><i class="fa-solid fa-truck-ramp-box"></i> Catatan Logistik Kurir</div>
              <div class="certan-popup-courier-desc">{info['courier_note']}</div>
            </div>
            <div class="certan-popup-actions">
              <button class="certan-btn-action" onclick="navigator.clipboard.writeText('{l_lat:.6f}, {l_lon:.6f}'); this.innerHTML='Tersalin!'; setTimeout(() => this.innerHTML='Salin GPS', 2000);">
                <i class="fa-solid fa-copy"></i> Salin GPS
              </button>
              <button class="certan-btn-action certan-btn-primary" onclick="focusLandmark({l_lat}, {l_lon}, '{clean_name}')">
                <i class="fa-solid fa-magnifying-glass-plus"></i> Fokus Lokasi
              </button>
            </div>
          </div>
        </div>
        """

        folium.Marker(
            location=[l_lat, l_lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=folium.Tooltip(f"[{info['category']}] {info['name']}"),
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(32, 40),
                icon_anchor=(16, 40),
                popup_anchor=(0, -40),
            )
        ).add_to(landmarks_layer)

    landmarks_layer.add_to(m)

    # Urutkan landmark berdasarkan jarak terdekat ke rute
    corridor_landmarks.sort(key=lambda x: x["dist_km"])

    # ---- Marker Khusus: Hub Asal & Tujuan ----
    if hub_node:
        hub_marker_html = f"""
        <div class="certan-origin-pin" title="Titik Asal: {hub_node.name}">
          <div class="certan-pulse-ring"></div>
          <div class="certan-hub-head">
            <i class="fa-solid fa-warehouse"></i>
          </div>
          <div class="certan-pin-needle certan-hub-needle"></div>
        </div>
        """
        hub_popup_html = f"""
        <div class="certan-popup-card">
          <div class="certan-popup-header" style="background: #2563EB;">
            <div class="certan-popup-badge"><i class="fa-solid fa-warehouse"></i> Titik Asal Logistik</div>
            <div class="certan-popup-coords">{hub_node.y:.5f}, {hub_node.x:.5f}</div>
          </div>
          <div class="certan-popup-body">
            <h4 class="certan-popup-title">{hub_node.name}</h4>
            <div class="certan-popup-address"><i class="fa-solid fa-location-dot"></i> Titik Awal Pengiriman Kurir</div>
            <div class="certan-popup-desc">Hub logistik keberangkatan armada kurir untuk distribusi paket di koridor Laguboti-Toba.</div>
            <div class="certan-popup-courier-box">
              <div class="certan-popup-courier-title"><i class="fa-solid fa-box"></i> Status Depo</div>
              <div class="certan-popup-courier-desc">Armada siap diberangkatkan menuju rute rekomendasi optimal A*.</div>
            </div>
            <div class="certan-popup-actions">
              <button class="certan-btn-action" onclick="navigator.clipboard.writeText('{hub_node.y:.6f}, {hub_node.x:.6f}'); this.innerHTML='Tersalin!'; setTimeout(() => this.innerHTML='Salin GPS', 2000);">
                <i class="fa-solid fa-copy"></i> Salin GPS
              </button>
              <button class="certan-btn-action certan-btn-primary" onclick="flyToOrigin()">
                <i class="fa-solid fa-crosshairs"></i> Pusatkan Asal
              </button>
            </div>
          </div>
        </div>
        """
        folium.Marker(
            location=[hub_node.y, hub_node.x],
            popup=folium.Popup(hub_popup_html, max_width=320),
            tooltip=folium.Tooltip(f"[Titik Asal] {hub_node.name}"),
            icon=folium.DivIcon(
                html=hub_marker_html,
                icon_size=(40, 48),
                icon_anchor=(20, 48),
                popup_anchor=(0, -48),
            )
        ).add_to(m)

    if goal_node:
        goal_marker_html = f"""
        <div class="certan-dest-pin" title="Titik Tujuan: {goal_node.name}">
          <div class="certan-pulse-ring certan-dest-pulse"></div>
          <div class="certan-dest-head">
            <i class="fa-solid fa-flag-checkered"></i>
          </div>
          <div class="certan-pin-needle certan-dest-needle"></div>
        </div>
        """
        goal_popup_html = f"""
        <div class="certan-popup-card">
          <div class="certan-popup-header" style="background: #059669;">
            <div class="certan-popup-badge"><i class="fa-solid fa-flag-checkered"></i> Tujuan Pengantaran</div>
            <div class="certan-popup-coords">{goal_node.y:.5f}, {goal_node.x:.5f}</div>
          </div>
          <div class="certan-popup-body">
            <h4 class="certan-popup-title">{goal_node.name}</h4>
            <div class="certan-popup-address"><i class="fa-solid fa-location-dot"></i> Alamat Target Pengiriman Paket</div>
            <div class="certan-popup-desc">Titik akhir tujuan pengantaran paket oleh kurir di lokasi penerima.</div>
            <div class="certan-popup-courier-box">
              <div class="certan-popup-courier-title"><i class="fa-solid fa-circle-check"></i> Target Delivery</div>
              <div class="certan-popup-courier-desc">Penerima berada di lokasi ini. Hubungi kontak penerima saat mendekati titik tujuan.</div>
            </div>
            <div class="certan-popup-actions">
              <button class="certan-btn-action" onclick="navigator.clipboard.writeText('{goal_node.y:.6f}, {goal_node.x:.6f}'); this.innerHTML='Tersalin!'; setTimeout(() => this.innerHTML='Salin GPS', 2000);">
                <i class="fa-solid fa-copy"></i> Salin GPS
              </button>
              <button class="certan-btn-action certan-btn-primary" onclick="flyToDestination()">
                <i class="fa-solid fa-crosshairs"></i> Pusatkan Tujuan
              </button>
            </div>
          </div>
        </div>
        """
        folium.Marker(
            location=[goal_node.y, goal_node.x],
            popup=folium.Popup(goal_popup_html, max_width=320),
            tooltip=folium.Tooltip(f"[Tujuan] {goal_node.name}"),
            icon=folium.DivIcon(
                html=goal_marker_html,
                icon_size=(40, 48),
                icon_anchor=(20, 48),
                popup_anchor=(0, -48),
            )
        ).add_to(m)

    # ---- Ekstraksi Waypoint Navigasi Rute (Tab 2) ----
    route_waypoints: list[dict[str, Any]] = []
    if astar_result.found and len(astar_result.path_ids) > 1:
        cum_dist = 0.0
        cum_cost = 0.0
        raw_steps: list[dict[str, Any]] = []

        for i, nid in enumerate(astar_result.path_ids):
            node = graph.get_node(nid)
            seg_dist = 0.0
            seg_cost = 0.0
            if i > 0:
                prev_nid = astar_result.path_ids[i - 1]
                for edge in graph.get_neighbors(prev_nid):
                    if edge.to_node.node_id == nid:
                        seg_dist = edge.distance_km
                        seg_cost = edge.total_cost
                        break
            cum_dist += seg_dist
            cum_cost += seg_cost

            raw_steps.append({
                "idx": i + 1,
                "node_id": nid,
                "name": node.name,
                "lat": node.y,
                "lon": node.x,
                "cum_dist": cum_dist,
                "cum_cost": cum_cost,
                "is_hub": (i == 0),
                "is_goal": (i == len(astar_result.path_ids) - 1),
            })

        # Sampel step jika simpul sangat banyak agar UI rapi
        if len(raw_steps) <= 10:
            route_waypoints = raw_steps
        else:
            step_stride = max(1, len(raw_steps) // 7)
            sampled_indices = set(range(0, len(raw_steps), step_stride))
            sampled_indices.add(0)
            sampled_indices.add(len(raw_steps) - 1)
            route_waypoints = [raw_steps[i] for i in sorted(sampled_indices)]

    # ---- Komputasi Metrik & Evaluasi Rute ----
    veh_text = vehicle_label if vehicle_label else "Sepeda Motor Matik Kurir (Honda BeAT 110cc) | 50.0 km/L | Pertalite Rp 10.000/L"
    # Bersihkan vehicle text dari emoji jika ada
    clean_veh_text = (
        veh_text.replace("🛵", "")
                .replace("🏍️", "")
                .replace("🚗", "")
                .replace("🚚", "")
                .strip()
    )

    saved_nodes = 0
    saved_pct = 0.0
    if ucs_result.found and astar_result.found and ucs_result.nodes_expanded > 0:
        saved_nodes = max(0, ucs_result.nodes_expanded - astar_result.nodes_expanded)
        saved_pct = (saved_nodes / ucs_result.nodes_expanded) * 100

    savings_rp = 0.0
    savings_pct = 0.0
    extra_km = 0.0
    comparison_html = ""
    status_rute_badge = "Rute Optimal Terverifikasi"

    if dist_result and dist_result.found and dist_fuel_cost is not None:
        savings_rp = dist_fuel_cost - astar_result.total_cost
        extra_km = astar_result.total_distance - dist_result.total_distance
        if abs(extra_km) < 0.01 and abs(savings_rp) < 1.0:
            status_rute_badge = "Optimal Sempurna (Jarak & BBM)"
            comparison_html = """
            <div class="certan-callout certan-callout-success">
              <div class="certan-callout-title"><i class="fa-solid fa-circle-check"></i> Status Solusi Rute</div>
              <div class="certan-callout-text">Rute terpendek secara jarak sekaligus merupakan rute paling hemat BBM.</div>
            </div>
            """
        elif savings_rp > 0:
            savings_pct = (savings_rp / dist_fuel_cost * 100) if dist_fuel_cost > 0 else 0.0
            status_rute_badge = f"Hemat BBM {savings_pct:.1f}%"
            comparison_html = f"""
            <div class="certan-callout certan-callout-success">
              <div class="certan-callout-title"><i class="fa-solid fa-gas-pump"></i> Penghematan Bahan Bakar</div>
              <div class="certan-callout-text">
                Hemat <b>Rp {savings_rp:,.0f} ({savings_pct:.1f}%)</b> dibanding rute terpendek.<br>
                <span style="color:#94A3B8; font-size:10.5px;">Kompensasi jarak: +{extra_km:.2f} km melalui ruas jalan berkonsumsi lebih rendah.</span>
              </div>
            </div>
            """
        else:
            comparison_html = f"""
            <div class="certan-callout certan-callout-info">
              <div class="certan-callout-title"><i class="fa-solid fa-route"></i> Hasil Evaluasi Rute</div>
              <div class="certan-callout-text">Rute optimal dengan jarak {astar_result.total_distance:.2f} km dan biaya BBM Rp {astar_result.total_cost:,.0f}.</div>
            </div>
            """
    else:
        comparison_html = f"""
        <div class="certan-callout certan-callout-info">
          <div class="certan-callout-title"><i class="fa-solid fa-route"></i> Jalur Optimal A*</div>
          <div class="certan-callout-text">Rute dihitung berdasarkan optimasi biaya bahan bakar riil kontur jalan OSM.</div>
        </div>
        """

    # Format Judul Skenario Ramah Pengguna
    display_title = scenario_name
    if scenario_name.startswith("chat_") and "_to_" in scenario_name:
        parts = scenario_name[5:].split("_to_")
        if len(parts) == 2:
            s_orig = LANDMARK_LABELS.get(parts[0], parts[0].replace("_", " ").title())
            s_dest = LANDMARK_LABELS.get(parts[1], parts[1].replace("_", " ").title())
            display_title = f"{s_orig} &rarr; {s_dest}"
    elif "_" in scenario_name:
        display_title = scenario_name.replace("_", " ").title()

    # Siapkan data JSON untuk JS interaktif
    js_astar_coords = json.dumps(astar_coords)
    js_hub_coords = json.dumps([hub_node.y, hub_node.x] if hub_node else [center_lat, center_lon])
    js_goal_coords = json.dumps([goal_node.y, goal_node.x] if goal_node else [center_lat, center_lon])

    # Bangun elemen HTML Waypoints (Tab 2)
    waypoints_html_list = []
    for s in route_waypoints:
        step_icon = "fa-warehouse" if s["is_hub"] else ("fa-flag-checkered" if s["is_goal"] else "fa-location-dot")
        step_color = "#2563EB" if s["is_hub"] else ("#059669" if s["is_goal"] else "#64748B")
        clean_step_name = s["name"].replace("'", "\\'")
        waypoints_html_list.append(f"""
        <div class="certan-step-row" onclick="focusStep({s['lat']}, {s['lon']}, '{clean_step_name}')">
          <div class="certan-step-badge" style="background: {step_color};">
            <i class="fa-solid {step_icon}"></i>
          </div>
          <div class="certan-step-content">
            <div class="certan-step-name">{s['name']}</div>
            <div class="certan-step-meta">Akumulasi: {s['cum_dist']:.2f} km &bull; Rp {s['cum_cost']:,.0f}</div>
          </div>
          <div class="certan-step-arrow"><i class="fa-solid fa-chevron-right"></i></div>
        </div>
        """)
    waypoints_html = "\n".join(waypoints_html_list) if waypoints_html_list else "<div style='color:#94A3B8; font-size:11px; padding:10px;'>Data titik jalur tidak tersedia.</div>"

    # Bangun elemen HTML Landmark Koridor (Tab 3)
    landmarks_html_list = []
    for lm in corridor_landmarks:
        clean_lm_title = lm["name"].replace("'", "\\'")
        landmarks_html_list.append(f"""
        <div class="certan-lm-item" data-cat="{lm['cat_key']}" onclick="focusLandmark({lm['lat']}, {lm['lon']}, '{clean_lm_title}')">
          <div class="certan-lm-icon" style="background: {lm['color']};">
            <i class="{lm['icon']}"></i>
          </div>
          <div class="certan-lm-content">
            <div class="certan-lm-title">{lm['name']}</div>
            <div class="certan-lm-category">{lm['category']}</div>
            <div class="certan-lm-address">{lm['address']}</div>
          </div>
          <button class="certan-lm-btn" title="Fokuskan Lokasi">
            <i class="fa-solid fa-location-arrow"></i>
          </button>
        </div>
        """)
    landmarks_html = "\n".join(landmarks_html_list) if landmarks_html_list else "<div style='color:#94A3B8; font-size:11px; padding:10px;'>Tidak ada landmark di koridor ini.</div>"

    # Baris Tabel Benchmark (Tab 4)
    dist_val_str = f"{dist_result.total_distance:.2f} km" if dist_result and dist_result.found else "-"
    dist_cost_str = f"Rp {dist_fuel_cost:,.0f}" if dist_fuel_cost is not None else "-"
    dist_nodes_str = str(dist_result.nodes_expanded) if dist_result and dist_result.found else "-"

    benchmark_table_html = f"""
    <table class="certan-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Rekomendasi A*</th>
          <th>Algoritma UCS</th>
          <th>Jarak Terpendek</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Biaya BBM Riil</b></td>
          <td style="color:#A7F3D0; font-weight:700;">Rp {astar_result.total_cost:,.0f}</td>
          <td>Rp {ucs_result.total_cost:,.0f}</td>
          <td style="color:#FDBA74;">{dist_cost_str}</td>
        </tr>
        <tr>
          <td><b>Total Jarak</b></td>
          <td>{astar_result.total_distance:.2f} km</td>
          <td>{ucs_result.total_distance:.2f} km</td>
          <td style="color:#FDBA74;">{dist_val_str}</td>
        </tr>
        <tr>
          <td><b>Simpul Diekspansi</b></td>
          <td style="color:#67E8F9; font-weight:700;">{astar_result.nodes_expanded}</td>
          <td>{ucs_result.nodes_expanded}</td>
          <td>{dist_nodes_str}</td>
        </tr>
        <tr>
          <td><b>Efisiensi Simpul</b></td>
          <td style="color:#A7F3D0; font-weight:700;">Hemat {saved_pct:.1f}%</td>
          <td>Baseline (0%)</td>
          <td>-</td>
        </tr>
        <tr>
          <td><b>Optimalitas BBM</b></td>
          <td style="color:#A7F3D0; font-weight:700;">Optimal (100%)</td>
          <td style="color:#A7F3D0;">Optimal (100%)</td>
          <td style="color:#FCA5A5;">Sub-optimal</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size:10.5px; color:#94A3B8; line-height:1.45; margin-top:10px;">
      Heuristik Euclidean yang <i>admissible</i> dan konsisten membuktikan bahwa pencarian A*
      menghasilkan jalur dengan biaya BBM yang sama optimalnya dengan UCS, namun dengan
      pemangkasan simpul signifikan sebesar <b>{saved_pct:.1f}%</b>.
    </div>
    """

    # Panel Kontrol Utama HTML
    panel_html = f"""
    <div class="certan-panel" id="certan-main-panel">
      <!-- Header Panel -->
      <div class="certan-panel-header">
        <div class="certan-header-top">
          <span class="certan-status-badge">
            <i class="fa-solid fa-route"></i> REKOMENDASI OPTIMAL (A*)
          </span>
          <button class="certan-btn-minimize" id="btn-toggle-panel" onclick="toggleCardPanel()" title="Perkecil / Perluas Panel">
            <i id="toggle-panel-icon" class="fa-solid fa-chevron-up"></i>
          </button>
        </div>
        <h3 class="certan-panel-title">{display_title}</h3>
        <!-- Mini Summary saat Panel Diciutkan -->
        <div class="certan-mini-summary" id="certan-mini-summary" onclick="toggleCardPanel()">
          <span class="certan-mini-stat">Rp {astar_result.total_cost:,.0f}</span>
          <span style="color:#64748B;">&bull;</span>
          <span class="certan-mini-stat">{astar_result.total_distance:.2f} km</span>
          <span class="certan-mini-tag">Buka Detail <i class="fa-solid fa-chevron-down"></i></span>
        </div>
      </div>

      <!-- Isi Panel yang Dapat Diciutkan -->
      <div class="certan-panel-collapsible" id="certan-panel-collapsible">
        <!-- Action Toolbar Peta Interaktif -->
        <div class="certan-action-bar">
          <button class="certan-quick-btn" onclick="fitRouteBounds()" title="Tampilkan seluruh jalur rute">
            <i class="fa-solid fa-expand"></i> Fokus Rute
          </button>
          <button class="certan-quick-btn" onclick="flyToOrigin()" title="Pusatkan ke titik asal">
            <i class="fa-solid fa-warehouse"></i> Ke Asal
          </button>
          <button class="certan-quick-btn" onclick="flyToDestination()" title="Pusatkan ke titik tujuan">
            <i class="fa-solid fa-flag-checkered"></i> Ke Tujuan
          </button>
          <button class="certan-quick-btn" onclick="resetMapView()" title="Reset tampilan peta">
            <i class="fa-solid fa-rotate-left"></i> Reset
          </button>
        </div>

        <!-- Tab Navigasi -->
        <div class="certan-tabs-nav">
          <button class="certan-tab-btn active" data-tab="tab-ringkasan" onclick="switchTab('tab-ringkasan')">
            <i class="fa-solid fa-chart-pie"></i> Ringkasan
          </button>
          <button class="certan-tab-btn" data-tab="tab-navigasi" onclick="switchTab('tab-navigasi')">
            <i class="fa-solid fa-signs-post"></i> Navigasi
          </button>
          <button class="certan-tab-btn" data-tab="tab-landmark" onclick="switchTab('tab-landmark')">
            <i class="fa-solid fa-map-location-dot"></i> Landmark ({len(corridor_landmarks)})
          </button>
          <button class="certan-tab-btn" data-tab="tab-benchmark" onclick="switchTab('tab-benchmark')">
            <i class="fa-solid fa-scale-balanced"></i> Benchmark
          </button>
        </div>

        <!-- Konten Tab Body -->
        <div class="certan-tabs-body">
          <!-- TAB 1: RINGKASAN -->
          <div class="certan-tab-pane active" id="tab-ringkasan">
            <!-- 2x2 Metric Grid -->
            <div class="certan-grid-2x2">
              <div class="certan-metric-card certan-border-indigo">
                <div class="certan-metric-label">Estimasi Biaya BBM</div>
                <div class="certan-metric-value" style="color: #FFFFFF;">
                  Rp {astar_result.total_cost:,.0f}
                </div>
                <div class="certan-metric-sub">Bahan Bakar Riil</div>
              </div>
              <div class="certan-metric-card certan-border-sky">
                <div class="certan-metric-label">Total Jarak Tempuh</div>
                <div class="certan-metric-value" style="color: #FFFFFF;">
                  {astar_result.total_distance:.2f} km
                </div>
                <div class="certan-metric-sub">Jalur Jalan OSM</div>
              </div>
              <div class="certan-metric-card certan-border-emerald">
                <div class="certan-metric-label">Penghematan BBM</div>
                <div class="certan-metric-value" style="color: #A7F3D0;">
                  {"Rp " + f"{savings_rp:,.0f}" if savings_rp > 0 else "Optimal 100%"}
                </div>
                <div class="certan-metric-sub">{"vs Jarak Terpendek" if savings_rp > 0 else "Biaya Paling Irit"}</div>
              </div>
              <div class="certan-metric-card certan-border-amber">
                <div class="certan-metric-label">Efisiensi Pencarian</div>
                <div class="certan-metric-value" style="color: #FDE68A;">
                  {saved_pct:.1f}%
                </div>
                <div class="certan-metric-sub">Pangkas Simpul A*</div>
              </div>
            </div>

            <!-- Box Spesifikasi Armada -->
            <div class="certan-subcard">
              <div class="certan-subcard-title">
                <i class="fa-solid fa-motorcycle" style="color:#38BDF8;"></i> Spesifikasi Armada Pengantaran
              </div>
              <div class="certan-subcard-text">
                {clean_veh_text}
              </div>
            </div>

            <!-- Evaluasi Penghematan & Status -->
            {comparison_html}

            <!-- Efisiensi Algoritma -->
            <div class="certan-info-note">
              <i class="fa-solid fa-bolt" style="color:#F59E0B; margin-right:4px;"></i>
              A* menghemat <b>{saved_pct:.1f}%</b> ekspansi simpul ({astar_result.nodes_expanded} vs {ucs_result.nodes_expanded} simpul eksplorasi UCS).
            </div>

            <!-- Legenda Garis Peta -->
            <div class="certan-legend-box">
              <div class="certan-legend-row">
                <span class="certan-legend-line" style="background:#7C3AED;"></span>
                <span><b>Ungu Solid:</b> Rute Rekomendasi Optimal (A*)</span>
              </div>
              <div class="certan-legend-row">
                <span class="certan-legend-line" style="background:transparent; border-top:3px dashed #EA580C;"></span>
                <span><b>Oranye Putus:</b> Rute Jarak Terpendek Murni</span>
              </div>
              <div class="certan-legend-row">
                <span class="certan-legend-line" style="background:#64748B; opacity:0.6;"></span>
                <span><b>Abu-abu:</b> Jaringan Jalan Nyata (OSM)</span>
              </div>
            </div>
          </div>

          <!-- TAB 2: NAVIGASI & SEGMEN RUTE -->
          <div class="certan-tab-pane" id="tab-navigasi">
            <div class="certan-pane-header-desc">
              Daftar titik persimpangan rute optimal. Klik segmen manapun untuk memperbesar tampilan peta ke titik tersebut:
            </div>
            <div class="certan-step-list">
              {waypoints_html}
            </div>
          </div>

          <!-- TAB 3: LANDMARK KORIDOR -->
          <div class="certan-tab-pane" id="tab-landmark">
            <!-- Filter Kategori Landmark -->
            <div class="certan-filter-bar">
              <button class="certan-filter-pill active" data-cat="all" onclick="filterLandmarks('all')">Semua</button>
              <button class="certan-filter-pill" data-cat="kesehatan" onclick="filterLandmarks('kesehatan')">Kesehatan</button>
              <button class="certan-filter-pill" data-cat="pendidikan" onclick="filterLandmarks('pendidikan')">Pendidikan</button>
              <button class="certan-filter-pill" data-cat="niaga" onclick="filterLandmarks('niaga')">Niaga</button>
              <button class="certan-filter-pill" data-cat="spbu" onclick="filterLandmarks('spbu')">SPBU</button>
              <button class="certan-filter-pill" data-cat="kuliner" onclick="filterLandmarks('kuliner')">Kuliner</button>
              <button class="certan-filter-pill" data-cat="wisata" onclick="filterLandmarks('wisata')">Wisata</button>
            </div>
            <div class="certan-lm-list" id="certan-lm-container">
              {landmarks_html}
            </div>
          </div>

          <!-- TAB 4: BENCHMARK ALGORITMA -->
          <div class="certan-tab-pane" id="tab-benchmark">
            <div class="certan-pane-header-desc">
              Perbandingan komputasi algoritma pada skenario yang sama:
            </div>
            {benchmark_table_html}
          </div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))

    # CSS & JS Styling Injeksi (Bebas Emoji, Bebas Gradien Aneh)
    custom_ui_assets = f"""
    <!-- FontAwesome 6 CDN untuk Ikon Bersih & Tajam -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>

    <style>
      /* --- Leaflet Global Enhancements --- */
      .leaflet-popup-content-wrapper {{
        background: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6) !important;
        padding: 0 !important;
        overflow: hidden !important;
      }}
      .leaflet-popup-content {{
        margin: 0 !important;
        line-height: 1.4 !important;
      }}
      .leaflet-popup-tip {{
        background: #0F172A !important;
      }}
      .leaflet-popup-close-button {{
        color: #94A3B8 !important;
        padding: 8px 10px !important;
        font-size: 16px !important;
      }}
      .leaflet-popup-close-button:hover {{
        color: #FFFFFF !important;
      }}

      /* --- Custom Marker Pins --- */
      .certan-map-pin {{
        position: relative;
        width: 32px;
        height: 40px;
        cursor: pointer;
        user-select: none;
        filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.45));
        transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1);
      }}
      .certan-map-pin:hover {{
        transform: translateY(-5px) scale(1.15);
        z-index: 1000 !important;
      }}
      .certan-pin-head {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--pin-color, #2563EB);
        border: 2px solid #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 13px;
      }}
      .certan-pin-needle {{
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 8px solid var(--pin-color, #2563EB);
      }}

      /* Hub Asal & Tujuan Marker */
      .certan-origin-pin, .certan-dest-pin {{
        position: relative;
        width: 40px;
        height: 48px;
        cursor: pointer;
        user-select: none;
        filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.55));
        transition: transform 0.2s ease;
      }}
      .certan-origin-pin:hover, .certan-dest-pin:hover {{
        transform: translateY(-6px) scale(1.15);
        z-index: 1000 !important;
      }}
      .certan-hub-head {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #2563EB;
        border: 3px solid #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 17px;
      }}
      .certan-dest-head {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #059669;
        border: 3px solid #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 17px;
      }}
      .certan-hub-needle {{
        border-top: 9px solid #2563EB;
      }}
      .certan-dest-needle {{
        border-top: 9px solid #059669;
      }}

      /* Animasi Pulse Ring */
      .certan-pulse-ring {{
        position: absolute;
        top: -6px;
        left: -6px;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        border: 2px solid #2563EB;
        animation: certan-pulse 2s infinite ease-out;
        pointer-events: none;
      }}
      .certan-dest-pulse {{
        border-color: #059669;
      }}
      @keyframes certan-pulse {{
        0% {{ transform: scale(0.8); opacity: 1; }}
        100% {{ transform: scale(1.4); opacity: 0; }}
      }}

      /* --- Popup Modal Card --- */
      .certan-popup-card {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        width: 290px;
        overflow: hidden;
      }}
      .certan-popup-header {{
        padding: 9px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .certan-popup-badge {{
        font-size: 10.5px;
        font-weight: 700;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 6px;
      }}
      .certan-popup-coords {{
        font-size: 10px;
        color: rgba(255, 255, 255, 0.85);
        font-family: monospace;
      }}
      .certan-popup-body {{
        padding: 12px 14px;
        background: #0F172A;
      }}
      .certan-popup-title {{
        font-size: 13.5px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 5px 0;
        line-height: 1.3;
      }}
      .certan-popup-address {{
        font-size: 11px;
        color: #94A3B8;
        margin-bottom: 7px;
        display: flex;
        align-items: flex-start;
        gap: 5px;
      }}
      .certan-popup-desc {{
        font-size: 11.5px;
        color: #CBD5E1;
        margin-bottom: 9px;
        line-height: 1.35;
      }}
      .certan-popup-courier-box {{
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 7px;
        padding: 7px 9px;
        margin-bottom: 10px;
      }}
      .certan-popup-courier-title {{
        font-size: 10px;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 5px;
      }}
      .certan-popup-courier-desc {{
        font-size: 10.5px;
        color: #94A3B8;
        line-height: 1.3;
      }}
      .certan-popup-actions {{
        display: flex;
        gap: 6px;
      }}
      .certan-btn-action {{
        flex: 1;
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
        border-radius: 6px;
        padding: 6px 8px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        transition: all 0.15s ease;
      }}
      .certan-btn-action:hover {{
        background: #334155;
        color: #FFFFFF;
      }}
      .certan-btn-primary {{
        background: #2563EB;
        border-color: #3B82F6;
        color: #FFFFFF;
      }}
      .certan-btn-primary:hover {{
        background: #1D4ED8;
      }}

      /* --- Floating Dashboard Control Panel --- */
      .certan-panel {{
        position: fixed;
        top: 14px;
        right: 14px;
        z-index: 9999;
        width: 370px;
        max-width: calc(100vw - 28px);
        background: rgba(15, 23, 42, 0.96);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.05);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #F8FAFC;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        max-height: calc(100vh - 28px);
        transition: width 0.25s ease, max-height 0.25s ease;
      }}
      .certan-panel.minimized {{
        max-height: 52px !important;
        overflow: hidden;
      }}
      .certan-panel-header {{
        padding: 12px 14px 10px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.98);
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .certan-header-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .certan-status-badge {{
        background: #059669;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }}
      .certan-btn-minimize {{
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #94A3B8;
        border-radius: 6px;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .certan-btn-minimize:hover {{
        background: #334155;
        color: #FFFFFF;
      }}
      .certan-panel-title {{
        margin: 0;
        font-size: 13.5px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.3;
      }}
      .certan-mini-summary {{
        display: none;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        cursor: pointer;
        padding-top: 2px;
      }}
      .certan-panel.minimized .certan-mini-summary {{
        display: flex;
      }}
      .certan-panel.minimized .certan-panel-title {{
        display: none;
      }}
      .certan-mini-stat {{
        font-weight: 700;
        color: #FFFFFF;
      }}
      .certan-mini-tag {{
        margin-left: auto;
        color: #38BDF8;
        font-size: 10.5px;
      }}

      /* Quick Action Bar */
      .certan-action-bar {{
        display: flex;
        gap: 6px;
        padding: 8px 14px;
        background: #1E293B;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }}
      .certan-quick-btn {{
        flex: 1;
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #CBD5E1;
        border-radius: 6px;
        padding: 6px 4px;
        font-size: 10.5px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        transition: all 0.15s ease;
      }}
      .certan-quick-btn:hover {{
        background: #2563EB;
        border-color: #3B82F6;
        color: #FFFFFF;
      }}

      /* Tabs Nav */
      .certan-tabs-nav {{
        display: flex;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.85);
      }}
      .certan-tab-btn {{
        flex: 1;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: #94A3B8;
        font-size: 11px;
        font-weight: 600;
        padding: 8px 2px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        transition: all 0.15s ease;
      }}
      .certan-tab-btn:hover {{
        color: #F1F5F9;
        background: rgba(255, 255, 255, 0.03);
      }}
      .certan-tab-btn.active {{
        color: #38BDF8;
        border-bottom-color: #38BDF8;
        background: rgba(56, 189, 248, 0.06);
      }}

      /* Tabs Body */
      .certan-tabs-body {{
        overflow-y: auto;
        max-height: calc(100vh - 200px);
        padding: 12px 14px;
      }}
      .certan-tabs-body::-webkit-scrollbar {{
        width: 5px;
      }}
      .certan-tabs-body::-webkit-scrollbar-thumb {{
        background: #334155;
        border-radius: 4px;
      }}
      .certan-tab-pane {{
        display: none;
      }}
      .certan-tab-pane.active {{
        display: block;
      }}
      .certan-pane-header-desc {{
        font-size: 11px;
        color: #94A3B8;
        margin-bottom: 10px;
        line-height: 1.4;
      }}

      /* 2x2 Metric Grid */
      .certan-grid-2x2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 10px;
      }}
      .certan-metric-card {{
        background: #1E293B;
        border-radius: 8px;
        padding: 8px 10px;
      }}
      .certan-border-indigo {{
        border-left: 3px solid #6366F1;
      }}
      .certan-border-sky {{
        border-left: 3px solid #0284C7;
      }}
      .certan-border-emerald {{
        border-left: 3px solid #059669;
      }}
      .certan-border-amber {{
        border-left: 3px solid #D97706;
      }}
      .certan-metric-label {{
        font-size: 10px;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.3px;
      }}
      .certan-metric-value {{
        font-size: 15px;
        font-weight: 700;
        margin: 3px 0 1px 0;
      }}
      .certan-metric-sub {{
        font-size: 9.5px;
        color: #64748B;
      }}

      /* Subcards & Callouts */
      .certan-subcard {{
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 8px;
      }}
      .certan-subcard-title {{
        font-size: 11px;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 5px;
      }}
      .certan-subcard-text {{
        font-size: 11px;
        color: #CBD5E1;
        line-height: 1.35;
      }}
      .certan-callout {{
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 8px;
      }}
      .certan-callout-success {{
        background: rgba(5, 150, 105, 0.15);
        border: 1px solid rgba(5, 150, 105, 0.4);
      }}
      .certan-callout-info {{
        background: rgba(2, 132, 199, 0.15);
        border: 1px solid rgba(2, 132, 199, 0.4);
      }}
      .certan-callout-title {{
        font-size: 11px;
        font-weight: 700;
        color: #A7F3D0;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 5px;
      }}
      .certan-callout-info .certan-callout-title {{
        color: #BAE6FD;
      }}
      .certan-callout-text {{
        font-size: 11px;
        color: #F1F5F9;
        line-height: 1.35;
      }}
      .certan-info-note {{
        font-size: 10.5px;
        color: #94A3B8;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 8px;
        margin-top: 8px;
        line-height: 1.4;
      }}
      .certan-legend-box {{
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 11px;
      }}
      .certan-legend-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
        color: #CBD5E1;
      }}
      .certan-legend-line {{
        width: 18px;
        height: 4px;
        border-radius: 2px;
        display: inline-block;
      }}

      /* Waypoints List (Tab 2) */
      .certan-step-list {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .certan-step-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 10px;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .certan-step-row:hover {{
        background: #334155;
        border-color: #38BDF8;
        transform: translateX(3px);
      }}
      .certan-step-badge {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        flex-shrink: 0;
      }}
      .certan-step-content {{
        flex: 1;
        overflow: hidden;
      }}
      .certan-step-name {{
        font-size: 12px;
        font-weight: 700;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .certan-step-meta {{
        font-size: 10px;
        color: #94A3B8;
        margin-top: 1px;
      }}
      .certan-step-arrow {{
        color: #64748B;
        font-size: 11px;
      }}

      /* Filter & Landmark List (Tab 3) */
      .certan-filter-bar {{
        display: flex;
        gap: 5px;
        overflow-x: auto;
        padding-bottom: 8px;
        margin-bottom: 8px;
      }}
      .certan-filter-bar::-webkit-scrollbar {{
        height: 3px;
      }}
      .certan-filter-pill {{
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #94A3B8;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 10.5px;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
      }}
      .certan-filter-pill:hover {{
        color: #FFFFFF;
        border-color: #38BDF8;
      }}
      .certan-filter-pill.active {{
        background: #2563EB;
        border-color: #3B82F6;
        color: #FFFFFF;
      }}
      .certan-lm-list {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .certan-lm-item {{
        display: flex;
        align-items: center;
        gap: 9px;
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 10px;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .certan-lm-item:hover {{
        background: #334155;
        border-color: #38BDF8;
        transform: translateX(3px);
      }}
      .certan-lm-icon {{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        flex-shrink: 0;
      }}
      .certan-lm-content {{
        flex: 1;
        overflow: hidden;
      }}
      .certan-lm-title {{
        font-size: 12px;
        font-weight: 700;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .certan-lm-category {{
        font-size: 10px;
        color: #38BDF8;
        font-weight: 600;
      }}
      .certan-lm-address {{
        font-size: 9.5px;
        color: #94A3B8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .certan-lm-btn {{
        background: transparent;
        border: none;
        color: #64748B;
        font-size: 12px;
        cursor: pointer;
        padding: 4px;
      }}
      .certan-lm-item:hover .certan-lm-btn {{
        color: #38BDF8;
      }}

      /* Benchmark Table (Tab 4) */
      .certan-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        margin-top: 4px;
      }}
      .certan-table th {{
        background: #0F172A;
        color: #94A3B8;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 9.5px;
        padding: 7px 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        text-align: left;
      }}
      .certan-table td {{
        padding: 7px 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
      }}
      .certan-table tr:hover td {{
        background: rgba(255, 255, 255, 0.02);
      }}

      /* Leaflet Control Layer Overrides */
      .leaflet-control-layers {{
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5) !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 11.5px !important;
        padding: 10px 14px !important;
        min-width: 260px !important;
      }}
      .leaflet-control-layers-base label, .leaflet-control-layers-overlays label {{
        margin-bottom: 5px !important;
        cursor: pointer !important;
        color: #E2E8F0 !important;
        display: flex !important;
        align-items: center !important;
      }}
      .leaflet-control-layers-base input, .leaflet-control-layers-overlays input {{
        margin-right: 8px !important;
        accent-color: #38BDF8 !important;
      }}
      .leaflet-control-layers-separator {{
        border-top: 1px solid rgba(255, 255, 255, 0.15) !important;
        margin: 8px 0 !important;
      }}
      .leaflet-bar {{
        border: none !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35) !important;
      }}
      .leaflet-bar a {{
        background-color: rgba(15, 23, 42, 0.92) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(6px) !important;
        transition: all 0.2s ease !important;
      }}
      .leaflet-bar a:hover {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
      }}
    </style>

    <script>
      // Data Koordinat Rute & Titik Penting
      var _certanRouteCoords = {js_astar_coords};
      var _certanHubCoords = {js_hub_coords};
      var _certanGoalCoords = {js_goal_coords};
      var _certanInitialCenter = [{center_lat}, {center_lon}];
      var _certanInitialZoom = 14;
      var _certanPulseMarker = null;

      function getCertanMap() {{
        if (window["{map_var_name}"]) return window["{map_var_name}"];
        for (var k in window) {{
          if (k.indexOf("map_") === 0 && window[k] && window[k].flyTo) {{
            return window[k];
          }}
        }}
        return null;
      }}

      function showPulseRing(latlng) {{
        var map = getCertanMap();
        if (!map) return;
        if (_certanPulseMarker) {{
          map.removeLayer(_certanPulseMarker);
        }}
        _certanPulseMarker = L.circleMarker(latlng, {{
          radius: 16,
          color: '#38BDF8',
          fillColor: '#38BDF8',
          fillOpacity: 0.35,
          weight: 3
        }}).addTo(map);
        setTimeout(function() {{
          if (_certanPulseMarker) {{
            map.removeLayer(_certanPulseMarker);
            _certanPulseMarker = null;
          }}
        }}, 3500);
      }}

      function fitRouteBounds() {{
        var map = getCertanMap();
        if (!map || !_certanRouteCoords.length) return;
        var bounds = L.latLngBounds(_certanRouteCoords);
        map.fitBounds(bounds, {{ padding: [40, 40], maxZoom: 16, animate: true }});
      }}

      function flyToOrigin() {{
        var map = getCertanMap();
        if (!map || !_certanHubCoords) return;
        map.flyTo(_certanHubCoords, 16, {{ animate: true }});
        showPulseRing(_certanHubCoords);
      }}

      function flyToDestination() {{
        var map = getCertanMap();
        if (!map || !_certanGoalCoords) return;
        map.flyTo(_certanGoalCoords, 16, {{ animate: true }});
        showPulseRing(_certanGoalCoords);
      }}

      function resetMapView() {{
        var map = getCertanMap();
        if (!map) return;
        map.flyTo(_certanInitialCenter, _certanInitialZoom, {{ animate: true }});
      }}

      function focusStep(lat, lon, name) {{
        var map = getCertanMap();
        if (!map) return;
        map.flyTo([lat, lon], 17, {{ animate: true }});
        showPulseRing([lat, lon]);
      }}

      function focusLandmark(lat, lon, lmName) {{
        var map = getCertanMap();
        if (!map) return;
        map.flyTo([lat, lon], 17, {{ animate: true }});
        showPulseRing([lat, lon]);
        // Buka popup landmark jika koordinat cocok
        map.eachLayer(function(layer) {{
          if (layer.getLatLng && layer.getPopup) {{
            var ll = layer.getLatLng();
            if (Math.abs(ll.lat - lat) < 0.00015 && Math.abs(ll.lng - lon) < 0.00015) {{
              setTimeout(function() {{ layer.openPopup(); }}, 250);
            }}
          }}
        }});
      }}

      function switchTab(tabId) {{
        document.querySelectorAll('.certan-tab-btn').forEach(function(btn) {{
          btn.classList.remove('active');
        }});
        document.querySelectorAll('.certan-tab-pane').forEach(function(pane) {{
          pane.classList.remove('active');
        }});
        var activeBtn = document.querySelector('[data-tab="' + tabId + '"]');
        var activePane = document.getElementById(tabId);
        if (activeBtn) activeBtn.classList.add('active');
        if (activePane) activePane.classList.add('active');
      }}

      function filterLandmarks(catKey) {{
        document.querySelectorAll('.certan-filter-pill').forEach(function(pill) {{
          pill.classList.remove('active');
        }});
        var activePill = document.querySelector('.certan-filter-pill[data-cat="' + catKey + '"]');
        if (activePill) activePill.classList.add('active');

        document.querySelectorAll('.certan-lm-item').forEach(function(item) {{
          if (catKey === 'all' || item.getAttribute('data-cat') === catKey) {{
            item.style.display = 'flex';
          }} else {{
            item.style.display = 'none';
          }}
        }});
      }}

      var _panelMinimized = false;
      function toggleCardPanel() {{
        _panelMinimized = !_panelMinimized;
        var panel = document.getElementById('certan-main-panel');
        var icon = document.getElementById('toggle-panel-icon');
        var collapsible = document.getElementById('certan-panel-collapsible');
        if (_panelMinimized) {{
          panel.classList.add('minimized');
          collapsible.style.display = 'none';
          if (icon) icon.className = 'fa-solid fa-chevron-down';
        }} else {{
          panel.classList.remove('minimized');
          collapsible.style.display = 'block';
          if (icon) icon.className = 'fa-solid fa-chevron-up';
        }}
      }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(custom_ui_assets))

    # Layer Control di pojok kiri bawah agar tidak menutupi dashboard
    folium.LayerControl(position="bottomleft", collapsed=False).add_to(m)

    # Bersihkan nama file dari karakter yang tidak valid di Windows
    safe_name = scenario_name.lower()
    for ch in (" ", "->", "→", ":", "/", "\\", "*", "?", "\"", "<", ">", "|"):
        safe_name = safe_name.replace(ch, "_")
    import re as _re
    safe_name = _re.sub(r"_+", "_", safe_name).strip("_")
    out_path = output_dir / f"laguboti_rute_{safe_name}.html"
    m.save(str(out_path))

    return out_path
