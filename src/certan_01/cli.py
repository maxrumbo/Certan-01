"""
cli.py
------
Antarmuka baris perintah (CLI) interaktif untuk simulasi dan benchmarking
optimasi rute kurir pengiriman paket menggunakan UCS dan A*.

Catatan: Output menggunakan UTF-8 agar karakter khusus tampil dengan benar
pada terminal Windows.

Fitur:
  1. Mode Demo     : Jalankan seluruh skenario pengiriman yang telah didefinisikan
                     dan tampilkan perbandingan kinerja UCS vs A* secara otomatis.
  2. Mode Kustom   : Pilih simpul asal dan tujuan secara bebas dari daftar
                     simpul yang tersedia dalam jaringan jalan.
  3. Mode Benchmark: Jalankan semua skenario lalu cetak tabel komparasi
                     metrik lengkap (runtime, biaya, nodes expanded) dalam
                     format yang siap disalin ke laporan.
  4. Ekspor MD     : Simpan tabel benchmark ke file Markdown (reports/benchmark_results.md).
  5. Ekspor CSV    : Simpan tabel benchmark ke file CSV (reports/benchmark_results.csv).
  6. Visualisasi   : Generate seluruh gambar PNG peta & grafik ke reports/figures/.

Cara menjalankan:
  uv run certan-01              -> Mode interaktif (tanya user)
  uv run certan-01 --demo       -> Langsung tampilkan semua skenario demo
  uv run certan-01 --bench      -> Hanya tampilkan tabel benchmark
  uv run certan-01 --export-md  -> Ekspor benchmark ke Markdown
  uv run certan-01 --export-csv -> Ekspor benchmark ke CSV
  uv run certan-01 --visualize  -> Generate semua gambar visualisasi PNG
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

from certan_01.data import build_urban_road_network, list_available_scenarios
from certan_01.heuristics import euclidean_heuristic
from certan_01.models import Graph
from certan_01.search import SearchResult, astar_search, uniform_cost_search
from certan_01.visualize import generate_all_figures


# ---------------------------------------------------------------------------
# Banner & Helper Display
# ---------------------------------------------------------------------------

BANNER = """
================================================================
  OPTIMASI RUTE KURIR -- UCS vs A*  (CERTAN-01 Milestone 1)
  Institut Teknologi Del | Sistem Cerdas
================================================================
"""


def _print_separator(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _print_graph_info(graph: Graph) -> None:
    """Tampilkan informasi ringkas tentang jaringan jalan yang dimuat."""
    print(f"\n[INFO] {graph.summary()}")
    hub  = graph.hub_node()
    goal = graph.goal_node()
    if hub:
        print(f"   HUB    : {hub.name} ({hub.node_id})")
    if goal:
        print(f"   TUJUAN : {goal.name} ({goal.node_id})")


def _print_comparison(ucs_result: SearchResult, astar_result: SearchResult) -> None:
    """Tampilkan perbandingan hasil UCS vs A* berdampingan."""
    print(ucs_result.summary())
    print(astar_result.summary())

    # Validasi kesamaan biaya optimal
    if ucs_result.found and astar_result.found:
        _print_separator()
        if abs(ucs_result.total_cost - astar_result.total_cost) < 1e-6:
            print("[OK] VERIFIKASI: UCS dan A* menghasilkan biaya optimal yang IDENTIK.")
        else:
            print("[!]  PERHATIAN: Biaya UCS dan A* berbeda -- periksa heuristik!")

        # Efisiensi relatif
        delta_nodes = ucs_result.nodes_expanded - astar_result.nodes_expanded
        if delta_nodes > 0:
            print(
                f"[>>] A* lebih efisien: mengeksplorasi {delta_nodes} simpul lebih sedikit "
                f"dari UCS ({astar_result.nodes_expanded} vs {ucs_result.nodes_expanded})."
            )
        elif delta_nodes == 0:
            print("[==] UCS dan A* mengeksplorasi jumlah simpul yang sama.")
        else:
            print(
                f"[<<] UCS lebih efisien pada kasus ini: {abs(delta_nodes)} simpul lebih "
                f"sedikit dari A*."
            )
        _print_separator()


def _print_benchmark_table(results: list[tuple[str, SearchResult, SearchResult]]) -> None:
    """
    Cetak tabel komparasi benchmark semua skenario.
    results: list of (scenario_label, ucs_result, astar_result)
    """
    print("\n" + "=" * 90)
    print(" TABEL BENCHMARK KOMPARASI: UCS vs A*")
    print("=" * 90)

    header = (
        f"{'Skenario':<35} | {'Algo':<5} | {'Biaya (Rp)':>12} | "
        f"{'Jarak (km)':>10} | {'Simpul':>6} | {'Runtime (ms)':>12}"
    )
    print(header)
    print("-" * 90)

    for label, ucs, astar in results:
        short_label = label[:33]
        for result in (ucs, astar):
            algo_short = "UCS" if "UCS" in result.algorithm else "A*"
            cost_str = f"Rp {result.total_cost:,.0f}" if result.found else "N/A"
            dist_str = f"{result.total_distance:.2f}" if result.found else "N/A"
            nodes_str = str(result.nodes_expanded)
            rt_str = f"{result.runtime_ms:.4f}"
            print(
                f"{short_label:<35} | {algo_short:<5} | {cost_str:>12} | "
                f"{dist_str:>10} | {nodes_str:>6} | {rt_str:>12}"
            )
        print("-" * 90)
        short_label = ""  # Hanya tampilkan label sekali per skenario

    print("=" * 90)
    print("Keterangan: Simpul = jumlah simpul yang dieksplorasi (nodes expanded)\n")


# ---------------------------------------------------------------------------
# Mode Demo: Jalankan semua skenario yang telah didefinisikan
# ---------------------------------------------------------------------------

def run_demo_mode() -> None:
    """Jalankan semua skenario demo dan tampilkan perbandingan UCS vs A*."""
    print(BANNER)
    scenarios = list_available_scenarios()
    benchmark_data: list[tuple[str, SearchResult, SearchResult]] = []

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n{'='*64}")
        print(f"  [Paket] {scenario['label']}")
        print(f"          {scenario['description']}")
        print(f"{'='*64}")

        graph = build_urban_road_network(
            hub_id=scenario["hub_id"],
            goal_id=scenario["goal_id"],
        )
        _print_graph_info(graph)

        ucs_result   = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        astar_result = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"],
            heuristic_fn=euclidean_heuristic,
        )

        _print_comparison(ucs_result, astar_result)
        benchmark_data.append((scenario["label"], ucs_result, astar_result))

    _print_benchmark_table(benchmark_data)


# ---------------------------------------------------------------------------
# Mode Kustom: Pilih simpul asal dan tujuan secara manual
# ---------------------------------------------------------------------------

def run_custom_mode() -> None:
    """Mode interaktif: pengguna memilih titik asal dan tujuan sendiri."""
    print(BANNER)

    # Tampilkan daftar simpul yang tersedia
    graph = build_urban_road_network()
    nodes = graph.get_all_nodes()
    print("\n[Simpul] Daftar Simpul Tersedia pada Jaringan Jalan Perkotaan:")
    print(f"   {'ID':<4}  {'Nama Lokasi':<35} {'Koordinat (x, y km)'}")
    _print_separator()
    for node in nodes:
        print(f"   [{node.node_id:<2}]  {node.name:<35} ({node.x:.1f}, {node.y:.1f})")
    _print_separator()

    try:
        start_id = input("\n[>] Masukkan ID simpul ASAL (Hub Kurir)  : ").strip().upper()
        goal_id  = input("[>] Masukkan ID simpul TUJUAN (Pelanggan): ").strip().upper()
    except (KeyboardInterrupt, EOFError):
        print("\nDibatalkan.")
        return

    # Validasi input
    try:
        graph.get_node(start_id)
        graph.get_node(goal_id)
    except KeyError as e:
        print(f"\n[ERROR] Simpul dengan ID '{e.args[0]}' tidak ditemukan dalam jaringan.")
        return

    if start_id == goal_id:
        print("\n[!] Titik asal dan tujuan sama. Tidak perlu pencarian.")
        return

    # Rebuild dengan hub dan goal yang dipilih
    graph = build_urban_road_network(hub_id=start_id, goal_id=goal_id)
    _print_graph_info(graph)

    print(f"\n[>>] Menjalankan UCS dan A* dari [{start_id}] ke [{goal_id}]...\n")
    ucs_result   = uniform_cost_search(graph, start_id, goal_id)
    astar_result = astar_search(
        graph, start_id, goal_id,
        heuristic_fn=euclidean_heuristic,
    )
    _print_comparison(ucs_result, astar_result)


# ---------------------------------------------------------------------------
# Mode Benchmark: Hanya cetak tabel komparasi
# ---------------------------------------------------------------------------

def run_benchmark_mode() -> None:
    """Jalankan semua skenario dan cetak hanya tabel benchmark tanpa narasi."""
    scenarios = list_available_scenarios()
    benchmark_data: list[tuple[str, SearchResult, SearchResult]] = []

    for scenario in scenarios:
        graph = build_urban_road_network(
            hub_id=scenario["hub_id"],
            goal_id=scenario["goal_id"],
        )
        ucs_result   = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        astar_result = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"],
            heuristic_fn=euclidean_heuristic,
        )
        benchmark_data.append((scenario["label"], ucs_result, astar_result))

    _print_benchmark_table(benchmark_data)


# ---------------------------------------------------------------------------
# Ekspor: Markdown & CSV
# ---------------------------------------------------------------------------

def _collect_benchmark_data() -> list[tuple[str, "SearchResult", "SearchResult"]]:
    """Jalankan semua skenario dan kembalikan data benchmark mentah."""
    scenarios = list_available_scenarios()
    benchmark_data: list[tuple[str, SearchResult, SearchResult]] = []
    for scenario in scenarios:
        graph = build_urban_road_network(
            hub_id=scenario["hub_id"],
            goal_id=scenario["goal_id"],
        )
        ucs_result   = uniform_cost_search(graph, scenario["hub_id"], scenario["goal_id"])
        astar_result = astar_search(
            graph, scenario["hub_id"], scenario["goal_id"],
            heuristic_fn=euclidean_heuristic,
        )
        benchmark_data.append((scenario["label"], ucs_result, astar_result))
    return benchmark_data


def export_benchmark_markdown(output_path: Path | None = None) -> Path:
    """
    Ekspor tabel komparasi benchmark ke file Markdown.

    File disimpan ke `reports/benchmark_results.md` (relatif ke direktori
    kerja saat ini) atau ke path yang ditentukan secara eksplisit.

    Args:
        output_path: Path tujuan file Markdown. Default: reports/benchmark_results.md.

    Returns:
        Path objek dari file yang berhasil ditulis.
    """
    if output_path is None:
        output_path = Path("reports") / "benchmark_results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _collect_benchmark_data()

    lines: list[str] = []
    lines.append("# Hasil Benchmark: UCS vs A*\n")
    lines.append(
        "Tabel komparasi metrik kinerja algoritma UCS dan A* "
        "pada jaringan jalan perkotaan (CERTAN-01 Milestone 1).\n"
    )
    lines.append("")
    lines.append(
        "| Skenario | Algoritma | Biaya Optimal (Rp) | "
        "Jarak (km) | Simpul Dijelajahi | Runtime (ms) |"
    )
    lines.append(
        "| :--- | :---: | ---: | ---: | ---: | ---: |"
    )

    for label, ucs, astar in data:
        for result in (ucs, astar):
            algo = "UCS" if "UCS" in result.algorithm else "A\\*"
            cost = f"Rp {result.total_cost:,.0f}" if result.found else "N/A"
            dist = f"{result.total_distance:.2f}"   if result.found else "N/A"
            nodes = str(result.nodes_expanded)
            rt   = f"{result.runtime_ms:.4f}"
            lines.append(
                f"| {label} | {algo} | {cost} | {dist} | {nodes} | {rt} |"
            )
            label = ""   # Tampilkan label hanya pada baris pertama per skenario

    lines.append("")
    lines.append(
        "> **Keterangan**: *Simpul Dijelajahi* = jumlah simpul yang dieksplorasi "
        "(*nodes expanded*) hingga solusi ditemukan."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_benchmark_csv(output_path: Path | None = None) -> Path:
    """
    Ekspor tabel komparasi benchmark ke file CSV.

    File disimpan ke `reports/benchmark_results.csv` (relatif ke direktori
    kerja saat ini) atau ke path yang ditentukan secara eksplisit.

    Kolom CSV:
      Skenario, Algoritma, Biaya_Rp, Jarak_km,
      Simpul_Dijelajahi, Runtime_ms, Rute

    Args:
        output_path: Path tujuan file CSV. Default: reports/benchmark_results.csv.

    Returns:
        Path objek dari file yang berhasil ditulis.
    """
    if output_path is None:
        output_path = Path("reports") / "benchmark_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _collect_benchmark_data()

    HEADERS = [
        "Skenario",
        "Algoritma",
        "Biaya_Rp",
        "Jarak_km",
        "Simpul_Dijelajahi",
        "Runtime_ms",
        "Rute",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for label, ucs, astar in data:
            for result in (ucs, astar):
                algo = "UCS" if "UCS" in result.algorithm else "A*"
                writer.writerow({
                    "Skenario"         : label,
                    "Algoritma"        : algo,
                    "Biaya_Rp"         : f"{result.total_cost:.2f}" if result.found else "",
                    "Jarak_km"         : f"{result.total_distance:.2f}" if result.found else "",
                    "Simpul_Dijelajahi": result.nodes_expanded,
                    "Runtime_ms"       : f"{result.runtime_ms:.4f}",
                    "Rute"             : " → ".join(result.path_names) if result.found else "",
                })

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Titik masuk utama program CLI.

    Argumen baris perintah:
      --demo       : Mode demo otomatis (semua skenario).
      --bench      : Mode benchmark saja (hanya tabel metrik).
      --export-md  : Ekspor tabel benchmark ke Markdown (reports/benchmark_results.md).
      --export-csv : Ekspor tabel benchmark ke CSV (reports/benchmark_results.csv).
      --visualize  : Generate seluruh gambar visualisasi PNG ke reports/figures/.
      (kosong)     : Mode interaktif kustom.

    Catatan: --export-md, --export-csv, dan --visualize dapat dikombinasikan sekaligus.
    """
    # Pastikan stdout menggunakan UTF-8 agar output tampil benar di Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = sys.argv[1:]

    # Tangani ekspor terlebih dahulu (bisa dikombinasikan dengan mode lain)
    exported_any = False
    if "--export-md" in args:
        out_path = export_benchmark_markdown()
        print(f"[OK] Benchmark Markdown diekspor ke: {out_path.resolve()}")
        exported_any = True
    if "--export-csv" in args:
        out_path = export_benchmark_csv()
        print(f"[OK] Benchmark CSV diekspor ke: {out_path.resolve()}")
        exported_any = True
    if "--visualize" in args:
        generate_all_figures()
        exported_any = True

    # Jika hanya ekspor/visualize (tanpa flag mode), selesai di sini
    if exported_any and not any(f in args for f in ("--demo", "--bench")):
        return

    if "--help" in args or "-h" in args:
        print("""
Penggunaan:
  uv run certan-01 [OPSI]
  uv run certan-route [OPSI]

Opsi:
  --demo            Jalankan demonstrasi otomatis pada 4 skenario kurir
  --bench, --benchmark
                    Jalankan pengujian benchmark metrik komparasi UCS vs A*
  --export-md       Ekspor tabel benchmark ke reports/benchmark_results.md
  --export-csv      Ekspor tabel benchmark ke reports/benchmark_results.csv
  --visualize       Hasilkan semua grafik visualisasi PNG ke reports/figures/
  -h, --help        Tampilkan pesan bantuan ini
  (tanpa opsi)      Masuk ke mode interaktif pemilihan rute kurir
""")
        return

    if "--demo" in args:
        run_demo_mode()
    elif "--bench" in args or "--benchmark" in args:
        run_benchmark_mode()
    elif not exported_any:
        run_custom_mode()
