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

from certan_01.data import (
    build_laguboti_road_network,
    build_urban_road_network,
    list_available_scenarios,
    list_laguboti_scenarios,
)
from certan_01.heuristics import euclidean_heuristic
from certan_01.models import Graph
from certan_01.search import SearchResult, astar_search, uniform_cost_search
from certan_01.visualize import generate_all_figures, generate_laguboti_folium_map


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
# Mode Laguboti: Skenario Nyata Berbasis Data OSM
# ---------------------------------------------------------------------------

def run_laguboti_mode() -> None:
    """
    Jalankan 4 skenario pengiriman paket nyata di Kecamatan Laguboti.
    Output menampilkan:
      1. Profil kendaraan kurir yang digunakan.
      2. Perbandingan 2 strategi rute: hemat BBM (A*) vs jarak terpendek.
      3. Satu kotak REKOMENDASI akhir yang jelas.
      4. Catatan akademis: efisiensi ekspansi simpul A* vs UCS.
    """
    from certan_01.osm_loader import (
        get_vehicle_label,
        load_laguboti_graph_distance_only,
    )

    print(BANNER)
    print("=" * 64)
    print("  MODE LAGUBOTI: Optimasi Rute Nyata Kecamatan Laguboti")
    print("  Data: OpenStreetMap | Model BBM: Pertamina (2026-04-01)")
    print("=" * 64)

    scenarios = list_laguboti_scenarios()
    html_files = []

    for idx, sc in enumerate(scenarios, start=1):
        _print_separator("=")
        print(f"\n[{idx}/{len(scenarios)}] {sc['label']}")
        print(f"  {sc['description']}")
        _print_separator()

        # ── Info Kendaraan ──────────────────────────────────────
        vehicle_label = get_vehicle_label(sc["vehicle"])
        print(f"\n  Kendaraan  : {vehicle_label}")

        # ── Load Graf BBM-Optimal ────────────────────────────────
        print("  Memuat data jaringan jalan Laguboti dari OSM...")
        try:
            graph_fuel = build_laguboti_road_network(
                hub_landmark=sc["hub_landmark"],
                goal_landmark=sc["goal_landmark"],
                vehicle=sc["vehicle"],
            )
            graph_dist = load_laguboti_graph_distance_only(
                hub_landmark=sc["hub_landmark"],
                goal_landmark=sc["goal_landmark"],
            )
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            continue

        hub  = graph_fuel.hub_node()
        goal = graph_fuel.goal_node()
        if hub is None or goal is None:
            print("\n[PERINGATAN] Simpul hub/goal tidak ditemukan. Lewati skenario ini.")
            continue

        print(f"  Asal       : {hub.name}")
        print(f"  Tujuan     : {goal.name}")
        print(f"  Jaringan   : {graph_fuel.summary()}")

        # ── Heuristik ────────────────────────────────────────────
        def heuristic(node, g=goal):
            return euclidean_heuristic(node, g)

        # ── Pencarian ────────────────────────────────────────────
        # Rute 1: Paling Hemat BBM — A* (bobot biaya bensin)
        astar_fuel = astar_search(graph_fuel, hub.node_id, goal.node_id, heuristic)
        # Rute 2: Jarak Terpendek — UCS (bobot jarak km murni)
        ucs_dist   = uniform_cost_search(graph_dist, hub.node_id, goal.node_id)
        # Rute 3: UCS berbasis BBM (untuk perbandingan akademis A* vs UCS)
        ucs_fuel   = uniform_cost_search(graph_fuel, hub.node_id, goal.node_id)

        # ── Hitung biaya bensin rute terpendek di graf BBM ───────
        # (jarak terpendek ≠ rute teririt, hitung biaya BBM riil-nya)
        dist_route_fuel_cost = 0.0
        if ucs_dist.found:
            for i in range(len(ucs_dist.path_ids) - 1):
                nfrom = ucs_dist.path_ids[i]
                nto   = ucs_dist.path_ids[i + 1]
                for edge in graph_fuel.get_neighbors(nfrom):
                    if edge.to_node.node_id == nto:
                        dist_route_fuel_cost += edge.total_cost
                        break

        # ── Tampilan Perbandingan ────────────────────────────────
        _print_separator("=")
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │         PERBANDINGAN STRATEGI RUTE KURIR            │")
        print("  └─────────────────────────────────────────────────────┘")

        print(f"\n  {'Strategi':<30} {'Jarak':>8}  {'Biaya BBM':>12}  {'Simpul':>8}")
        _print_separator("-", 64)

        if astar_fuel.found:
            print(f"  {'[A*]  Paling Hemat BBM':<30} "
                  f"{astar_fuel.total_distance:>6.2f} km"
                  f"  {'Rp ' + f'{astar_fuel.total_cost:,.0f}':>12}"
                  f"  {astar_fuel.nodes_expanded:>6} eks")

        if ucs_dist.found:
            print(f"  {'[UCS] Jarak Terpendek':<30} "
                  f"{ucs_dist.total_distance:>6.2f} km"
                  f"  {'Rp ' + f'{dist_route_fuel_cost:,.0f}':>12}"
                  f"  {ucs_dist.nodes_expanded:>6} eks")

        _print_separator("-", 64)

        # ── Hitung Penghematan ───────────────────────────────────
        savings_rp   = dist_route_fuel_cost - astar_fuel.total_cost
        savings_pct  = (savings_rp / dist_route_fuel_cost * 100) if dist_route_fuel_cost > 0 else 0
        extra_km     = astar_fuel.total_distance - ucs_dist.total_distance
        same_route   = abs(extra_km) < 0.01 and abs(savings_rp) < 1

        # ── REKOMENDASI ──────────────────────────────────────────
        print()
        print("  ╔════════════════════════════════════════════════════════╗")
        print("  ║                  REKOMENDASI SISTEM                   ║")
        print("  ╠════════════════════════════════════════════════════════╣")
        if astar_fuel.found:
            print(f"  ║  Gunakan Rute A* (Paling Hemat BBM):               ║")
            print(f"  ║  • Jarak    : {astar_fuel.total_distance:.2f} km{' ' * (40 - len(f'{astar_fuel.total_distance:.2f} km'))}║")
            print(f"  ║  • Biaya    : Rp {astar_fuel.total_cost:,.0f}{' ' * (38 - len(f'Rp {astar_fuel.total_cost:,.0f}'))}║")
            if same_route:
                print(f"  ║  • Catatan  : Rute terpendek = rute teririt BBM!   ║")
            elif savings_rp > 0:
                print(f"  ║  • Hemat    : Rp {savings_rp:,.0f} ({savings_pct:.1f}%) vs jarak terpendek{' ' * max(0, 14 - len(f'Rp {savings_rp:,.0f} ({savings_pct:.1f}%)'))}║")
                print(f"  ║  • Tradeoff : +{extra_km:.2f} km lebih jauh, tapi lebih irit BBM   ║")
        print("  ╚════════════════════════════════════════════════════════╝")

        # ── Catatan Akademis (Efisiensi A* vs UCS) ──────────────
        if ucs_fuel.found and astar_fuel.found and ucs_fuel.nodes_expanded > 0:
            node_saved = ucs_fuel.nodes_expanded - astar_fuel.nodes_expanded
            node_pct   = node_saved / ucs_fuel.nodes_expanded * 100
            print(f"\n  [Akademis] A* ekspansi {astar_fuel.nodes_expanded} simpul "
                  f"vs UCS {ucs_fuel.nodes_expanded} simpul "
                  f"(hemat {node_pct:.1f}% eksplorasi)")

        # ── Peta Interaktif ──────────────────────────────────────
        print(f"\n  Menghasilkan peta interaktif HTML...")
        try:
            html_path = generate_laguboti_folium_map(
                graph=graph_fuel,
                ucs_result=ucs_fuel,
                astar_result=astar_fuel,
                scenario_name=sc["label"],
                vehicle_label=vehicle_label,
                dist_result=ucs_dist,
                dist_fuel_cost=dist_route_fuel_cost,
            )
            html_files.append(html_path)
            print(f"  [OK] Peta: {html_path}")
        except Exception as e:
            print(f"  [PERINGATAN] Gagal buat peta: {e}")

    # Ringkasan akhir
    _print_separator("=")
    print(f"\n[SELESAI] {len(scenarios)} skenario Laguboti dijalankan.")
    if html_files:
        print("  Peta interaktif tersimpan di:")
        for f in html_files:
            print(f"    -> {f}")
    print()


def generate_lagubiti_folium_map_safe(graph, ucs_result, astar_result, scenario_name):
    """Wrapper aman untuk generate_laguboti_folium_map (backward compat)."""
    return generate_laguboti_folium_map(
        graph=graph,
        ucs_result=ucs_result,
        astar_result=astar_result,
        scenario_name=scenario_name,
    )



from certan_01.chatbot import run_chatbot_cli


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:

    """
    Titik masuk utama program CLI.

    Argumen baris perintah:
      --chat       : Jalankan mode percakapan Chatbot Asisten Kurir (Default).
      --laguboti   : Jalankan 4 skenario rute nyata Laguboti (OSM + BBM Pertamina).
      --demo       : Mode demo otomatis (semua skenario perkotaan).
      --bench      : Mode benchmark saja (hanya tabel metrik).
      --export-md  : Ekspor tabel benchmark ke Markdown (reports/benchmark_results.md).
      --export-csv : Ekspor tabel benchmark ke CSV (reports/benchmark_results.csv).
      --visualize  : Generate seluruh gambar visualisasi PNG ke reports/figures/.
      --custom     : Mode interaktif kustom manual 12 simpul lama.
      (kosong)     : Langsung masuk ke Chatbot Asisten Kurir.
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
    if exported_any and not any(f in args for f in ("--demo", "--bench", "--laguboti", "--chat")):
        return

    if "--help" in args or "-h" in args:
        print("""
Penggunaan:
  uv run certan-01 [OPSI]
  uv run certan-route [OPSI]

Opsi Utama:
  (tanpa opsi)      Jalankan Chatbot Asisten Pintar Kurir (Mode Percakapan Interaktif)
  --chat            Jalankan Chatbot Asisten Pintar Kurir
  --laguboti        Jalankan 4 skenario rute nyata Laguboti (OSM + BBM Pertamina)
  --demo            Jalankan demonstrasi otomatis pada 4 skenario kurir
  --bench, --benchmark
                    Jalankan pengujian benchmark metrik komparasi UCS vs A*
  --export-md       Ekspor tabel benchmark ke reports/benchmark_results.md
  --export-csv      Ekspor tabel benchmark ke reports/benchmark_results.csv
  --visualize       Hasilkan semua grafik visualisasi PNG ke reports/figures/
  --custom          Mode interaktif manual 12 simpul perkotaan (Legacy)
  -h, --help        Tampilkan pesan bantuan ini
""")
        return

    if "--demo" in args:
        run_demo_mode()
    elif "--bench" in args or "--benchmark" in args:
        run_benchmark_mode()
    elif "--laguboti" in args:
        run_laguboti_mode()
    elif "--custom" in args:
        run_custom_mode()
    elif "--chat" in args or not exported_any:
        run_chatbot_cli()
