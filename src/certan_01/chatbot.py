"""
chatbot.py
----------
Modul Chatbot Pintar Asisten Kurir (Courier AI Assistant).
Menyediakan antarmuka berbasis percakapan (conversational agent) untuk membantu
kurir menentukan jenis kendaraan, lokasi asal dan tujuan, serta mendapatkan
rekomendasi rute pengiriman paket yang paling hemat bahan bakar (BBM).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from certan_01.data import build_laguboti_road_network
from certan_01.heuristics import euclidean_heuristic
from certan_01.models import Graph
from certan_01.orchestrator import (
    DispatchPlan,
    _map_vehicle_profile,
    dispatch_and_route,
)
from certan_01.osm_loader import (
    LANDMARK_LABELS,
    LANDMARK_SYNONYMS,
    LANDMARKS,
    check_ambiguous_landmark,
    get_available_vehicles,
    get_vehicle_label,
    load_laguboti_graph_distance_only,
    parse_coordinates,
    resolve_landmark_input,
    resolve_vehicle_input,
)
from certan_01.search import SearchResult, astar_search, uniform_cost_search
from certan_01.solver import (
    Courier,
    Package,
    create_default_courier_fleet,
    load_real_ecommerce_packages,
)
from certan_01.visualize import generate_laguboti_folium_map


# ---------------------------------------------------------------------------
# Konstanta & Tampilan Chatbot
# ---------------------------------------------------------------------------

BOT_NAME = "KurirBot"

CHAT_BANNER = """
==============================================================
  ASISTEN RUTE KURIR (CERTAN-01)
  Navigasi Cerdas & Rute Hemat BBM | Wilayah Toba
==============================================================
"""

VEHICLE_MENU_PROMPT = """Pilih jenis kendaraan kurir:
  [1] Motor Matik (Honda BeAT / Vario)
  [2] Motor Bebek (Honda Revo / Supra)
  [3] Motor Sport (150cc)
  [4] Mobil Pick-Up (GranMax / Box Kecil)
  [5] Mobil Box Diesel (Mitsubishi L300)

Ketik nomor [1-5] atau nama kendaraan:"""

LANDMARK_LIST_PROMPT = """Area & Tempat di Koridor Toba:
  - Balige    : Mixue, Indomaret, Alfamart, Pasar Onan, RS HKBP, Hotel Labersa, Pantai Bulbul, SPBU
  - Laguboti  : IT Del Sitoluama, SMA Unggul Del, JNE Laguboti, Pasar Laguboti, SPBU, Polsek
  - Porsea    : Pasar Porsea, RSUD Porsea, Dermaga Porsea, Makam Sisingamangaraja
  - Silangit  : Bandara Silangit, Pasar Siborong-borong
  - Tarutung  : Pusat Pasar Tarutung, RSUD Tarutung, Salib Kasih, Kantor HKBP Pearaja
  - Ruas Jalan: Jl. Gereja, Jl. Sisingamangaraja, Jl. By Pass, Jl. Jambu, Jl. Hutabulu Mejan, dll.

Tips: Ketik nama tempat langsung (misal: 'mixue', 'it del', 'jl gereja') atau 'cari <nama>'."""



def _format_location_label(loc: str | None) -> str:
    """Format label lokasi ramah pengguna baik untuk landmark maupun koordinat GPS."""
    if not loc:
        return ""
    if loc in LANDMARK_LABELS:
        return LANDMARK_LABELS[loc]
    parsed = parse_coordinates(loc)
    if parsed:
        return f"Koordinat GPS ({parsed[0]:.4f}, {parsed[1]:.4f})"
    return loc


@dataclass
class ChatbotState:
    """Menyimpan state percakapan kurir."""
    vehicle: str | None = None
    origin: str | None = None
    destination: str | None = None
    last_result: dict[str, Any] | None = None
    last_dispatch_plan: DispatchPlan | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    pending_clarification: dict[str, Any] | None = None


class CourierChatbot:
    """
    Mesin Chatbot percakapan asisten kurir untuk penentuan rute hemat BBM.
    Mendukung dialog bertahap (step-by-step) maupun pengenalan kalimat langsung
    (one-shot slot filling).
    """

    def __init__(self) -> None:
        self.state = ChatbotState()

    def reset(self) -> None:
        """Reset state percakapan ke kondisi awal."""
        self.state = ChatbotState()

    # -----------------------------------------------------------------------
    # Slot Extraction (Ekstraksi Entitas dari Teks)
    # -----------------------------------------------------------------------

    def extract_vehicle(self, text: str) -> str | None:
        """Ekstraksi entitas kendaraan dari teks input."""
        return resolve_vehicle_input(text)

    def extract_landmarks(self, text: str) -> tuple[str | None, str | None]:
        """
        Ekstraksi asal dan tujuan dari teks.
        Mengenali pola umum bahasa Indonesia:
          - "dari X ke Y"
          - "antar dari X ke Y"
          - "ke Y dari X"
          - atau penyebutan dua landmark dalam satu kalimat
        """
        clean_text = text.lower()
        origin: str | None = None
        destination: str | None = None

        # Pola 1: "dari ... ke ..."
        match_dari_ke = re.search(r"dari\s+(.+?)\s+ke\s+(.+)", clean_text)
        if match_dari_ke:
            raw_orig = match_dari_ke.group(1).strip()
            raw_dest = match_dari_ke.group(2).strip()
            # Bersihkan kata sambung kendaraan di akhir (misal "ke it del pake motor")
            raw_dest = re.split(r"\s+(pake|pakai|dengan|naik|menggunakan)\s+", raw_dest)[0]
            origin = resolve_landmark_input(raw_orig)
            destination = resolve_landmark_input(raw_dest)
            if origin or destination:
                return origin, destination

        # Pola 2: "ke ... dari ..."
        match_ke_dari = re.search(r"ke\s+(.+?)\s+dari\s+(.+)", clean_text)
        if match_ke_dari:
            raw_dest = match_ke_dari.group(1).strip()
            raw_orig = match_ke_dari.group(2).strip()
            raw_orig = re.split(r"\s+(pake|pakai|dengan|naik|menggunakan)\s+", raw_orig)[0]
            destination = resolve_landmark_input(raw_dest)
            origin = resolve_landmark_input(raw_orig)
            if origin or destination:
                return origin, destination

        # Pola 3: Deteksi landmark individual yang muncul dalam kalimat
        found_spanned: list[tuple[int, str]] = []
        matched_spans: list[tuple[int, int]] = []

        all_kw_pairs: list[tuple[str, str]] = []
        for lkey, kws in LANDMARK_SYNONYMS.items():
            for kw in kws:
                all_kw_pairs.append((kw, lkey))
        all_kw_pairs.sort(key=lambda x: len(x[0]), reverse=True)

        for kw, key in all_kw_pairs:
            # Multi-kata atau mengandung titik (misal "jl. sisingamangaraja"): substring match
            if " " in kw or "." in kw:
                idx = clean_text.find(kw)
                if idx == -1:
                    continue
                span = (idx, idx + len(kw))
            else:
                pattern = r"\b" + re.escape(kw) + r"\b"
                m = re.search(pattern, clean_text)
                if not m:
                    continue
                span = m.span()

            overlap = any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in matched_spans)
            if not overlap:
                matched_spans.append(span)
                if not any(k == key for _, k in found_spanned):
                    found_spanned.append((span[0], key))

        # Urutkan berdasarkan posisi kemunculan dalam kalimat pengguna
        found_spanned.sort(key=lambda x: x[0])
        found_landmarks = [k for _, k in found_spanned]

        if len(found_landmarks) >= 2:
            return found_landmarks[0], found_landmarks[1]
        elif len(found_landmarks) == 1:
            if "dari " in clean_text:
                return found_landmarks[0], None
            elif "ke " in clean_text or "tujuan" in clean_text:
                return None, found_landmarks[0]
            else:
                return found_landmarks[0], None

        return None, None

    # -----------------------------------------------------------------------
    # Solver / Engine Integration
    # -----------------------------------------------------------------------


    def solve_route(self, vehicle: str, origin: str, destination: str) -> dict[str, Any]:
        """
        Jalankan A* dan UCS untuk menghitung rute kurir pada jaringan Laguboti.
        """
        graph_fuel = build_laguboti_road_network(
            hub_landmark=origin,
            goal_landmark=destination,
            vehicle=vehicle,
        )
        graph_dist = load_laguboti_graph_distance_only(
            hub_landmark=origin,
            goal_landmark=destination,
        )

        hub = graph_fuel.hub_node()
        goal = graph_fuel.goal_node()

        if hub is None or goal is None:
            raise ValueError("Simpul hub atau goal tidak ditemukan pada jaringan jalan.")

        min_cost = graph_fuel.min_cost_per_km
        def heuristic(node, g=goal):
            return euclidean_heuristic(node, g, c_min_per_km=min_cost)

        # 1. Rute Paling Hemat BBM via A*
        astar_fuel = astar_search(graph_fuel, hub.node_id, goal.node_id, heuristic)

        # 2. Rute Jarak Terpendek via UCS
        ucs_dist = uniform_cost_search(graph_dist, hub.node_id, goal.node_id)

        # 3. UCS BBM (untuk evaluasi pemangkasan simpul)
        ucs_fuel = uniform_cost_search(graph_fuel, hub.node_id, goal.node_id)

        # Hitung biaya bensin jika lewat rute terpendek
        dist_fuel_cost = 0.0
        if ucs_dist.found:
            for i in range(len(ucs_dist.path_ids) - 1):
                nfrom = ucs_dist.path_ids[i]
                nto = ucs_dist.path_ids[i + 1]
                for edge in graph_fuel.get_neighbors(nfrom):
                    if edge.to_node.node_id == nto:
                        dist_fuel_cost += edge.total_cost
                        break

        # Generate Peta Folium
        vehicle_label = get_vehicle_label(vehicle)
        html_map_path = None
        try:
            html_map_path = generate_laguboti_folium_map(
                graph=graph_fuel,
                ucs_result=ucs_fuel,
                astar_result=astar_fuel,
                scenario_name=f"chat_{origin}_to_{destination}",
                vehicle_label=vehicle_label,
                dist_result=ucs_dist,
                dist_fuel_cost=dist_fuel_cost,
            )
        except Exception:
            pass

        # Koordinat asal dan tujuan
        if origin in LANDMARKS:
            orig_coords = LANDMARKS[origin]
            orig_label = LANDMARK_LABELS.get(origin, origin)
        else:
            parsed = parse_coordinates(origin)
            if parsed:
                orig_coords = parsed
                orig_label = f"Koordinat GPS ({parsed[0]:.4f}, {parsed[1]:.4f})"
            else:
                orig_coords = (2.3667, 99.1247)
                orig_label = origin

        if destination in LANDMARKS:
            dest_coords = LANDMARKS[destination]
            dest_label = LANDMARK_LABELS.get(destination, destination)
        else:
            parsed = parse_coordinates(destination)
            if parsed:
                dest_coords = parsed
                dest_label = f"Koordinat GPS ({parsed[0]:.4f}, {parsed[1]:.4f})"
            else:
                dest_coords = (2.3832, 99.1486)
                dest_label = destination

        savings_rp = dist_fuel_cost - astar_fuel.total_cost
        savings_pct = (savings_rp / dist_fuel_cost * 100) if dist_fuel_cost > 0 else 0.0
        node_saved = ucs_fuel.nodes_expanded - astar_fuel.nodes_expanded
        node_saved_pct = (node_saved / ucs_fuel.nodes_expanded * 100) if ucs_fuel.nodes_expanded > 0 else 0.0

        return {
            "vehicle": vehicle,
            "vehicle_label": vehicle_label,
            "origin": origin,
            "origin_label": orig_label,
            "destination": destination,
            "destination_label": dest_label,
            "astar_fuel": astar_fuel,
            "ucs_dist": ucs_dist,
            "ucs_fuel": ucs_fuel,
            "dist_fuel_cost": dist_fuel_cost,
            "savings_rp": savings_rp,
            "savings_pct": savings_pct,
            "node_saved": node_saved,
            "node_saved_pct": node_saved_pct,
            "map_path": html_map_path,
        }

    # -----------------------------------------------------------------------
    # Response Formatter
    # -----------------------------------------------------------------------

    def format_solution_response(self, res: dict[str, Any]) -> str:
        """Format hasil kalkulasi optimasi rute ke dalam pesan balasan percakapan yang bersih dan profesional."""
        astar: SearchResult = res["astar_fuel"]
        ucs: SearchResult = res["ucs_dist"]
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("RUTE PENGIRIMAN PAKET")
        lines.append("=" * 60)
        lines.append(f"Kendaraan    : {res['vehicle_label']}")
        lines.append(f"Titik Jemput : {res['origin_label']}")
        lines.append(f"Tujuan Paket : {res['destination_label']}")
        lines.append("-" * 60)

        if not astar.found:
            lines.append("Tidak ditemukan jalur jalan yang menghubungkan kedua lokasi ini.")
            lines.append("=" * 60)
            return "\n".join(lines)

        lines.append("ROUTE HEMAT BBM (Direkomendasikan):")
        lines.append(f"- Estimasi BBM : Rp {astar.total_cost:,.0f}")
        lines.append(f"- Jarak Tempuh : {astar.total_distance:.2f} km")

        if astar.path_names:
            if len(astar.path_names) > 3:
                path_str = f"{astar.path_names[0]} -> {astar.path_names[1]} -> ... -> {astar.path_names[-1]}"
            else:
                path_str = " -> ".join(astar.path_names)
            lines.append(f"- Jalur Utama  : {path_str}")

        lines.append("\nPERBANDINGAN DENGAN RUTE TERPENDEK:")
        lines.append(f"- Jarak        : {ucs.total_distance:.2f} km (BBM: Rp {res['dist_fuel_cost']:,.0f})")

        if res["savings_rp"] > 10:
            lines.append(
                f"- Hemat BBM    : Rp {res['savings_rp']:,.0f} ({res['savings_pct']:.1f}%) "
                f"lebih hemat dibanding rute terpendek."
            )
        else:
            lines.append(
                "- Keterangan   : Rute ini sudah yang paling hemat BBM."
            )

        if res.get("map_path"):
            map_file = res["map_path"]
            lines.append("\nPETA RUTE INTERAKTIF (FOLIUM & OPENSTREETMAP):")
            lines.append(f"- File HTML    : {map_file}")
            lines.append("  (Buka file ini di browser untuk melihat visualisasi peta & landmark)")

        lines.append("=" * 60)
        lines.append("Ketik 'antar lagi' untuk rute baru, 'lokasi' untuk daftar tempat, atau 'keluar'.")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Modul Integrasi Dispatch & Alokasi Armada (M2 CSP -> M1 A*)
    # -----------------------------------------------------------------------

    def handle_dispatch_command(self, limit: int = 5) -> str:
        """
        Jalankan alur terintegrasi M2 (CSP Alokasi) -> M1 (A* Routing)
        dan kembalikan ringkasan rencana dispatch untuk seluruh armada.
        """
        try:
            packages = load_real_ecommerce_packages(limit=limit)
            fleet = create_default_courier_fleet(fleet_size=3)
            hub = "hub_jne_laguboti"

            plan = dispatch_and_route(
                packages=packages,
                couriers=fleet,
                hub_landmark=hub,
                verbose=False,
            )
            self.state.last_dispatch_plan = plan

            if not plan.is_feasible:
                return (
                    f"[ALOKASI TIDAK BERHASIL]\n"
                    f"Tidak dapat menemukan alokasi legal untuk {len(packages)} paket.\n"
                    f"Alasan: {plan.infeasible_reason}"
                )

            lines: list[str] = []
            lines.append("RENCANA DISPATCH & ALOKASI ARMADA (M2 CSP -> M1 A*)")
            lines.append("=" * 60)
            lines.append("Depot Asal    : JNE Hub Laguboti")
            lines.append(f"Total Paket   : {len(packages)} paket e-commerce")
            lines.append("Status Batasan: 100% FEASIBLE (Semua batasan terpenuhi)")
            lines.append(f"Total BBM     : Rp {plan.total_fleet_fuel_cost:,.0f} | Jarak: {plan.total_fleet_distance:.2f} km")
            lines.append("-" * 60)

            for cr in plan.courier_routes:
                if not cr.packages:
                    continue
                lines.append(f"\n[{cr.courier.courier_id}] {cr.courier.name} ({cr.courier.vehicle_type}):")
                lines.append(f"   Beban: {cr.courier_load:.1f}/{cr.courier.max_capacity_kg:.0f} kg | Waktu: {cr.courier_time:.1f}/{cr.courier.max_work_hours:.1f} jam")
                lines.append(f"   BBM: Rp {cr.total_fuel_cost:,.0f} ({cr.total_distance:.2f} km)")
                for pkg in cr.packages:
                    res = cr.routes.get(pkg.package_id)
                    dest_str = _format_location_label(pkg.destination_node) if isinstance(pkg.destination_node, str) else f"GPS ({pkg.destination_node[0]:.4f}, {pkg.destination_node[1]:.4f})"
                    bulky_str = " [BULKY]" if pkg.is_bulky else ""
                    if res and res.found:
                        lines.append(f"   • {pkg.package_id} ({pkg.weight_kg} kg{bulky_str}) -> {dest_str}")
                        lines.append(f"     BBM: Rp {res.total_cost:,.0f} ({res.total_distance:.2f} km)")
                    else:
                        lines.append(f"   • {pkg.package_id} ({pkg.weight_kg} kg{bulky_str}) -> {dest_str}")

            lines.append("\n" + "=" * 60)
            lines.append("Petunjuk:")
            lines.append("  • Ketik 'antar <id>' (misal: 'antar P01') untuk langsung rute paket.")
            lines.append("  • Ketik 'paket saya' untuk melihat paket khusus kendaraanmu.")
            lines.append("  • Ketik 'antar lagi' untuk rute baru.")

            return "\n".join(lines)

        except Exception as exc:
            return f"Gagal memproses dispatch: {exc}"

    def handle_my_packages_command(self) -> str:
        """Tampilkan daftar paket yang dialokasikan khusus untuk kendaraan pengguna saat ini."""
        if not self.state.last_dispatch_plan:
            self.handle_dispatch_command(limit=5)

        plan = self.state.last_dispatch_plan
        if not plan or not plan.is_feasible:
            return "Belum ada rencana dispatch aktif. Ketik 'alokasi' untuk membuat rencana alokasi paket hari ini."

        user_veh = self.state.vehicle
        if not user_veh:
            return (
                "Kamu belum memilih kendaraan hari ini.\n"
                "Pilih kendaraanmu terlebih dahulu:\n\n"
                f"{VEHICLE_MENU_PROMPT}"
            )

        matched_cr = None
        for cr in plan.courier_routes:
            veh_profile = _map_vehicle_profile(cr.courier.vehicle_type)
            if veh_profile == user_veh:
                matched_cr = cr
                break

        if not matched_cr or not matched_cr.packages:
            matched_cr = next((cr for cr in plan.courier_routes if cr.packages), None)
            if not matched_cr:
                return "Tidak ada paket yang ditugaskan ke armada saat ini."

        lines: list[str] = []
        lines.append(f"DAFTAR TUGAS PAKET UNTUK KENDARAANMU ({get_vehicle_label(user_veh)})")
        lines.append("=" * 60)
        lines.append(f"Kurir Penugasan : {matched_cr.courier.name} [{matched_cr.courier.courier_id}]")
        pct_load = (matched_cr.courier_load / matched_cr.courier.max_capacity_kg * 100) if matched_cr.courier.max_capacity_kg > 0 else 0
        lines.append(f"Beban Angkut    : {matched_cr.courier_load:.1f} / {matched_cr.courier.max_capacity_kg:.0f} kg ({pct_load:.1f}%)")
        lines.append(f"Estimasi Jam    : {matched_cr.courier_time:.1f} / {matched_cr.courier.max_work_hours:.1f} jam")
        lines.append(f"Total Biaya BBM : Rp {matched_cr.total_fuel_cost:,.0f} (Jarak: {matched_cr.total_distance:.2f} km)")
        lines.append("-" * 60)
        lines.append("Daftar Paket yang Harus Diantar:")

        for i, pkg in enumerate(matched_cr.packages, 1):
            dest_str = _format_location_label(pkg.destination_node) if isinstance(pkg.destination_node, str) else f"GPS ({pkg.destination_node[0]:.4f}, {pkg.destination_node[1]:.4f})"
            res = matched_cr.routes.get(pkg.package_id)
            cost_str = f"Rp {res.total_cost:,.0f}" if res and res.found else "N/A"
            dist_str = f"{res.total_distance:.2f} km" if res and res.found else "N/A"
            bulky_tag = " [BULKY]" if pkg.is_bulky else ""
            lines.append(f"  {i}. [{pkg.package_id}] {dest_str} ({pkg.weight_kg} kg{bulky_tag})")
            lines.append(f"     BBM: {cost_str} | Jarak: {dist_str}")

        lines.append("\n" + "=" * 60)
        first_pkg_id = matched_cr.packages[0].package_id
        lines.append(f"Ketik 'antar {first_pkg_id}' untuk memulai rute ke paket pertama.")
        return "\n".join(lines)


    # -----------------------------------------------------------------------
    # Main Dialog Handler
    # -----------------------------------------------------------------------

    def handle_message(self, user_message: str) -> str:
        """
        Proses satu putaran pesan teks dari pengguna dan kembalikan balasan bot.
        """
        raw = user_message.strip()
        text = raw.lower()

        # Simpan ke riwayat
        self.state.history.append(("user", raw))

        # Perintah Khusus
        if text in ("keluar", "exit", "quit", "selesai", "bye", "dadah"):
            return "Sampai jumpa! Selamat bertugas dan hati-hati di jalan."

        if text in ("bantuan", "help", "panduan"):
            return (
                f"BANTUAN ASISTEN KURIR ({BOT_NAME}):\n"
                f"Ketik rute secara langsung atau ikuti pertanyaan bertahap.\n"
                f"  Contoh: 'Antar dari JNE Laguboti ke IT Del naik Vario'\n\n"
                f"Perintah Alokasi & Armada:\n"
                f"  - alokasi          : Rencana pembagian paket seluruh armada (CSP & A*)\n"
                f"  - paket saya       : Daftar tugas paket untuk kendaraan Anda\n"
                f"  - antar <id_paket> : Navigasi langsung ke paket (misal: 'antar P01')\n\n"
                f"Perintah Navigasi:\n"
                f"  - antar lagi       : Cari rute baru (kendaraan tetap)\n"
                f"  - ganti kendaraan  : Ubah kendaraan kurir\n"
                f"  - lokasi           : Lihat daftar tempat dan nama jalan\n"
                f"  - cari <nama>      : Cari lokasi tertentu (misal: 'cari mixue')\n"
                f"  - reset            : Mulai ulang percakapan dari awal\n"
                f"  - keluar           : Tutup aplikasi"
            )

        # -------------------------------------------------------------------
        # Handler Fitur Terintegrasi M2 -> M1: Dispatch & Alokasi
        # -------------------------------------------------------------------

        # 1. Alokasi / Dispatch armada lengkap
        if text in (
            "dispatch", "alokasi", "bagi paket", "jadwal", "manifest",
            "cek paket", "paket hari ini", "semua kurir", "/dispatch",
            "alokasi paket", "rencana dispatch"
        ) or text.startswith("dispatch") or text.startswith("alokasi"):
            # Jika user ketik 'alokasi 10' atau 'dispatch 10'
            parts = text.split()
            limit = 5
            if len(parts) > 1 and parts[1].isdigit():
                limit = int(parts[1])
            return self.handle_dispatch_command(limit=limit)

        # 2. Paket saya (berdasarkan kendaraan pengguna saat ini)
        if text in ("paket saya", "tugas saya", "muatan saya", "paketku", "tugasku", "muatanku"):
            return self.handle_my_packages_command()

        # 3. Navigasi langsung ke paket tertentu dari rencana dispatch (misal: 'antar p01', 'rute p02')
        match_pkg = re.match(r"^(antar|rute|kirim|navigasi)\s+(p\d+)$", text)
        if match_pkg:
            target_pkg_id = match_pkg.group(2).upper()
            if not self.state.last_dispatch_plan:
                self.handle_dispatch_command(limit=5)

            plan = self.state.last_dispatch_plan
            found_pkg = None
            if plan:
                for cr in plan.courier_routes:
                    for pkg in cr.packages:
                        if pkg.package_id.upper() == target_pkg_id:
                            found_pkg = pkg
                            break
                    if found_pkg:
                        break

            if found_pkg:
                if not self.state.vehicle:
                    self.state.vehicle = "motor_matik_kurir"
                self.state.origin = "hub_jne_laguboti"
                self.state.destination = found_pkg.destination_node if isinstance(found_pkg.destination_node, str) else f"{found_pkg.destination_node[0]}, {found_pkg.destination_node[1]}"
                res = self.solve_route(
                    vehicle=self.state.vehicle,
                    origin=self.state.origin,
                    destination=self.state.destination,
                )
                self.state.last_result = res
                dest_name = _format_location_label(self.state.destination)
                prefix = f"Navigasi Langsung untuk Paket {target_pkg_id} ({found_pkg.weight_kg} kg -> {dest_name}):\n\n"
                return prefix + self.format_solution_response(res)
            else:
                return f"Paket '{target_pkg_id}' tidak ditemukan di rencana dispatch. Ketik 'alokasi' untuk melihat daftar paket hari ini."

        if text.startswith("cari "):
            query = text[5:].strip()
            matches = []
            for k, label in LANDMARK_LABELS.items():
                if query in label.lower() or query in k.lower():
                    matches.append(label)
                else:
                    for syn in LANDMARK_SYNONYMS.get(k, []):
                        if query in syn:
                            matches.append(label)
                            break
            if matches:
                matches_str = "\n".join(f"  - {m}" for m in sorted(list(set(matches))))
                return (
                    f"Hasil pencarian untuk '{query}':\n"
                    f"{matches_str}\n\n"
                    f"Ketik salah satu nama di atas atau masukkan koordinat GPS:"
                )
            else:
                return (
                    f"Tidak ditemukan lokasi dengan kata kunci '{query}'.\n"
                    f"Tips: Ketik 'lokasi' untuk melihat daftar area koridor Toba."
                )

        if text in ("lokasi", "alamat", "daftar lokasi", "tempat", "opsi", "pilihan", "list", "daftar", "daftar tempat", "menu lokasi"):
            status_text = ""
            if self.state.origin:
                orig_label = _format_location_label(self.state.origin)
                status_text = f"Titik jemput saat ini: {orig_label}\n\n"
            return (
                f"{LANDMARK_LIST_PROMPT}\n\n"
                f"{status_text}Ketik nama lokasi yang diinginkan atau masukkan koordinat GPS:"
            )

        if "ganti kendaraan" in text or "ubah kendaraan" in text or "ganti motor" in text or "ganti mobil" in text:
            self.state.vehicle = None
            return f"Silakan ganti kendaraanmu terlebih dahulu:\n\n{VEHICLE_MENU_PROMPT}"

        # "antar lagi" / "rute baru" = hanya hapus rute, kendaraan tetap
        if text in ("antar lagi", "rute baru", "ulang"):
            self.state.origin = None
            self.state.destination = None
            self.state.last_result = None
            self.state.pending_clarification = None
            if self.state.vehicle:
                veh_label = get_vehicle_label(self.state.vehicle)
                return (
                    f"Siap antar lagi! Kendaraan masih {veh_label}.\n"
                    f"Masukkan titik penjemputan berikutnya (atau ketik 'ganti kendaraan'):"
                )
            else:
                return f"Tentukan titik jemput:\n\n{VEHICLE_MENU_PROMPT}"

        # "reset" / "mulai lagi" = hapus semua termasuk kendaraan (mulai dari awal)
        if text in ("reset", "mulai lagi", "mulai ulang"):
            self.state.vehicle = None
            self.state.origin = None
            self.state.destination = None
            self.state.last_result = None
            self.state.pending_clarification = None
            return f"Sesi direset penuh. Pilih kendaraan untuk memulai:\n\n{VEHICLE_MENU_PROMPT}"


        # -------------------------------------------------------------------
        # 0. Penanganan Klarifikasi Ambiguitas (misal memilih opsi [1]/[2])
        # -------------------------------------------------------------------
        if self.state.pending_clarification:
            clarify_info = self.state.pending_clarification
            candidates = clarify_info.get("candidates", [])
            slot = clarify_info.get("slot", "destination")
            selected_landmark = None

            if text.isdigit():
                choice_idx = int(text) - 1
                if 0 <= choice_idx < len(candidates):
                    selected_landmark = candidates[choice_idx]
            else:
                resolved = resolve_landmark_input(text)
                if resolved in candidates:
                    selected_landmark = resolved
                else:
                    for cand in candidates:
                        cand_label = LANDMARK_LABELS.get(cand, cand).lower()
                        if text in cand_label or any(syn in text for syn in LANDMARK_SYNONYMS.get(cand, [])):
                            selected_landmark = cand
                            break

            if selected_landmark:
                self.state.pending_clarification = None
                if slot == "origin":
                    self.state.origin = selected_landmark
                else:
                    self.state.destination = selected_landmark
            else:
                options_str = "\n".join(
                    f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                )
                return (
                    f"Pilihan '{raw}' belum jelas.\n"
                    f"Silakan ketik nomor pilihannya [1-{len(candidates)}] atau nama lengkapnya:\n"
                    f"{options_str}"
                )


        # -------------------------------------------------------------------
        # 1. State-Aware Slot Extraction dari Input Pengguna
        # -------------------------------------------------------------------

        # Tahap A: Kendaraan belum ditentukan
        if self.state.vehicle is None:
            detected_veh = self.extract_vehicle(text)
            if detected_veh:
                self.state.vehicle = detected_veh
                # Cek apakah ada juga penentuan asal & tujuan eksplisit (pola kalimat utuh "dari ... ke ...")
                explicit_orig, explicit_dest = self.extract_landmarks(text)
                if explicit_orig:
                    self.state.origin = explicit_orig
                if explicit_dest:
                    self.state.destination = explicit_dest
                elif explicit_orig:
                    ambig_dest = check_ambiguous_landmark(text)
                    if ambig_dest:
                        term, candidates = ambig_dest
                        self.state.pending_clarification = {
                            "slot": "destination",
                            "term": term,
                            "candidates": candidates,
                        }
                        orig_label = _format_location_label(self.state.origin)
                        options_str = "\n".join(
                            f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                        )
                        return (
                            f"Titik jemput: {orig_label}\n\n"
                            f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                            f"{options_str}\n\n"
                            f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik tujuan:"
                        )
            else:
                # Cek apakah pengguna langsung memasukkan lokasi, jalan, atau rute pengiriman tanpa sebut kendaraan
                explicit_orig, explicit_dest = self.extract_landmarks(text)
                single_loc = resolve_landmark_input(text) if (not explicit_orig and not explicit_dest) else None

                if explicit_orig or explicit_dest or single_loc:
                    # Default kendaraan kurir: Sepeda Motor Matik (paling umum)
                    self.state.vehicle = "motor_matik_kurir"
                    if explicit_orig:
                        self.state.origin = explicit_orig
                    if explicit_dest:
                        self.state.destination = explicit_dest
                    elif single_loc:
                        self.state.origin = single_loc

                    # Cek jika tujuan ambigu saat input kalimat utuh
                    if self.state.origin and not self.state.destination:
                        ambig_dest = check_ambiguous_landmark(text)
                        if ambig_dest:
                            term, candidates = ambig_dest
                            self.state.pending_clarification = {
                                "slot": "destination",
                                "term": term,
                                "candidates": candidates,
                            }
                            orig_label = _format_location_label(self.state.origin)
                            options_str = "\n".join(
                                f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                            )
                            return (
                                f"Titik jemput: {orig_label}\n\n"
                                f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                                f"{options_str}\n\n"
                                f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik tujuan:"
                            )

                elif check_ambiguous_landmark(text):
                    term, candidates = check_ambiguous_landmark(text)
                    self.state.vehicle = "motor_matik_kurir"
                    self.state.pending_clarification = {
                        "slot": "origin",
                        "term": term,
                        "candidates": candidates,
                    }
                    options_str = "\n".join(
                        f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                    )
                    return (
                        f"Lokasi '{term}' memiliki {len(candidates)} pilihan lokasi di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik asal:"
                    )
                elif any(w in text for w in ("halo", "hai", "selamat", "pagi", "siang", "sore", "malam", "assalamualaikum", "tes", "test")):
                    return (
                        f"Halo! Saya {BOT_NAME}, asisten rute pengiriman paket Anda.\n"
                        f"Sebelum mencari rute paling hemat bensin, kendaraan apa yang kamu pakai hari ini?\n\n"
                        f"{VEHICLE_MENU_PROMPT}"
                    )
                else:
                    return (
                        f"Pilihan '{raw}' tidak dikenali.\n"
                        f"Ketik nomor kendaraan [1-5] atau langsung masukkan rute (contoh: 'dari JNE ke IT Del'):\n\n"
                        f"{VEHICLE_MENU_PROMPT}"
                    )

        # Tahap B: Titik Asal belum ditentukan (diproses di giliran berikutnya)
        elif self.state.origin is None:
            detected_orig, detected_dest = self.extract_landmarks(text)
            if detected_orig:
                self.state.origin = detected_orig
            if detected_dest:
                self.state.destination = detected_dest
            if not detected_orig and not detected_dest:
                ambig = check_ambiguous_landmark(text)
                if ambig:
                    term, candidates = ambig
                    self.state.pending_clarification = {
                        "slot": "origin",
                        "term": term,
                        "candidates": candidates,
                    }
                    options_str = "\n".join(
                        f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                    )
                    return (
                        f"Lokasi '{term}' memiliki {len(candidates)} pilihan lokasi di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik asal:"
                    )

                single_landmark = resolve_landmark_input(text)
                if single_landmark:
                    self.state.origin = single_landmark
                else:
                    return (
                        f"Lokasi titik jemput '{raw}' tidak ditemukan di peta jaringan jalan.\n"
                        f"Ketik nama jalan, nama tempat (misal: 'JNE', 'IT Del', 'Pasar Balige'), atau ketik 'lokasi':"
                    )

        # Tahap C: Titik Tujuan belum ditentukan (diproses di giliran berikutnya)
        elif self.state.destination is None:
            detected_orig, detected_dest = self.extract_landmarks(text)
            if detected_dest:
                self.state.destination = detected_dest
            elif detected_orig and detected_orig != self.state.origin:
                self.state.destination = detected_orig
            else:
                ambig = check_ambiguous_landmark(text)
                if ambig:
                    term, candidates = ambig
                    self.state.pending_clarification = {
                        "slot": "destination",
                        "term": term,
                        "candidates": candidates,
                    }
                    orig_label = _format_location_label(self.state.origin)
                    options_str = "\n".join(
                        f"  [{i+1}] {LANDMARK_LABELS.get(c, c)}" for i, c in enumerate(candidates)
                    )
                    return (
                        f"Titik jemput: {orig_label}\n\n"
                        f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik tujuan:"
                    )

                single_landmark = resolve_landmark_input(text)
                if single_landmark:
                    self.state.destination = single_landmark
                else:
                    orig_label = _format_location_label(self.state.origin)
                    return (
                        f"Lokasi tujuan '{raw}' tidak ditemukan di peta jaringan jalan.\n"
                        f"Titik jemput: {orig_label}\n"
                        f"Ketik nama jalan atau tempat tujuan (atau ketik 'lokasi' untuk panduan):"
                    )

        # -------------------------------------------------------------------
        # 2. Periksa Status Slot & Tentukan Tindakan Berikutnya
        # -------------------------------------------------------------------

        # Jika asal belum ada (tanyakan asal)
        if not self.state.origin:
            veh_label = get_vehicle_label(self.state.vehicle)
            return (
                f"Kendaraan: {veh_label}\n\n"
                f"Silakan tentukan titik asal (lokasi penjemputan paket):\n"
                f"Contoh: 'JNE Laguboti', 'Mixue Balige', 'Jl. Gereja', atau ketik 'lokasi'."
            )

        # Jika tujuan belum ada (tanyakan tujuan)
        if not self.state.destination:
            orig_label = _format_location_label(self.state.origin)
            return (
                f"Titik jemput: {orig_label}\n\n"
                f"Silakan tentukan titik tujuan pengantaran paket:\n"
                f"Contoh: 'IT Del', 'SMA Unggul Del', 'Pasar Porsea', atau ketik 'lokasi'."
            )

        # Validasi kasus khusus: Asal == Tujuan
        if self.state.origin == self.state.destination:
            self.state.destination = None
            return (
                "Titik asal dan tujuan tidak boleh sama karena paket sudah berada di lokasi tujuan.\n"
                "Silakan masukkan lokasi tujuan pengiriman yang berbeda:"
            )

        # -------------------------------------------------------------------
        # 3. Semua Slot Lengkap: Eksekusi Pencarian Rute & Format Hasil
        # -------------------------------------------------------------------
        try:
            res = self.solve_route(
                vehicle=self.state.vehicle,
                origin=self.state.origin,
                destination=self.state.destination,
            )
            self.state.last_result = res
            return self.format_solution_response(res)
        except Exception as e:
            return f"Maaf, terjadi kendala saat memproses rute: {e}"


# ---------------------------------------------------------------------------
# Terminal Interactive Runner
# ---------------------------------------------------------------------------

def run_chatbot_cli() -> None:
    """Jalankan sesi percakapan Chatbot secara interaktif di terminal."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(CHAT_BANNER.strip())
    bot = CourierChatbot()

    # Sambutan awal
    welcome_msg = bot.handle_message("halo")
    print(f"\n{BOT_NAME}:\n{welcome_msg}\n")

    while True:
        try:
            user_input = input("Kurir > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{BOT_NAME}: Sesi selesai. Hati-hati di jalan!\n")
            break

        if not user_input:
            continue

        response = bot.handle_message(user_input)
        print(f"\n{BOT_NAME}:\n{response}\n")

        # Cek apakah user berniat keluar
        if user_input.lower() in ("keluar", "exit", "quit", "bye"):
            break
