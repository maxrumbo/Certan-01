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
from certan_01.osm_loader import (
    LANDMARK_LABELS,
    LANDMARK_SYNONYMS,
    LANDMARKS,
    check_ambiguous_landmark,
    generate_google_maps_url,
    get_available_vehicles,
    get_vehicle_label,
    load_laguboti_graph_distance_only,
    parse_coordinates,
    resolve_landmark_input,
    resolve_vehicle_input,
)
from certan_01.search import SearchResult, astar_search, uniform_cost_search
from certan_01.visualize import generate_laguboti_folium_map


# ---------------------------------------------------------------------------
# Konstanta & Tampilan Chatbot
# ---------------------------------------------------------------------------

BOT_NAME = "KurirBot"

CHAT_BANNER = """
======================================================================
  ASISTEN PINTAR KURIR (CERTAN-01 CHATBOT)
  Optimasi Rute Pengiriman Paket Koridor Toba - Tapanuli Utara
  (Porsea - Laguboti - Balige - Siborong-borong - Tarutung)
======================================================================
"""

VEHICLE_MENU_PROMPT = """Pilih jenis kendaraan kurir yang digunakan:
  [1] Motor Matik (Honda BeAT / Vario 110-125cc) - Pertalite (50 km/L)
  [2] Motor Bebek (Honda Revo / Supra 100-125cc) - Pertalite (55 km/L)
  [3] Motor Sport / Kopling (125-150cc) - Pertamax (45 km/L)
  [4] Mobil Pick-Up / Box Kecil (1000-1300cc Bensin) - Pertamax (12 km/L)
  [5] Mobil Box Diesel (Mitsubishi L300 / Colt Diesel) - Biosolar (10 km/L)

Ketik nama kendaraan (contoh: 'vario', 'supra', 'l300') atau nomornya [1-5]:"""

LANDMARK_LIST_PROMPT = """Area & Tempat Terkenal di Koridor Toba - Tapanuli Utara:
  • Balige          : Mixue Balige, Indomaret, Alfamart, Pasar Onan, RS HKBP, Hotel Labersa, Hotel Niagara, Pantai Bulbul, Pelabuhan, Polres, BRI, Mandiri, Hutanta Coffee, Tepi Danau Bistro
  • Laguboti        : IT Del Sitoluama, SMA Unggul Del, Hub JNE / Pos, Pasar Tradisional, SPBU, Polsek, Indomaret, Ambarado Cafe, RM BPK Fly Over
  • Porsea & Sigumpar: Pasar Porsea, RSUD Porsea, Dermaga Porsea, Pantai Parparean, Makam Sisingamangaraja, Jembatan Asahan
  • Siborong-borong : Bandara Silangit, Pasar Siborong-borong, Jl. Dolok Sanggul
  • Tarutung        : Pusat Kota / Pasar Tarutung, RSUD Tarutung, Salib Kasih, Kantor Pusat HKBP Pearaja, Jl. Pahae
  • Ruas Jalan OSM  : Jl. Gereja, Jl. Sisingamangaraja, Jl. By Pass, Jl. Jambu, Jl. Hutabulu Mejan, dll.

[Tips]:
  - Ketik nama tempat terkenal langsung (misal: 'mixue', 'indomaret', 'it del', 'labersa', 'bri')
  - Ketik nama jalan langsung (misal: 'jl gereja', 'jl sisingamangaraja', 'jl jambu')
  - Ketik 'cari <nama>' untuk mencari lokasi (misal: 'cari mixue', 'cari hotel', 'cari bank')"""



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

        def heuristic(node, g=goal):
            return euclidean_heuristic(node, g)

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

        # Kumpulkan node objek jalur A* untuk pembentukan waypoints Google Maps
        path_nodes = []
        if astar_fuel.found:
            for nid in astar_fuel.path_ids:
                try:
                    path_nodes.append(graph_fuel.get_node(nid))
                except KeyError:
                    pass

        gmaps_url = generate_google_maps_url(
            origin_coords=orig_coords,
            dest_coords=dest_coords,
            path_nodes=path_nodes,
            origin_name=orig_label,
            dest_name=dest_label,
        )

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
            "gmaps_url": gmaps_url,
        }

    # -----------------------------------------------------------------------
    # Response Formatter
    # -----------------------------------------------------------------------

    def format_solution_response(self, res: dict[str, Any]) -> str:
        """Format hasil kalkulasi optimasi rute ke dalam pesan balasan percakapan yang bersih dan profesional."""
        astar: SearchResult = res["astar_fuel"]
        ucs: SearchResult = res["ucs_dist"]
        lines: list[str] = []

        lines.append("RUTE PENGIRIMAN PAKET")
        lines.append("=" * 60)
        lines.append(f"Kendaraan    : {res['vehicle_label']}")
        lines.append(f"Titik Jemput : {res['origin_label']}")
        lines.append(f"Tujuan Paket : {res['destination_label']}")
        lines.append("-" * 60)

        if not astar.found:
            lines.append("Maaf, tidak ditemukan rute jalan yang menghubungkan kedua lokasi ini.")
            return "\n".join(lines)

        lines.append("\nROUTE HEMAT BBM (Direkomendasikan):")
        lines.append(f"- Estimasi Biaya BBM : Rp {astar.total_cost:,.0f}")
        lines.append(f"- Jarak Tempuh       : {astar.total_distance:.2f} km")

        if astar.path_names:
            if len(astar.path_names) > 3:
                path_str = f"{astar.path_names[0]} -> {astar.path_names[1]} -> ... -> {astar.path_names[-1]}"
            else:
                path_str = " -> ".join(astar.path_names)
            lines.append(f"- Jalur Utama        : {path_str}")

        lines.append("\nPERBANDINGAN DENGAN RUTE TERPENDEK:")
        lines.append(f"- Jarak Terpendek    : {ucs.total_distance:.2f} km (BBM: Rp {res['dist_fuel_cost']:,.0f})")

        if res["savings_rp"] > 10:
            lines.append(
                f"- Potensi Hemat BBM  : Rp {res['savings_rp']:,.0f} ({res['savings_pct']:.1f}%) "
                f"dengan rute ini dibanding rute terpendek."
            )
        else:
            lines.append(
                "- Rute ini sudah merupakan jalur paling hemat BBM."
            )

        lines.append("\nNAVIGASI (GOOGLE MAPS):")
        lines.append(f"Buka di HP: {res['gmaps_url']}")

        if res["map_path"]:
            lines.append(f"\nPeta tersimpan di: {res['map_path']}")

        lines.append("=" * 60)
        lines.append("Ketik 'antar lagi' untuk rute baru (kendaraan tetap), 'reset' untuk mulai dari awal, atau 'keluar'.")
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
            return "Sampai jumpa! Selamat bertugas dan utamakan keselamatan di jalan."

        if text in ("bantuan", "help", "panduan"):
            return (
                f"**BANTUAN ASISTEN KURIR ({BOT_NAME})**:\n"
                f"Ketik nama kendaraan, lalu lokasi jemput, lalu tujuan. Atau langsung:\n"
                f"  Contoh: 'Saya bawa motor Supra, antar dari JNE ke IT Del'\n\n"
                f"Perintah:\n"
                f"  • **antar lagi**    : Rute baru, kendaraan tetap (tidak perlu pilih ulang)\n"
                f"  • **reset**         : Mulai dari awal, termasuk ganti kendaraan\n"
                f"  • **ganti kendaraan**: Ubah kendaraan saja\n"
                f"  • **lokasi**        : Lihat daftar lokasi yang tersedia\n"
                f"  • **keluar**        : Selesai"
            )

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
                    f"Silakan ketik salah satu nama di atas atau masukkan koordinat GPS:"
                )
            else:
                return (
                    f"Tidak ditemukan lokasi dengan kata kunci '{query}'.\n"
                    f"Tips: Ketik 'lokasi' untuk melihat daftar area, atau masukkan titik koordinat GPS (lat, lon)."
                )

        if text in ("lokasi", "alamat", "daftar lokasi", "tempat", "opsi", "pilihan", "list", "daftar", "daftar tempat", "menu lokasi"):
            status_text = ""
            if self.state.origin:
                orig_label = _format_location_label(self.state.origin)
                status_text = f"Titik jemput saat ini: **{orig_label}**\n\n"
            return (
                f"{LANDMARK_LIST_PROMPT}\n\n"
                f"{status_text}Silakan ketik nama lokasi yang diinginkan atau masukkan koordinat GPS:"
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
                    f"Siap antar lagi! Kendaraan masih **{veh_label}**.\n"
                    f"Tentukan titik jemput berikutnya (atau ketik 'ganti kendaraan'):"
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
                            f"Titik jemput paket tercatat di **{orig_label}**.\n\n"
                            f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                            f"{options_str}\n\n"
                            f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan tujuan pengantaran:"
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
                                f"Titik jemput paket tercatat di **{orig_label}**.\n\n"
                                f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                                f"{options_str}\n\n"
                                f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan tujuan pengantaran:"
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
                        f"Lokasi '{term}' memiliki {len(candidates)} pilihan di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik jemput paket:"
                    )
                elif any(w in text for w in ("halo", "hai", "selamat", "pagi", "siang", "sore", "malam", "assalamualaikum", "tes", "test")):
                    return (
                        f"Halo! Saya **{BOT_NAME}**, asisten cerdas rute pengiriman paket Anda.\n"
                        f"Sebelum mencari rute paling hemat bensin, kendaraan apa yang kamu pakai hari ini?\n\n"
                        f"{VEHICLE_MENU_PROMPT}"
                    )
                else:
                    return (
                        f"Pilihan kendaraan atau lokasi '{raw}' tidak dikenali.\n"
                        f"Ketik nomor kendaraan [1-5] (misal: '1' untuk Motor Matik) atau langsung ketik nama jalan / tujuan (misal: 'Jl. Gereja', 'IT Del'):\n\n"
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
                        f"Lokasi '{term}' memiliki {len(candidates)} pilihan di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan titik jemput:"
                    )

                single_landmark = resolve_landmark_input(text)
                if single_landmark:
                    self.state.origin = single_landmark
                else:
                    return (
                        f"Lokasi titik jemput '{raw}' tidak ditemukan di peta jaringan jalan.\n"
                        f"Silakan ketik nama jalan (misal: 'Jl. Gereja', 'Jl. Sisingamangaraja'), nama tempat (misal: 'JNE', 'IT Del', 'Pasar Balige'), atau ketik 'lokasi' untuk panduan area:"
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
                        f"Titik jemput saat ini: **{orig_label}**.\n\n"
                        f"Untuk tujuan '{term}', terdapat {len(candidates)} pilihan lokasi di wilayah ini:\n"
                        f"{options_str}\n\n"
                        f"Ketik nomor pilihannya [1-{len(candidates)}] untuk menentukan tujuan pengantaran:"
                    )

                single_landmark = resolve_landmark_input(text)
                if single_landmark:
                    self.state.destination = single_landmark
                else:
                    orig_label = _format_location_label(self.state.origin)
                    return (
                        f"Lokasi tujuan '{raw}' tidak ditemukan di peta jaringan jalan.\n"
                        f"Titik jemput saat ini: **{orig_label}**\n"
                        f"Silakan ketik nama jalan (misal: 'Jl. Gereja', 'Jl. Sisingamangaraja'), nama tempat (misal: 'IT Del', 'Balige'), atau ketik 'lokasi' untuk panduan area:"
                    )

        # -------------------------------------------------------------------
        # 2. Periksa Status Slot & Tentukan Tindakan Berikutnya
        # -------------------------------------------------------------------

        # Jika asal belum ada (tanyakan asal)
        if not self.state.origin:
            veh_label = get_vehicle_label(self.state.vehicle)
            return (
                f"Kendaraan tercatat: **{veh_label}**.\n\n"
                f"Silakan tentukan titik asal (lokasi penjemputan paket):\n"
                f"Ketik nama jalan (misal: 'Jl. Gereja', 'Jl. Sisingamangaraja'), nama tempat (misal: 'JNE', 'Silangit', 'Balige'), atau koordinat GPS.\n"
                f"(💡 Tips: Ketik 'lokasi' untuk melihat daftar opsi tempat, atau 'cari <nama>' untuk mencari)"
            )

        # Jika tujuan belum ada (tanyakan tujuan)
        if not self.state.destination:
            orig_label = _format_location_label(self.state.origin)
            return (
                f"Titik jemput paket tercatat di **{orig_label}**.\n\n"
                f"Silakan tentukan titik tujuan pengantaran paket:\n"
                f"Ketik nama jalan (misal: 'Jl. Gereja', 'Jl. Sisingamangaraja'), nama tempat (misal: 'IT Del', 'Porsea', 'Tarutung'), atau koordinat GPS.\n"
                f"(💡 Tips: Ketik 'lokasi' untuk melihat daftar opsi tempat, atau 'cari <nama>' untuk mencari)"
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
            return f"❌ Maaf, terjadi kendala saat memproses rute: {e}"


# ---------------------------------------------------------------------------
# Terminal Interactive Runner
# ---------------------------------------------------------------------------

def run_chatbot_cli() -> None:
    """Jalankan sesi percakapan Chatbot secara interaktif di terminal."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(CHAT_BANNER)
    bot = CourierChatbot()

    # Sambutan awal
    welcome_msg = bot.handle_message("halo")
    print(f"\n{BOT_NAME}: {welcome_msg}\n")

    while True:
        try:
            user_input = input("Kurir > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{BOT_NAME}: Sesi diakhiri. Semangat bertugas dan hati-hati di jalan! 👋\n")
            break

        if not user_input:
            continue

        response = bot.handle_message(user_input)
        print(f"\n{BOT_NAME}:\n{response}\n")

        # Cek apakah user berniat keluar
        if user_input.lower() in ("keluar", "exit", "quit", "bye"):
            break
