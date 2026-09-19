"""
osm_loader.py
-------------
Modul pemuatan dan konversi data jaringan jalan nyata Laguboti dari format
OpenStreetMap (GraphML) ke objek Graph internal sistem.

Sumber Data:
  - Topologi Jalan : data/laguboti_network.graphml
    (OSMnx graph_from_point, radius 4.5 km dari IT Del, network_type='drive')
  - Model Biaya BBM: data/vehicle_fuel_model.json
    (profil kendaraan kurir + faktor konsumsi per tipe jalan OSM)

Cara Konversi:
  1. Baca GraphML menggunakan osmnx / networkx.
  2. Setiap OSM node -> Node(node_id=str(osmid), name=nama_jalan, x=lon, y=lat).
  3. Setiap OSM edge -> Edge(distance_km, fuel_cost_per_km, extra_cost).
     fuel_cost_per_km = base_cost_per_km * consumption_multiplier[highway_type]
  4. Tandai node terdekat dari koordinat landmark sebagai hub / goal.

Catatan penting:
  - Koordinat x=longitude, y=latitude (sesuai konvensi internal models.py).
  - Heuristik Euclidean di heuristics.py tetap valid karena jarak Euclidean
    dalam ruang (lon, lat) tetap merupakan lower-bound jarak jalan nyata.
  - Semua unit: jarak dalam km, biaya dalam Rupiah (IDR).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import osmnx as ox

from certan_01.models import Graph, Node


# ---------------------------------------------------------------------------
# Path Default
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]   # src/certan_01/osm_loader.py -> src/certan_01 -> src -> repo root
_GRAPHML_PATH  = _ROOT / "data" / "laguboti_network.graphml"
_TOBA_GRAPHML_PATH = _ROOT / "data" / "toba_taput_network.graphml"
_FUEL_CFG_PATH = _ROOT / "data" / "vehicle_fuel_model.json"

# In-memory cache graf OSM untuk mempercepat perutean ulang tanpa baca disk
_OSM_GRAPH_CACHE: dict[str, nx.MultiDiGraph] = {}


def get_cached_osm_graph(path: Path | str) -> nx.MultiDiGraph:
    """Muat dan cache objek nx.MultiDiGraph agar tidak membaca ulang dari disk."""
    p = Path(path).resolve()
    key = str(p)
    if key not in _OSM_GRAPH_CACHE:
        _OSM_GRAPH_CACHE[key] = ox.load_graphml(p)
    return _OSM_GRAPH_CACHE[key]


# ---------------------------------------------------------------------------
# Koordinat Landmark Nyata (Koridor Porsea - Laguboti - Balige - Tarutung)
# Digunakan untuk snap ke OSM node terdekat
# ---------------------------------------------------------------------------

LANDMARKS: dict[str, tuple[float, float]] = {
    # --- Wilayah Laguboti ---
    "hub_jne_laguboti"    : (2.3667, 99.1247),   # Pusat Pasar / Area JNE Laguboti
    "it_del_sitoluama"    : (2.3832, 99.1486),   # Institut Teknologi Del
    "sma_unggul_del"      : (2.3881, 99.1505),   # SMA Unggul Del
    "simpang_balige_jalan": (2.3550, 99.1150),   # Simpang ke Jl. Dr. TB Silalahi
    "pasar_laguboti"      : (2.3660, 99.1240),   # Pasar Tradisional Laguboti
    "spbu_laguboti"       : (2.3615, 99.1215),   # SPBU Pertamina Laguboti
    "stasiun_laguboti"    : (2.3705, 99.1285),   # Stasiun Laguboti
    "kantor_camat"        : (2.3645, 99.1225),   # Kantor Camat Laguboti
    "polsek_laguboti"     : (2.3675, 99.1255),   # Polsek Laguboti
    "puskesmas_laguboti"  : (2.3685, 99.1270),   # Puskesmas Laguboti
    "desa_sitoluama"      : (2.3840, 99.1460),   # Desa Sitoluama
    "desa_lumban_bagasan" : (2.3750, 99.1350),   # Desa Lumban Bagasan

    # --- Wilayah Balige (Ibukota Kab. Toba) ---
    "pusat_pasar_balige"   : (2.3333, 99.0645),   # Pusat Pasar Tradisional Onan Balige
    "rs_hkbp_balige"       : (2.3361, 99.0722),   # RS HKBP Balige
    "simpang_bypass_balige": (2.3275, 99.0550),   # Simpang Jalur By Pass Balige
    "museum_tb_silalahi"   : (2.3488, 99.0832),   # TB Silalahi Center Balige
    "pantai_bulbul"        : (2.3450, 99.0700),   # Pantai Pasir Putih Lumban Bulbul
    "kantor_bupati_toba"   : (2.3310, 99.0610),   # Kantor Bupati Toba Balige
    "hotel_labersa"        : (2.3350, 99.0950),   # Hotel Labersa Grand Toba
    "hotel_niagara"        : (2.3310, 99.0640),   # Hotel Niagara Balige
    "hotel_pardede"        : (2.3365, 99.0730),   # Hotel Pardede International Balige

    # --- Wilayah Porsea & Sigumpar (Toba) ---
    "pasar_porsea"         : (2.4462, 99.1448),   # Pusat Pasar / Dermaga Porsea
    "simpang_sigumpar"     : (2.4085, 99.1352),   # Simpang Sigumpar (Porsea - Laguboti)
    "makam_sisingamangaraja": (2.3950, 99.1300),  # Makam Raja Sisingamangaraja XII

    # --- Wilayah Desa Sekitar Laguboti ---
    "desa_aruan"           : (2.3710, 99.1180),   # Desa Aruan Laguboti
    "desa_sibuea"          : (2.3780, 99.1220),   # Desa Sibuea Laguboti

    # --- Wilayah Siborong-borong (Tapanuli Utara) ---
    "bandara_silangit"     : (2.2608, 98.9917),   # Bandara Internasional Silangit (DTB)
    "pasar_siborongborong" : (2.2135, 98.9723),   # Pusat Pasar Siborong-borong
    "jl_dolok_sanggul"     : (2.2192, 98.9543),   # Jl. Dolok Sanggul - Siborong-borong

    # --- Wilayah Tarutung (Ibukota Kab. Tapanuli Utara) ---
    "pusat_kota_tarutung"  : (2.0235, 98.9667),   # Pusat Pasar & Alun-alun Tarutung
    "rsud_tarutung"        : (2.0285, 98.9615),   # RSUD Swadana Tarutung
    "salib_kasih"          : (2.0080, 98.9780),   # Wisata Salib Kasih Siatas Barita
    "kantor_pusat_hkbp"    : (2.0250, 98.9720),   # Kantor Pusat HKBP Pearaja Tarutung

    # --- Ruas Jalan Terdaftar di Peta OSM (Balige, Laguboti, Porsea, Tarutung) ---
    "jl_sisingamangaraja"     : (2.3343, 99.0669),   # Jl. Sisingamangaraja Balige
    "jl_gereja_balige"        : (2.3311, 99.0669),   # Jl. Gereja Balige
    "jl_hutabulu_mejan"       : (2.3315, 99.0845),   # Jl. Hutabulu Mejan Balige
    "jl_jambu_balige"         : (2.3258, 99.0688),   # Jl. Jambu Balige
    "jl_uma_rihit"            : (2.3271, 99.0695),   # Jl. Uma Rihit Balige
    "jl_pardolok_tolong"      : (2.3353, 99.0666),   # Jl. Pardolok Tolong Balige
    "jl_pardomuan_balige"     : (2.3277, 99.0440),   # Jl. Pardomuan Balige
    "jl_porsea_balige"        : (2.3744, 99.1246),   # Jl. Porsea - Balige
    "jl_siborongborong_balige": (2.2651, 98.9989),   # Jl. Siborong-borong - Balige
    "jl_lumbanjulu"           : (2.2186, 99.0086),   # Jl. Lumbanjulu
    "jl_bengkel"              : (2.1577, 98.9623),   # Jl. Bengkel
    "jl_pahae_tarutung"       : (2.0173, 98.9839),   # Jl. Pahae Tarutung
    "jl_sibolga_tarutung"     : (2.0012, 98.9597),   # Jl. Sibolga - Tarutung
    "jl_parapat_porsea"       : (2.4601, 99.1494),   # Jl. Parapat - Porsea
    "jembatan_sungai_asahan"  : (2.4423, 99.1566),   # Jembatan Sungai Asahan Porsea
    "jl_lintas_tengah_sumatera": (2.0049, 98.9906),  # Jl. Lintas Tengah Sumatera
    "trans_sumatra_highway"   : (2.3538, 99.0954),   # Trans Sumatra Highway

    # --- Tempat Terkenal & Populer Google Maps (Balige, Laguboti, Porsea) ---
    "mixue_balige"            : (2.3340, 99.0665),   # Mixue Balige (Jl. Sisingamangaraja)
    "indomaret_balige"        : (2.3335, 99.0650),   # Indomaret Balige
    "alfamart_balige"         : (2.3342, 99.0660),   # Alfamart Balige
    "indomaret_laguboti"      : (2.3665, 99.1245),   # Indomaret Laguboti
    "alfamart_laguboti"       : (2.3670, 99.1250),   # Alfamart Laguboti
    "hutanta_coffee"          : (2.3360, 99.0710),   # Hutanta Coffee & Resto Balige
    "tepi_danau_bistro"       : (2.3420, 99.0730),   # Tepi Danau Bistro Balige
    "lagos_cafe"              : (2.3440, 99.0750),   # Lago's Cafe Balige (Pantai Sibolahotang)
    "ambarado_cafe"           : (2.3850, 99.1420),   # Ambarado Cafe & Resto Laguboti
    "rm_bpk_panca"            : (2.3350, 99.0650),   # RM BPK Panca Balige
    "rm_bpk_flyover"          : (2.3680, 99.1260),   # RM BPK Fly Over Laguboti
    "spbu_balige"             : (2.3370, 99.0700),   # SPBU Pertamina Balige (Juanda)
    "apotek_kimia_farma_balige": (2.3340, 99.0658),  # Apotek Kimia Farma Balige
    "bri_balige"              : (2.3330, 99.0640),   # Bank BRI Cabang Balige
    "mandiri_balige"          : (2.3335, 99.0648),   # Bank Mandiri Balige
    "bni_balige"              : (2.3340, 99.0655),   # Bank BNI Balige
    "bank_sumut_balige"       : (2.3325, 99.0635),   # Bank Sumut Balige
    "bank_sumut_laguboti"     : (2.3668, 99.1248),   # Bank Sumut Laguboti
    "bri_laguboti"            : (2.3660, 99.1240),   # Bank BRI Unit Laguboti
    "polres_toba"             : (2.3290, 99.0580),   # Polres Toba Balige
    "pelabuhan_balige"        : (2.3380, 99.0680),   # Pelabuhan / Dermaga Kapal Balige
    "lapangan_sisingamangaraja_balige": (2.3330, 99.0630), # Lapangan Sisingamangaraja Balige (Alun-alun)
    "kantor_pos_balige"       : (2.3338, 99.0652),   # Kantor Pos Balige
    "sman_1_balige"           : (2.3385, 99.0740),   # SMA Negeri 1 Balige
    "sman_2_balige"           : (2.3410, 99.0760),   # SMA Negeri 2 Balige (Soposurung)
    "gereja_hkbp_balige"      : (2.3315, 99.0670),   # Gereja HKBP Balige Kota
    "rsud_porsea"             : (2.4475, 99.1465),   # RSUD Porsea
    "pantai_parparean"        : (2.4510, 99.1520),   # Pantai Pasir Putih Parparean Porsea
}

LANDMARK_LABELS: dict[str, str] = {
    # Laguboti
    "hub_jne_laguboti"        : "Hub JNE / Kantor Pos Laguboti",
    "it_del_sitoluama"        : "Institut Teknologi Del (Sitoluama)",
    "sma_unggul_del"          : "SMA Unggul Del",
    "simpang_balige_jalan"    : "Simpang Jl. Dr. TB. Silalahi",
    "pasar_laguboti"          : "Pasar Tradisional Laguboti",
    "spbu_laguboti"           : "SPBU Pertamina Laguboti",
    "stasiun_laguboti"        : "Stasiun Laguboti",
    "kantor_camat"            : "Kantor Camat Laguboti",
    "polsek_laguboti"         : "Polsek Laguboti",
    "puskesmas_laguboti"      : "Puskesmas Laguboti",
    "desa_sitoluama"          : "Desa Sitoluama",
    "desa_lumban_bagasan"     : "Desa Lumban Bagasan",
    "desa_aruan"              : "Desa Aruan (Laguboti)",
    "desa_sibuea"             : "Desa Sibuea (Laguboti)",

    # Balige
    "pusat_pasar_balige"      : "Pusat Pasar Balige (Onan)",
    "rs_hkbp_balige"          : "Rumah Sakit HKBP Balige",
    "simpang_bypass_balige"   : "Simpang Jalur By Pass Balige",
    "museum_tb_silalahi"      : "Museum TB Silalahi Center",
    "pantai_bulbul"           : "Pantai Lumban Bulbul Balige",
    "kantor_bupati_toba"      : "Kantor Bupati Toba (Balige)",

    # Porsea & Sigumpar
    "pasar_porsea"            : "Pusat Pasar & Dermaga Porsea",
    "simpang_sigumpar"        : "Simpang Sigumpar (Porsea)",
    "makam_sisingamangaraja"  : "Makam Sisingamangaraja XII (Sigumpar)",

    # Siborong-borong
    "bandara_silangit"        : "Bandara Internasional Silangit (DTB)",
    "pasar_siborongborong"    : "Pusat Pasar Siborong-borong",
    "jl_dolok_sanggul"        : "Jl. Dolok Sanggul - Siborong-borong",

    # Tarutung
    "pusat_kota_tarutung"     : "Pusat Kota / Pasar Tarutung",
    "rsud_tarutung"           : "RSUD Tarutung",
    "salib_kasih"             : "Wisata Salib Kasih Siatas Barita",
    "kantor_pusat_hkbp"       : "Kantor Pusat HKBP Pearaja Tarutung",

    # Hotel
    "hotel_labersa"           : "Hotel Labersa Grand Toba (Balige)",
    "hotel_niagara"           : "Hotel Niagara Balige",
    "hotel_pardede"           : "Hotel Pardede International (Balige)",

    # Ruas Jalan OSM
    "jl_sisingamangaraja"     : "Jl. Sisingamangaraja (Balige)",
    "jl_gereja_balige"        : "Jl. Gereja (Balige)",
    "jl_hutabulu_mejan"       : "Jl. Hutabulu Mejan (Balige)",
    "jl_jambu_balige"         : "Jl. Jambu (Balige)",
    "jl_uma_rihit"            : "Jl. Uma Rihit (Balige)",
    "jl_pardolok_tolong"      : "Jl. Pardolok Tolong (Balige)",
    "jl_pardomuan_balige"     : "Jl. Pardomuan (Balige)",
    "jl_porsea_balige"        : "Jl. Porsea - Balige (Lintas)",
    "jl_siborongborong_balige": "Jl. Siborong-borong - Balige",
    "jl_lumbanjulu"           : "Jl. Lumbanjulu",
    "jl_bengkel"              : "Jl. Bengkel",
    "jl_pahae_tarutung"       : "Jl. Pahae (Tarutung)",
    "jl_sibolga_tarutung"     : "Jl. Sibolga - Tarutung",
    "jl_parapat_porsea"       : "Jl. Parapat - Porsea",
    "jembatan_sungai_asahan"  : "Jembatan Sungai Asahan (Porsea)",
    "jl_lintas_tengah_sumatera": "Jl. Lintas Tengah Sumatera",
    "trans_sumatra_highway"   : "Trans Sumatra Highway",

    # Tempat Terkenal Populer
    "mixue_balige"            : "Mixue Balige (Jl. Sisingamangaraja)",
    "indomaret_balige"        : "Indomaret Balige (Jl. Sisingamangaraja)",
    "alfamart_balige"         : "Alfamart Balige",
    "indomaret_laguboti"      : "Indomaret Laguboti",
    "alfamart_laguboti"       : "Alfamart Laguboti",
    "hutanta_coffee"          : "Hutanta Coffee & Resto Balige",
    "tepi_danau_bistro"       : "Tepi Danau Bistro Balige",
    "lagos_cafe"              : "Lago's Cafe Balige",
    "ambarado_cafe"           : "Ambarado Cafe & Resto Laguboti",
    "rm_bpk_panca"            : "RM BPK Panca Balige",
    "rm_bpk_flyover"          : "RM BPK Fly Over Laguboti",
    "spbu_balige"             : "SPBU Pertamina Balige (Juanda)",
    "apotek_kimia_farma_balige": "Apotek Kimia Farma Balige",
    "bri_balige"              : "Bank BRI Cabang Balige",
    "mandiri_balige"          : "Bank Mandiri Balige",
    "bni_balige"              : "Bank BNI Balige",
    "bank_sumut_balige"       : "Bank Sumut Balige",
    "bank_sumut_laguboti"     : "Bank Sumut Laguboti",
    "bri_laguboti"            : "Bank BRI Unit Laguboti",
    "polres_toba"             : "Polres Toba (Balige)",
    "pelabuhan_balige"        : "Pelabuhan / Dermaga Balige",
    "lapangan_sisingamangaraja_balige": "Lapangan Sisingamangaraja (Alun-alun Balige)",
    "kantor_pos_balige"       : "Kantor Pos Balige",
    "sman_1_balige"           : "SMA Negeri 1 Balige",
    "sman_2_balige"           : "SMA Negeri 2 Balige (Soposurung)",
    "gereja_hkbp_balige"      : "Gereja HKBP Balige Kota",
    "rsud_porsea"             : "RSUD Porsea",
    "pantai_parparean"        : "Pantai Pasir Putih Parparean (Porsea)",
}


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """
    Ekstrak pasangan koordinat (latitude, longitude) dari teks atau URL Google Maps.
    Contoh input valid:
      - "2.3832, 99.1486"
      - "2.3832,99.1486"
      - "2.3832 99.1486"
    """
    import re
    match = re.search(r"(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)", text.strip())
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (round(lat, 6), round(lon, 6))
        except ValueError:
            pass
    return None


def generate_google_maps_url(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    path_nodes: list[Any] | None = None,
    origin_name: str | None = None,
    dest_name: str | None = None,
) -> str:
    """
    Hasilkan URL Google Maps Direction resmi.
    Jika nama tempat tersedia, gunakan parameter query nama tempat (Google Maps URLs format)
    agar Google Maps menampilkan nama lokasi yang tepat (misal: 'Mixue Balige' dan 'Institut Teknologi Del')
    bukan nama bisnis acak di sebelah lokasi hasil reverse-geocoding koordinat mentah.
    """
    import re
    import urllib.parse

    orig_str = f"{origin_coords[0]:.6f},{origin_coords[1]:.6f}"
    dest_str = f"{dest_coords[0]:.6f},{dest_coords[1]:.6f}"

    if origin_name and dest_name:
        clean_orig = re.sub(r"\s*\([^)]*\)", "", origin_name).strip()
        clean_dest = re.sub(r"\s*\([^)]*\)", "", dest_name).strip()
        if clean_orig and clean_dest:
            orig_q = urllib.parse.quote_plus(clean_orig)
            dest_q = urllib.parse.quote_plus(clean_dest)
            return f"https://www.google.com/maps/dir/?api=1&origin={orig_q}&destination={dest_q}"

    return f"https://www.google.com/maps/dir/{orig_str}/{dest_str}/"



# ---------------------------------------------------------------------------
# Helper: Baca Konfigurasi BBM
# ---------------------------------------------------------------------------

def _load_fuel_config(path: Path = _FUEL_CFG_PATH) -> dict:
    """Baca file konfigurasi kendaraan & BBM dari JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_fuel_cost_per_km(highway: str, vehicle_profile: dict, taxonomy: dict) -> float:
    """
    Hitung biaya bahan bakar per km untuk tipe jalan tertentu.

    Formula:
        biaya/km = base_cost_per_km * consumption_multiplier[highway_type]

    Args:
        highway        : Nilai tag OSM 'highway' (misal: 'trunk', 'residential').
        vehicle_profile: Dict profil kendaraan dari vehicle_fuel_model.json.
        taxonomy       : Dict road_traffic_taxonomy dari vehicle_fuel_model.json.

    Returns:
        float: Biaya BBM per km dalam Rupiah.
    """
    if isinstance(highway, list):
        highway = highway[0]

    base_cost = vehicle_profile["base_cost_per_km_idr"]
    road_info = taxonomy.get(highway, taxonomy.get("residential"))
    multiplier = road_info["consumption_multiplier"]

    return round(base_cost * multiplier, 2)


# ---------------------------------------------------------------------------
# Pemilihan File GraphML & Pemuatan Graf
# ---------------------------------------------------------------------------

LAGUBOTI_LOCAL_LANDMARKS = {
    "hub_jne_laguboti", "it_del_sitoluama", "sma_unggul_del", "simpang_balige_jalan",
    "pasar_laguboti", "spbu_laguboti", "stasiun_laguboti", "kantor_camat",
    "polsek_laguboti", "puskesmas_laguboti", "desa_sitoluama", "desa_lumban_bagasan",
}


def resolve_graph_path(
    hub_landmark: Any = None,
    goal_landmark: Any = None,
    explicit_path: Path | str | None = None,
) -> Path:
    """
    Tentukan file GraphML terbaik secara otomatis.
    Jika kedua titik merupakan landmark lokal Laguboti dan file lokal tersedia,
    gunakan laguboti_network.graphml (super cepat untuk unit test/demo).
    Jika salah satu titik berada di wilayah regional (Balige, Porsea, Silangit, Tarutung)
    atau berupa koordinat GPS, gunakan toba_taput_network.graphml.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p

    if _TOBA_GRAPHML_PATH.exists():
        is_hub_local = isinstance(hub_landmark, str) and hub_landmark in LAGUBOTI_LOCAL_LANDMARKS
        is_goal_local = isinstance(goal_landmark, str) and goal_landmark in LAGUBOTI_LOCAL_LANDMARKS
        if is_hub_local and is_goal_local and _GRAPHML_PATH.exists():
            return _GRAPHML_PATH
        return _TOBA_GRAPHML_PATH

    return _GRAPHML_PATH


def load_laguboti_graph(
    hub_landmark: str = "hub_jne_laguboti",
    goal_landmark: str = "it_del_sitoluama",
    vehicle: str = "motor_matik_kurir",
    graphml_path: Path | str | None = None,
    fuel_cfg_path: Path = _FUEL_CFG_PATH,
) -> Graph:
    """
    Muat jaringan jalan nyata (Laguboti atau Koridor Regional Toba-Taput) dari GraphML
    dan konversi ke Graph internal sistem.
    """
    target_graphml = resolve_graph_path(hub_landmark, goal_landmark, graphml_path)
    if not target_graphml.exists():
        raise FileNotFoundError(
            f"File jaringan jalan tidak ditemukan: {target_graphml}\n"
            "Pastikan file toba_taput_network.graphml atau laguboti_network.graphml sudah ada."
        )
    if not fuel_cfg_path.exists():
        raise FileNotFoundError(f"File konfigurasi BBM tidak ditemukan: {fuel_cfg_path}")

    # 1. Muat konfigurasi BBM
    cfg = _load_fuel_config(fuel_cfg_path)
    vehicle_profile = cfg["vehicle_profiles"][vehicle]
    taxonomy = cfg["road_traffic_taxonomy"]

    # 2. Muat graf OSM dari GraphML (menggunakan in-memory cache)
    G_osm = get_cached_osm_graph(target_graphml)

    # 3. Snap landmark atau koordinat GPS ke OSM node terdekat
    def _resolve_coords(point: Any) -> tuple[float, float]:
        if isinstance(point, (tuple, list)):
            return float(point[0]), float(point[1])
        if isinstance(point, str) and point in LANDMARKS:
            return LANDMARKS[point]
        parsed = parse_coordinates(str(point))
        if parsed:
            return parsed
        raise KeyError(f"Titik lokasi '{point}' tidak ditemukan di landmark maupun format koordinat.")

    hub_lat,  hub_lon  = _resolve_coords(hub_landmark)
    goal_lat, goal_lon = _resolve_coords(goal_landmark)

    hub_osm_id  = ox.nearest_nodes(G_osm, X=hub_lon,  Y=hub_lat)
    goal_osm_id = ox.nearest_nodes(G_osm, X=goal_lon, Y=goal_lat)

    # 4. Gunakan komponen terhubung terbesar agar graf pasti connected
    largest_wcc = max(nx.weakly_connected_components(G_osm), key=len)
    G_sub = G_osm.subgraph(largest_wcc).copy()

    # 5. Konversi ke Graph internal
    graph = Graph()

    # 5a. Tambahkan semua node
    for osmid, data in G_sub.nodes(data=True):
        node_id = str(osmid)
        lat     = float(data.get("y", 0.0))   # latitude
        lon     = float(data.get("x", 0.0))   # longitude

        name_raw = data.get("name", None)
        if isinstance(name_raw, list):
            name_raw = name_raw[0]
        name = name_raw if name_raw else f"Persimpangan-{node_id[-5:]}"

        is_hub  = (osmid == hub_osm_id)
        is_goal = (osmid == goal_osm_id)

        # x=lon, y=lat sesuai konvensi models.py (heuristik Euclidean)
        graph.add_node(Node(node_id, name, x=lon, y=lat, is_hub=is_hub, is_goal=is_goal))

    # 5b. Tambahkan semua edge (deduplikasi agar undirected)
    added_pairs: set[tuple[str, str]] = set()
    for u, v, data in G_sub.edges(data=True):
        uid, vid = str(u), str(v)
        pair = (min(uid, vid), max(uid, vid))
        if pair in added_pairs:
            continue

        length_m = float(data.get("length", 0.0))
        dist_km  = length_m / 1000.0
        if dist_km <= 0:
            continue

        highway     = data.get("highway", "residential")
        cost_per_km = _get_fuel_cost_per_km(highway, vehicle_profile, taxonomy)

        graph.add_edge(uid, vid, dist_km, cost_per_km, extra_cost=0.0)
        added_pairs.add(pair)

    return graph


def load_laguboti_graph_distance_only(
    hub_landmark: Any = "hub_jne_laguboti",
    goal_landmark: Any = "it_del_sitoluama",
    graphml_path: Path | str | None = None,
) -> Graph:
    """
    Muat jaringan jalan dengan bobot JARAK MURNI (km) saja,
    tanpa memperhitungkan biaya BBM atau kondisi jalan.
    """
    target_graphml = resolve_graph_path(hub_landmark, goal_landmark, graphml_path)
    if not target_graphml.exists():
        raise FileNotFoundError(f"File jaringan jalan tidak ditemukan: {target_graphml}")

    G_osm = get_cached_osm_graph(target_graphml)

    def _resolve_coords(point: Any) -> tuple[float, float]:
        if isinstance(point, (tuple, list)):
            return float(point[0]), float(point[1])
        if isinstance(point, str) and point in LANDMARKS:
            return LANDMARKS[point]
        parsed = parse_coordinates(str(point))
        if parsed:
            return parsed
        raise KeyError(f"Titik lokasi '{point}' tidak ditemukan di landmark maupun format koordinat.")

    hub_lat,  hub_lon  = _resolve_coords(hub_landmark)
    goal_lat, goal_lon = _resolve_coords(goal_landmark)
    hub_osm_id  = ox.nearest_nodes(G_osm, X=hub_lon,  Y=hub_lat)
    goal_osm_id = ox.nearest_nodes(G_osm, X=goal_lon, Y=goal_lat)


    largest_wcc = max(nx.weakly_connected_components(G_osm), key=len)
    G_sub = G_osm.subgraph(largest_wcc).copy()

    graph = Graph()
    for osmid, data in G_sub.nodes(data=True):
        node_id  = str(osmid)
        lat      = float(data.get("y", 0.0))
        lon      = float(data.get("x", 0.0))
        name_raw = data.get("name", None)
        if isinstance(name_raw, list):
            name_raw = name_raw[0]
        name    = name_raw if name_raw else f"Persimpangan-{node_id[-5:]}"
        is_hub  = (osmid == hub_osm_id)
        is_goal = (osmid == goal_osm_id)
        graph.add_node(Node(node_id, name, x=lon, y=lat, is_hub=is_hub, is_goal=is_goal))

    added_pairs: set[tuple[str, str]] = set()
    for u, v, data in G_sub.edges(data=True):
        uid, vid = str(u), str(v)
        pair = (min(uid, vid), max(uid, vid))
        if pair in added_pairs:
            continue
        length_m = float(data.get("length", 0.0))
        dist_km  = length_m / 1000.0
        if dist_km <= 0:
            continue
        # fuel_cost_per_km = 1.0 agar total_cost = jarak km (bukan Rupiah)
        graph.add_edge(uid, vid, dist_km, 1.0, extra_cost=0.0)
        added_pairs.add(pair)

    return graph


def get_available_vehicles(fuel_cfg_path: Path = _FUEL_CFG_PATH) -> dict[str, dict]:
    """
    Kembalikan dictionary semua profil kendaraan kurir yang tersedia.

    Returns:
        dict[str, dict]: Mapping ID kendaraan ke atribut profilnya.
    """
    cfg = _load_fuel_config(fuel_cfg_path)
    return cfg.get("vehicle_profiles", {})


def get_vehicle_label(vehicle: str = "motor_matik_kurir",
                      fuel_cfg_path: Path = _FUEL_CFG_PATH) -> str:
    """
    Kembalikan label kendaraan dan info BBM singkat untuk ditampilkan di output.

    Returns:
        str: Contoh "Motor Matik Honda BeAT (50 km/L, Pertalite Rp 10.000/L)"
    """
    cfg = _load_fuel_config(fuel_cfg_path)
    profile = cfg["vehicle_profiles"].get(vehicle, {})
    label  = profile.get("label", vehicle)
    eff    = profile.get("fuel_efficiency_km_per_liter", "-")
    fuel   = profile.get("fuel_type", "-")
    price  = cfg["active_fuel_prices_idr_per_liter"].get(fuel, "-")
    return f"{label} | {eff} km/L | {fuel.capitalize()} Rp {price:,}/L"


# Daftar sinonim / alias kendaraan untuk pencocokan percakapan chatbot
VEHICLE_SYNONYMS: dict[str, list[str]] = {
    "motor_premium_kurir": [
        "cbr150", "cb150r", "cb150", "vixion", "ninja", "gsx", "cbr",
        "sport", "kopling", "r15", "byson", "verza", "megapro", "tiger", "manual",
    ],
    "mobil_box_diesel": [
        "mobil box l300", "mobil box solar", "box l300", "l300 solar",
        "colt diesel", "mobil box diesel", "mobil diesel", "box diesel",
        "diesel", "l300", "traga", "colt", "canter", "elf", "dutro",
        "truk", "truck", "mobil solar", "biosolar", "solar",
    ],

    "mobil_box_kurir": [
        "mobil pick up", "mobil pickup", "pick-up", "pick up", "pickup",
        "blind van", "mobil bensin", "mobil box", "granmax", "carry",
        "box", "van", "mobil kecil",
    ],
    "motor_bebek_kurir": [
        "motor bebek", "supra x", "revo fit", "jupiter mx", "jupiter z",
        "bebek", "revo", "supra", "jupiter", "blade", "vega", "smash",
        "shogun", "astrea", "karisma",
    ],
    "motor_matik_kurir": [
        "motor matik", "motor matic", "beat karbu", "beat fi", "vario 125",
        "vario 150", "vario 160", "matik", "matic", "vario", "beat",
        "scoopy", "mio", "spacy", "fazzio", "filano", "genio", "lexi",
        "aerox", "nmax", "pcx",
    ],
}


def resolve_vehicle_input(
    user_input: str,
    fuel_cfg_path: Path = _FUEL_CFG_PATH,
) -> str | None:
    """
    Cocokkan input teks pengguna (angka, nama merek, tipe kendaraan)
    ke key profil kendaraan resmi di vehicle_fuel_model.json.

    Args:
        user_input: String input dari pengguna (misal: "vario", "supra", "2").

    Returns:
        str | None: Key kendaraan (misal: "motor_matik_kurir") atau None jika tidak cocok.
    """
    text = user_input.strip().lower()
    vehicles = get_available_vehicles(fuel_cfg_path)

    # 1. Cek kecocokan langsung dengan key resmi
    if text in vehicles:
        return text

    # 2. Cek index urutan (1-based angka tunggal)
    vehicle_keys = list(vehicles.keys())
    tokens = text.split()
    if len(tokens) == 1 and tokens[0].isdigit():
        idx = int(tokens[0]) - 1
        if 0 <= idx < len(vehicle_keys):
            return vehicle_keys[idx]

    # 3. Cek pencocokan sinonim kata kunci menggunakan word boundary
    import re
    # Kumpulkan semua pasangan (keyword, vehicle_key) diurutkan dari keyword terpanjang
    all_pairs: list[tuple[str, str]] = []
    for vehicle_key, keywords in VEHICLE_SYNONYMS.items():
        if vehicle_key in vehicles:
            for kw in keywords:
                all_pairs.append((kw, vehicle_key))

    # Urutkan berdasarkan panjang keyword secara descending
    all_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for kw, vehicle_key in all_pairs:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text):
            return vehicle_key

    return None


AMBIGUOUS_LANDMARKS: dict[str, list[str]] = {
    "del": ["it_del_sitoluama", "sma_unggul_del"],
}


def check_ambiguous_landmark(text: str) -> tuple[str, list[str]] | None:
    """
    Cek apakah input teks mengacu pada nama ambigu yang merujuk ke beberapa lokasi berbeda.
    Misal: 'del' -> IT Del vs SMA Unggul Del.
    Mengembalikan tuple (term, candidates) atau None.
    """
    import re
    clean = text.lower().strip()
    # Jangan tandai ambigu jika sudah spesifik 'it del' atau 'sma del'
    if any(spec in clean for spec in ("it del", "itdel", "institut teknologi", "sma unggul", "sma del", "sud")):
        return None

    # Bersihkan kata awalan seperti 'ke', 'dari', 'antar ke'
    clean_keyword = re.sub(r"^(ke|dari|antar ke|menuju|di)\s+", "", clean).strip()

    if clean_keyword in AMBIGUOUS_LANDMARKS:
        return clean_keyword, AMBIGUOUS_LANDMARKS[clean_keyword]

    # Cek kata tunggal dalam teks
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", clean)
    for token in tokens:
        if token in AMBIGUOUS_LANDMARKS:
            return token, AMBIGUOUS_LANDMARKS[token]

    return None


LANDMARK_SYNONYMS: dict[str, list[str]] = {
    # Laguboti
    "sma_unggul_del": [
        "sma unggul del", "sma unggul", "sma del", "sekolah del", "sud",
    ],
    "it_del_sitoluama": [
        "institut teknologi del", "kampus it del", "kampus del", "it del",
        "sitoluama", "itdel", "asrama del",
    ],
    "hub_jne_laguboti": [
        "kantor pos laguboti", "jne laguboti", "kantor pos", "hub kurir",
        "jne", "pos", "hub", "depo", "gudang",
    ],
    "pasar_laguboti": [
        "pasar tradisional laguboti", "pasar tradisional", "pasar laguboti",
        "pasar", "pajak", "onang",
    ],
    "spbu_laguboti": [
        "spbu laguboti", "spbu pertamina laguboti", "spbu pertamina",
        "pom bensin laguboti", "pom bensin", "spbu",
    ],
    "stasiun_laguboti": [
        "stasiun laguboti", "stasiun", "rel kereta", "rel",
    ],
    "kantor_camat": [
        "kantor camat laguboti", "kantor camat", "camat",
    ],
    "polsek_laguboti": [
        "polsek laguboti", "polsek", "kantor polisi", "polisi",
    ],
    "puskesmas_laguboti": [
        "puskesmas laguboti", "puskesmas", "klinik",
    ],
    "desa_sitoluama": [
        "desa sitoluama",
    ],
    "desa_lumban_bagasan": [
        "desa lumban bagasan", "lumban bagasan",
    ],
    "desa_aruan": [
        "desa aruan", "aruan",
    ],
    "desa_sibuea": [
        "desa sibuea", "sibuea",
    ],
    "simpang_balige_jalan": [
        "simpang jl dr tb silalahi", "simpang tb silalahi", "simpang balige", "arah balige",
    ],

    # Balige
    "pusat_pasar_balige": [
        "pasar balige", "onan balige", "pusat balige", "kota balige", "balige",
    ],
    "rs_hkbp_balige": [
        "rs hkbp balige", "rs hkbp", "rumah sakit hkbp balige", "rumah sakit balige",
    ],
    "simpang_bypass_balige": [
        "bypass balige", "by pass balige", "jalan bypass balige", "jalan by pass balige",
    ],
    "museum_tb_silalahi": [
        "museum tb silalahi center", "tb silalahi center", "museum tb silalahi", "tb silalahi",
    ],
    "pantai_bulbul": [
        "pantai pasir putih lumban bulbul", "pantai lumban bulbul", "pantai bulbul", "lumban bulbul", "bulbul",
    ],
    "kantor_bupati_toba": [
        "kantor bupati toba", "kantor bupati", "bupati toba",
    ],

    # Porsea & Sigumpar
    "pasar_porsea": [
        "pasar tradisional porsea", "pasar porsea", "dermaga porsea", "pelabuhan porsea",
        "kota porsea", "porsea",
    ],
    "simpang_sigumpar": [
        "simpang sigumpar", "sigumpar",
    ],
    "makam_sisingamangaraja": [
        "makam pahlawan sisingamangaraja", "makam sisingamangaraja", "makam sisingamangaraja xii",
    ],

    # Siborong-borong
    "bandara_silangit": [
        "bandara internasional silangit", "bandara silangit", "silangit airport",
        "bandara dtb", "silangit", "airport",
    ],
    "pasar_siborongborong": [
        "pasar tradisional siborongborong", "pasar siborongborong", "pasar siborong-borong",
        "siborongborong", "siborong borong",
    ],
    "jl_dolok_sanggul": [
        "jalan dolok sanggul", "jl dolok sanggul", "dolok sanggul",
    ],

    # Tarutung
    "pusat_kota_tarutung": [
        "pusat kota tarutung", "pasar tradisional tarutung", "pasar tarutung", "kota tarutung", "tarutung",
    ],
    "rsud_tarutung": [
        "rsud swadana tarutung", "rsud tarutung", "rumah sakit tarutung", "rs tarutung",
    ],
    "salib_kasih": [
        "wisata salib kasih", "salib kasih siatas barita", "salib kasih", "siatas barita",
    ],
    "kantor_pusat_hkbp": [
        "kantor pusat hkbp pearaja", "kantor pusat hkbp", "hkbp pearaja", "pearaja",
    ],

    # Hotel
    "hotel_labersa": [
        "hotel labersa", "labersa grand toba", "labersa", "hotel labersa grand",
        "labersa toba", "grand toba",
    ],
    "hotel_niagara": [
        "hotel niagara balige", "hotel niagara", "niagara balige", "niagara",
    ],
    "hotel_pardede": [
        "hotel pardede international", "hotel pardede", "pardede international", "pardede",
    ],

    # Ruas Jalan OSM (Semua jalan nyata di koridor Toba - Taput)
    "jl_sisingamangaraja": [
        "jl. sisingamangaraja", "jalan sisingamangaraja", "jl sisingamangaraja",
        "jalan sm raja", "jl sm raja", "sm raja", "sisingamangaraja",
    ],
    "jl_gereja_balige": [
        "jalan gereja", "jl gereja", "jl. gereja", "jalan gereja balige", "jl gereja balige",
        "gereja balige", "gereja",
    ],
    "jl_hutabulu_mejan": [
        "jalan hutabulu mejan", "jl hutabulu mejan", "jl. hutabulu mejan", "hutabulu mejan", "hutabulu",
    ],
    "simpang_bypass_balige": [
        "jalan bypass balige", "jl bypass balige", "jl. bypass balige", "bypass balige", "by pass balige",
        "jalan bypass", "jl bypass", "bypass",
    ],
    "jl_jambu_balige": [
        "jalan jambu", "jl jambu", "jl. jambu", "jambu balige", "jambu",
    ],
    "jl_uma_rihit": [
        "jalan uma rihit", "jl uma rihit", "jl. uma rihit", "uma rihit", "rihit",
    ],
    "jl_pardolok_tolong": [
        "jalan pardolok tolong", "jl pardolok tolong", "jl. pardolok tolong", "pardolok tolong", "pardolok",
    ],
    "jl_pardomuan_balige": [
        "jalan pardomuan", "jl pardomuan", "jl. pardomuan", "pardomuan balige", "pardomuan",
    ],
    "jl_porsea_balige": [
        "jalan porsea balige", "jalan porsea-balige", "jl porsea balige", "jl. porsea - balige", "porsea balige",
    ],
    "jl_siborongborong_balige": [
        "jalan siborong borong balige", "jalan siborong-borong-balige", "jl siborong borong balige", "siborong borong balige",
    ],
    "jl_lumbanjulu": [
        "jalan lumbanjulu", "jl lumbanjulu", "jl. lumbanjulu", "lumbanjulu", "lumban julu",
    ],
    "jl_bengkel": [
        "jalan bengkel", "jl bengkel", "jl. bengkel", "bengkel",
    ],
    "jl_pahae_tarutung": [
        "jalan pahae", "jl pahae", "jl. pahae", "pahae tarutung", "pahae",
    ],
    "jl_sibolga_tarutung": [
        "jalan sibolga tarutung", "jalan sibolga-tarutung", "jl sibolga tarutung", "sibolga tarutung", "arah sibolga",
    ],
    "jl_parapat_porsea": [
        "jalan parapat porsea", "jalan parapat-porsea", "jl parapat porsea", "parapat porsea", "arah parapat",
    ],
    "jembatan_sungai_asahan": [
        "jembatan sungai asahan", "sungai asahan", "jembatan asahan", "asahan",
    ],
    "jl_lintas_tengah_sumatera": [
        "jalan lintas tengah sumatera", "jl lintas tengah sumatera", "lintas tengah sumatera", "jalinsum",
    ],
    "trans_sumatra_highway": [
        "trans sumatra highway", "trans sumatera highway", "trans sumatra",
    ],

    # Tempat Terkenal & Populer Google Maps
    "mixue_balige": [
        "mixue balige", "mixue", "es krim mixue", "eskrim mixue", "toko mixue", "kedai mixue",
    ],
    "indomaret_balige": [
        "indomaret balige", "indomaret", "indo maret balige", "indo maret",
    ],
    "alfamart_balige": [
        "alfamart balige", "alfamart", "alfa mart balige", "alfa mart",
    ],
    "indomaret_laguboti": [
        "indomaret laguboti", "indomaret del",
    ],
    "alfamart_laguboti": [
        "alfamart laguboti",
    ],
    "hutanta_coffee": [
        "hutanta coffee and resto", "hutanta coffee & resto", "hutanta coffee", "hutanta cafe", "hutanta resto", "hutanta",
    ],
    "tepi_danau_bistro": [
        "tepi danau bistro", "tepi danau cafe", "tepi danau",
    ],
    "lagos_cafe": [
        "lagos cafe", "lago cafe", "pantai sibolahotang", "sibolahotang", "lagos",
    ],
    "ambarado_cafe": [
        "ambarado cafe and resto", "ambarado cafe & resto", "ambarado cafe", "ambarado resto", "ambarado",
    ],
    "rm_bpk_panca": [
        "rm bpk panca", "bpk panca balige", "bpk panca",
    ],
    "rm_bpk_flyover": [
        "rm bpk flyover", "rm bpk fly over", "bpk flyover", "bpk fly over", "fly over laguboti",
    ],
    "spbu_balige": [
        "spbu balige", "spbu pertamina balige", "pom bensin balige", "spbu juanda", "spbu jl sisingamangaraja",
    ],
    "apotek_kimia_farma_balige": [
        "apotek kimia farma balige", "kimia farma balige", "apotek kimia farma", "kimia farma",
    ],
    "bri_balige": [
        "bank bri balige", "bri balige", "bank bri", "bri",
    ],
    "mandiri_balige": [
        "bank mandiri balige", "mandiri balige", "bank mandiri", "mandiri",
    ],
    "bni_balige": [
        "bank bni balige", "bni balige", "bank bni", "bni",
    ],
    "bank_sumut_balige": [
        "bank sumut balige", "bank sumut",
    ],
    "bank_sumut_laguboti": [
        "bank sumut laguboti",
    ],
    "bri_laguboti": [
        "bank bri laguboti", "bri laguboti",
    ],
    "polres_toba": [
        "polres toba", "polres toba samosir", "polres balige", "kantor polisi toba", "polres",
    ],
    "pelabuhan_balige": [
        "pelabuhan balige", "dermaga balige", "pelabuhan kapal balige",
    ],
    "lapangan_sisingamangaraja_balige": [
        "lapangan sisingamangaraja", "alun-alun balige", "alun alun balige", "lapangan balige",
    ],
    "kantor_pos_balige": [
        "kantor pos balige", "pos balige",
    ],
    "sman_1_balige": [
        "sma negeri 1 balige", "sman 1 balige", "sma 1 balige",
    ],
    "sman_2_balige": [
        "sma negeri 2 balige", "sman 2 balige", "sma 2 balige", "asrama soposurung", "soposurung",
    ],
    "gereja_hkbp_balige": [
        "gereja hkbp balige", "hkbp balige kota", "hkbp balige",
    ],
    "rsud_porsea": [
        "rsud porsea", "rumah sakit porsea", "rs porsea",
    ],
    "pantai_parparean": [
        "pantai parparean", "pantai pasir putih parparean", "parparean",
    ],
}

# ---------------------------------------------------------------------------
# Peta nama jalan OSM -> koordinat tengah (rata-rata dari semua segmen)
# Dipakai sebagai fallback resolusi jalan dari graph
# ---------------------------------------------------------------------------
_OSM_STREET_COORDS_CACHE: dict[str, tuple[float, float]] | None = None


def _get_osm_street_coords() -> dict[str, tuple[float, float]]:
    """
    Baca nama jalan unik dari graph OSM dan hitung koordinat tengahnya.
    Di-cache sekali pakai agar tidak berulang kali baca file graphml.
    """
    global _OSM_STREET_COORDS_CACHE
    if _OSM_STREET_COORDS_CACHE is not None:
        return _OSM_STREET_COORDS_CACHE

    result: dict[str, list[tuple[float, float]]] = {}
    try:
        if _TOBA_GRAPHML_PATH.exists():
            G = get_cached_osm_graph(_TOBA_GRAPHML_PATH)
        elif _GRAPHML_PATH.exists():
            G = get_cached_osm_graph(_GRAPHML_PATH)
        else:
            _OSM_STREET_COORDS_CACHE = {}
            return {}

        for u, v, data in G.edges(data=True):
            raw_name = data.get("name")
            if not raw_name:
                continue
            names = raw_name if isinstance(raw_name, list) else [raw_name]
            for name in names:
                if not name or len(name) < 3:
                    continue
                key = name.lower().strip()
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                mid_lat = (float(u_data.get("y", 0)) + float(v_data.get("y", 0))) / 2
                mid_lon = (float(u_data.get("x", 0)) + float(v_data.get("x", 0))) / 2
                if key not in result:
                    result[key] = []
                result[key].append((mid_lat, mid_lon))

        # Rata-ratakan semua segmen ke satu titik tengah representatif
        _OSM_STREET_COORDS_CACHE = {
            k: (
                round(sum(lat for lat, _ in pts) / len(pts), 6),
                round(sum(lon for _, lon in pts) / len(pts), 6),
            )
            for k, pts in result.items()
        }
    except Exception:
        _OSM_STREET_COORDS_CACHE = {}

    return _OSM_STREET_COORDS_CACHE



def resolve_landmark_input(user_input: str) -> str | None:
    """
    Cocokkan teks input lokasi dari pengguna ke key landmark resmi di Laguboti,
    atau koordinat GPS numerik (latitude, longitude).

    Args:
        user_input: Teks nama lokasi atau koordinat (misal "jne", "it del", "2.3832, 99.1486").

    Returns:
        str | None: Kunci landmark, string koordinat "lat,lon", atau None jika belum cocok.
    """
    text = user_input.strip().lower()

    # 1. Cek apakah input berupa koordinat GPS (lat, lon)
    coords = parse_coordinates(text)
    if coords:
        return f"{coords[0]},{coords[1]}"

    # 2. Cek kecocokan langsung dengan key LANDMARKS
    if text in LANDMARKS:
        return text

    # 3. Cek nomor urut landmark
    landmark_keys = list(LANDMARKS.keys())
    tokens = text.split()
    if len(tokens) == 1 and tokens[0].isdigit():
        idx = int(tokens[0]) - 1
        if 0 <= idx < len(landmark_keys):
            return landmark_keys[idx]

    # 4. Kumpulkan semua pasangan (keyword, landmark_key) diurutkan dari terpanjang
    import re
    all_pairs: list[tuple[str, str]] = []
    for landmark_key, keywords in LANDMARK_SYNONYMS.items():
        for kw in keywords:
            all_pairs.append((kw, landmark_key))

    all_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for kw, landmark_key in all_pairs:
        # Bersihkan titik pada abbreviasi jalan ("jl.") agar regex tidak error
        # dan cukup gunakan partial match tanpa word-boundary untuk pola multi-kata
        kw_clean = re.escape(kw)
        if " " in kw or "." in kw:
            # Multi-kata atau pola dengan titik: gunakan substring biasa
            if kw in text:
                return landmark_key
        else:
            pattern = r"\b" + kw_clean + r"\b"
            if re.search(pattern, text):
                return landmark_key

    # 5. Fallback: cari nama jalan langsung dari graph OSM (substring match)
    street_coords = _get_osm_street_coords()
    # Coba berbagai variasi query
    stripped = re.sub(r"^(jl\.?\s+|jalan\s+)", "", text).strip()
    queries = list(dict.fromkeys([text, stripped, "jalan " + stripped]))  # unik, urut

    for q in queries:
        if not q:
            continue
        # Exact match ke nama jalan OSM
        if q in street_coords:
            lat, lon = street_coords[q]
            return f"{lat},{lon}"
        # Substring: query ada di dalam nama jalan OSM, atau sebaliknya
        for street_name, coords in street_coords.items():
            if len(q) >= 4 and (q in street_name or street_name in q):
                lat, lon = coords
                return f"{lat},{lon}"

    return None





# ---------------------------------------------------------------------------
# Helper: Dapatkan node_id dari landmark key
# ---------------------------------------------------------------------------

def get_landmark_node_id(
    landmark_key: str,
    graphml_path: Path | str | None = None,
) -> str:
    """
    Kembalikan node_id (string OSM ID) untuk landmark tertentu.

    Args:
        landmark_key: Kunci dari dict LANDMARKS.
        graphml_path: Path opsional ke GraphML (jika None, dideteksi otomatis).

    Returns:
        str: node_id yang bisa dipakai ke graph.get_node(node_id).

    Raises:
        KeyError: Jika landmark_key tidak ada di LANDMARKS.
    """
    if landmark_key not in LANDMARKS:
        raise KeyError(
            f"Landmark '{landmark_key}' tidak ditemukan. "
            f"Pilihan: {list(LANDMARKS.keys())}"
        )
    lat, lon = LANDMARKS[landmark_key]
    target_graphml = resolve_graph_path(landmark_key, landmark_key, graphml_path)
    G_osm = get_cached_osm_graph(target_graphml)
    osm_id = ox.nearest_nodes(G_osm, X=lon, Y=lat)
    return str(osm_id)
