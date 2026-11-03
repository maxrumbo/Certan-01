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
    # --- Wilayah Laguboti (Presisi Jalan Nyata < 0.8 meter dari OSM Graph) ---
    "hub_jne_laguboti"    : (2.36644, 99.12925),   # Hub JNE / Kantor Pos Laguboti (Node 6310150906)
    "pasar_laguboti"      : (2.36675, 99.13033),   # Pasar Tradisional Laguboti (Node 6310150918)
    "it_del_sitoluama"    : (2.38329, 99.14852),   # Institut Teknologi Del Sitoluama (Gerbang Utama Node 6312201325)
    "sma_unggul_del"      : (2.38879, 99.15068),   # SMA Unggul Del (Gerbang Utama Node 6312199629)
    "spbu_laguboti"       : (2.36159, 99.13381),   # SPBU Pertamina Laguboti Lintas Sumatera (Node 6310150731)
    "kantor_camat"        : (2.36444, 99.12919),   # Kantor Camat Laguboti (Node 6310150783)
    "polsek_laguboti"     : (2.36775, 99.13045),   # Polsek Laguboti (Node 6310150763)
    "puskesmas_laguboti"  : (2.36804, 99.12927),   # Puskesmas Laguboti (Node 6310153633)
    "desa_sitoluama"      : (2.38437, 99.14614),   # Desa Sitoluama Akses Jalan (Node 7702147437)
    "desa_lumban_bagasan" : (2.36997, 99.13049),   # Desa Lumban Bagasan Laguboti (Node 6310150287)
    "simpang_balige_jalan": (2.35522, 99.12652),   # Simpang Jl. Dr. TB Silalahi (Node 5108027374)
    "desa_aruan"          : (2.35654, 99.12627),   # Desa Aruan Laguboti (Node 6310152167)
    "desa_sibuea"         : (2.35752, 99.12679),   # Desa Sibuea Laguboti (Node 6310146963)
    "indomaret_laguboti"  : (2.36665, 99.12960),   # Indomaret Laguboti Pasar (Node 6310150916)
    "alfamart_laguboti"   : (2.36758, 99.13202),   # Alfamart Laguboti (Node 6310150772)
    "bank_sumut_laguboti" : (2.36644, 99.12925),   # Bank Sumut Capem Laguboti (Node 6310150906)
    "bri_laguboti"        : (2.36578, 99.12877),   # Bank BRI Unit Laguboti (Node 6310150890)
    "rm_bpk_flyover"      : (2.36963, 99.13043),   # RM BPK Fly Over Laguboti (Node 6310151791)
    "ambarado_cafe"       : (2.38461, 99.14606),   # Ambarado Cafe & Resto Sitoluama (Node 6312199293)
    "gereja_hkbp_laguboti": (2.36719, 99.13259),   # Gereja HKBP Laguboti Kota (Node 7708927025)
    "smp_negeri_1_laguboti": (2.36480, 99.13100),  # SMP Negeri 1 Laguboti (Node 6310150779)
    "sman_1_laguboti"     : (2.36775, 99.13045),   # SMA Negeri 1 Laguboti (Node 6310150763)
    "klinik_del_sitoluama": (2.38316, 99.14955),   # Klinik Pratama Del Sitoluama (Node 6312199091)
    "kantor_pos_laguboti" : (2.36644, 99.12925),   # Kantor Pos Indonesia Laguboti (Node 6310150906)
    "stasiun_laguboti"    : (2.36719, 99.13259),   # Area Terminal / Eks Stasiun Laguboti (Node 7708927025)

    # --- Wilayah Balige (Ibukota Kab. Toba) ---
    "pusat_pasar_balige"   : (2.3338, 99.0645),   # Pusat Pasar Tradisional Onan Balige
    "mixue_balige"         : (2.3340, 99.0665),   # Mixue Balige (Jl. Sisingamangaraja No. 59)
    "indomaret_balige"     : (2.3338, 99.0652),   # Indomaret Balige (Jl. Sisingamangaraja)
    "alfamart_balige"      : (2.3341, 99.0660),   # Alfamart Balige (Jl. Sisingamangaraja)
    "rs_hkbp_balige"       : (2.3340, 99.0670),   # RS HKBP Balige (Jl. Gereja No. 17)
    "gereja_hkbp_balige"   : (2.3325, 99.0668),   # Gereja HKBP Balige Kota (Jl. Gereja)
    "simpang_bypass_balige": (2.3248, 99.0400),   # Simpang Jalur By Pass Balige
    "museum_tb_silalahi"   : (2.3488, 99.0832),   # Museum TB Silalahi Center Pagar Batu
    "pantai_bulbul"        : (2.3465, 99.0845),   # Pantai Pasir Putih Lumban Bulbul Balige
    "kantor_bupati_toba"   : (2.3310, 99.0610),   # Kantor Bupati Toba Balige
    "polres_toba"          : (2.3220, 99.0600),   # Polres Toba Balige (Jl. Bypass)
    "hotel_labersa"        : (2.3480, 99.0880),   # Hotel Labersa Grand Toba & Convention Center
    "hotel_niagara"        : (2.3310, 99.0640),   # Hotel Niagara Balige
    "hotel_pardede"        : (2.3365, 99.0730),   # Hotel Pardede International Balige
    "spbu_balige"          : (2.3370, 99.0680),   # SPBU Pertamina Balige (Juanda)
    "pelabuhan_balige"     : (2.3385, 99.0655),   # Pelabuhan / Dermaga Kapal Balige
    "lapangan_sisingamangaraja_balige": (2.3335, 99.0635), # Lapangan Sisingamangaraja Balige (Alun-alun)
    "kantor_pos_balige"    : (2.3339, 99.0653),   # Kantor Pos Balige
    "bri_balige"           : (2.3336, 99.0642),   # Bank BRI Kanca Balige
    "mandiri_balige"       : (2.3337, 99.0648),   # Bank Mandiri Balige
    "bni_balige"           : (2.3340, 99.0655),   # Bank BNI Balige
    "bank_sumut_balige"    : (2.3334, 99.0638),   # Bank Sumut Balige
    "apotek_kimia_farma_balige": (2.3340, 99.0658), # Apotek Kimia Farma Balige
    "sman_1_balige"        : (2.3385, 99.0740),   # SMA Negeri 1 Balige
    "sman_2_balige"        : (2.3410, 99.0760),   # SMA Negeri 2 Balige (Soposurung)
    "hutanta_coffee"       : (2.3420, 99.0750),   # Hutanta Coffee & Resto Balige
    "tepi_danau_bistro"    : (2.3430, 99.0760),   # Tepi Danau Bistro Balige
    "lagos_cafe"           : (2.3440, 99.0750),   # Lago's Cafe Balige (Pantai Sibolahotang)
    "rm_bpk_panca"         : (2.3350, 99.0650),   # RM BPK Panca Balige

    # --- Wilayah Porsea & Sigumpar (Toba) ---
    "pasar_porsea"         : (2.4462, 99.1448),   # Pusat Pasar / Dermaga Porsea
    "rsud_porsea"          : (2.4475, 99.1465),   # RSUD Porsea
    "simpang_sigumpar"     : (2.4048, 99.1665),   # Simpang Sigumpar (Porsea - Laguboti)
    "makam_sisingamangaraja": (2.4048, 99.1665),  # Makam Pahlawan Raja Sisingamangaraja XII (Sigumpar)
    "pantai_parparean"     : (2.4510, 99.1520),   # Pantai Pasir Putih Parparean Porsea
    "jembatan_sungai_asahan": (2.4423, 99.1566),  # Jembatan Sungai Asahan Porsea

    # --- Wilayah Siborong-borong (Tapanuli Utara) ---
    "bandara_silangit"     : (2.2608, 98.9917),   # Bandara Internasional Silangit (DTB)
    "pasar_siborongborong" : (2.2135, 98.9723),   # Pusat Pasar Siborong-borong
    "jl_dolok_sanggul"     : (2.2192, 98.9543),   # Jl. Dolok Sanggul - Siborong-borong

    # --- Wilayah Tarutung (Ibukota Kab. Tapanuli Utara) ---
    "pusat_kota_tarutung"  : (2.0235, 98.9667),   # Pusat Pasar & Alun-alun Tarutung
    "rsud_tarutung"        : (2.0285, 98.9615),   # RSUD Swadana Tarutung
    "salib_kasih"          : (2.0080, 98.9780),   # Wisata Salib Kasih Siatas Barita
    "kantor_pusat_hkbp"    : (2.0250, 98.9720),   # Kantor Pusat HKBP Pearaja Tarutung

    # --- Ruas Jalan OSM Representatif ---
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
    "jl_lintas_tengah_sumatera": (2.0049, 98.9906),  # Jl. Lintas Tengah Sumatera
    "trans_sumatra_highway"   : (2.3538, 99.0954),   # Trans Sumatra Highway
}

LANDMARK_LABELS: dict[str, str] = {
    # Laguboti
    "hub_jne_laguboti"        : "Hub JNE / Kantor Pos Laguboti",
    "it_del_sitoluama"        : "Institut Teknologi Del (Sitoluama)",
    "sma_unggul_del"          : "SMA Unggul Del",
    "simpang_balige_jalan"    : "Simpang Jl. Dr. TB. Silalahi",
    "pasar_laguboti"          : "Pasar Tradisional Laguboti",
    "spbu_laguboti"           : "SPBU Pertamina Laguboti (Lintas)",
    "stasiun_laguboti"        : "Area Terminal / Eks Stasiun Laguboti",
    "kantor_camat"            : "Kantor Camat Laguboti",
    "polsek_laguboti"         : "Polsek Laguboti",
    "puskesmas_laguboti"      : "Puskesmas Laguboti",
    "desa_sitoluama"          : "Desa Sitoluama",
    "desa_lumban_bagasan"     : "Desa Lumban Bagasan",
    "desa_aruan"              : "Desa Aruan (Laguboti)",
    "desa_sibuea"             : "Desa Sibuea (Laguboti)",
    "indomaret_laguboti"      : "Indomaret Laguboti",
    "alfamart_laguboti"       : "Alfamart Laguboti",
    "bank_sumut_laguboti"     : "Bank Sumut Capem Laguboti",
    "bri_laguboti"            : "Bank BRI Unit Laguboti",
    "rm_bpk_flyover"          : "RM BPK Fly Over Laguboti",
    "ambarado_cafe"           : "Ambarado Cafe & Resto Laguboti",
    "gereja_hkbp_laguboti"    : "Gereja HKBP Laguboti Kota",
    "smp_negeri_1_laguboti"   : "SMP Negeri 1 Laguboti",
    "sman_1_laguboti"         : "SMA Negeri 1 Laguboti",
    "klinik_del_sitoluama"    : "Klinik Pratama Del Sitoluama",
    "kantor_pos_laguboti"     : "Kantor Pos Indonesia Laguboti",

    # Balige
    "pusat_pasar_balige"      : "Pusat Pasar Balige (Onan)",
    "rs_hkbp_balige"          : "Rumah Sakit HKBP Balige",
    "simpang_bypass_balige"   : "Simpang Jalur By Pass Balige",
    "museum_tb_silalahi"      : "Museum TB Silalahi Center",
    "pantai_bulbul"           : "Pantai Lumban Bulbul Balige",
    "kantor_bupati_toba"      : "Kantor Bupati Toba (Balige)",
    "polres_toba"             : "Polres Toba (Balige)",
    "hotel_labersa"           : "Hotel Labersa Grand Toba & Convention",
    "hotel_niagara"           : "Hotel Niagara Balige",
    "hotel_pardede"           : "Hotel Pardede International (Balige)",
    "spbu_balige"             : "SPBU Pertamina Balige (Juanda)",
    "pelabuhan_balige"        : "Pelabuhan / Dermaga Kapal Balige",
    "lapangan_sisingamangaraja_balige": "Lapangan Sisingamangaraja (Alun-alun Balige)",
    "kantor_pos_balige"       : "Kantor Pos Balige",
    "bri_balige"              : "Bank BRI Cabang Balige",
    "mandiri_balige"          : "Bank Mandiri Balige",
    "bni_balige"              : "Bank BNI Balige",
    "bank_sumut_balige"       : "Bank Sumut Balige",
    "apotek_kimia_farma_balige": "Apotek Kimia Farma Balige",
    "mixue_balige"            : "Mixue Balige (Jl. Sisingamangaraja)",
    "indomaret_balige"        : "Indomaret Balige (Jl. Sisingamangaraja)",
    "alfamart_balige"         : "Alfamart Balige",
    "sman_1_balige"           : "SMA Negeri 1 Balige",
    "sman_2_balige"           : "SMA Negeri 2 Balige (Soposurung)",
    "gereja_hkbp_balige"      : "Gereja HKBP Balige Kota",
    "hutanta_coffee"          : "Hutanta Coffee & Resto Balige",
    "tepi_danau_bistro"       : "Tepi Danau Bistro Balige",
    "lagos_cafe"              : "Lago's Cafe Balige",
    "rm_bpk_panca"            : "RM BPK Panca Balige",

    # Porsea & Sigumpar
    "pasar_porsea"            : "Pusat Pasar & Dermaga Porsea",
    "rsud_porsea"             : "RSUD Porsea",
    "simpang_sigumpar"        : "Simpang Sigumpar (Porsea)",
    "makam_sisingamangaraja"  : "Makam Sisingamangaraja XII (Sigumpar)",
    "pantai_parparean"        : "Pantai Pasir Putih Parparean (Porsea)",
    "jembatan_sungai_asahan"  : "Jembatan Sungai Asahan (Porsea)",

    # Siborong-borong
    "bandara_silangit"        : "Bandara Internasional Silangit (DTB)",
    "pasar_siborongborong"    : "Pusat Pasar Siborong-borong",
    "jl_dolok_sanggul"        : "Jl. Dolok Sanggul - Siborong-borong",

    # Tarutung
    "pusat_kota_tarutung"     : "Pusat Kota / Pasar Tarutung",
    "rsud_tarutung"           : "RSUD Tarutung",
    "salib_kasih"             : "Wisata Salib Kasih Siatas Barita",
    "kantor_pusat_hkbp"       : "Kantor Pusat HKBP Pearaja Tarutung",

    # Ruas Jalan OSM (Titik representatif jalan)
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
    "jl_lintas_tengah_sumatera": "Jl. Lintas Tengah Sumatera",
    "trans_sumatra_highway"   : "Trans Sumatra Highway",
}

# ---------------------------------------------------------------------------
# Metadata Detail Landmark untuk Visualisasi & Rekomendasi Rute
# ---------------------------------------------------------------------------

LANDMARK_DETAILS: dict[str, dict[str, Any]] = {
    # Laguboti
    "hub_jne_laguboti": {
        "name": "Hub JNE / Kantor Pos Laguboti",
        "category": "Logistik & Ekspedisi",
        "cat_key": "publik",
        "icon": "fa-solid fa-truck-fast",
        "color": "#2563EB",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Depo utama logistik kurir, pusat sortir paket dan drop point pengiriman wilayah Laguboti.",
        "courier_note": "Akses bongkar muat motor & mobil lancar. Tersedia zona sortir paket dan parkir kurir.",
    },
    "it_del_sitoluama": {
        "name": "Institut Teknologi Del (Sitoluama)",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Jl. Sisingamangaraja, Sitoluama, Laguboti",
        "description": "Kampus perguruan tinggi teknologi terkemuka di tepian Danau Toba dengan ribuan sivitas akademika.",
        "courier_note": "Penerimaan paket terpusat di Pos Keamanan Gerbang Utama (Security Gate). Parkir motor kurir tersedia.",
    },
    "sma_unggul_del": {
        "name": "SMA Unggul Del",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Desa Sitoluama, Laguboti",
        "description": "Sekolah menengah atas berasrama unggulan nasional dengan sistem kedisiplinan ketat.",
        "courier_note": "Pengantaran paket drop di Pos Satpam Depan. Hubungi penerima via telepon sebelum gerbang.",
    },
    "pasar_laguboti": {
        "name": "Pasar Tradisional Onan Laguboti",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Pasar, Laguboti",
        "description": "Pusat transaksi niaga komoditas pangan, kebutuhan harian, dan sandang masyarakat Laguboti.",
        "courier_note": "Sangat ramai pada hari pekan (onan). Sebaiknya kurir memarkir motor di tepi jalan utama.",
    },
    "spbu_laguboti": {
        "name": "SPBU Pertamina 14.223.328 Laguboti",
        "category": "Stasiun Pengisian BBM",
        "cat_key": "spbu",
        "icon": "fa-solid fa-gas-pump",
        "color": "#B91C1C",
        "address": "Jl. Lintas Sumatera, Laguboti",
        "description": "Stasiun pengisian bahan bakar resmi Pertamina (Pertalite, Pertamax, Solar) operasional 24 jam.",
        "courier_note": "Titik pengisian BBM strategis armada kurir di koridor lintas Laguboti-Porsea.",
    },
    "kantor_camat": {
        "name": "Kantor Camat Laguboti",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-landmark",
        "color": "#475569",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Pusat administrasi pemerintahan tingkat kecamatan dan pelayanan terpadu kependudukan warga.",
        "courier_note": "Drop paket di loket resepsionis / bagian persuratan kantor. Jam layanan 08:00 - 16:00 WIB.",
    },
    "polsek_laguboti": {
        "name": "Polsek Laguboti",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-shield-halved",
        "color": "#475569",
        "address": "Jl. Dr. TB. Silalahi, Laguboti",
        "description": "Markas kepolisian sektor pengamanan wilayah hukum dan ketertiban masyarakat Laguboti.",
        "courier_note": "Akses pengantaran 24 jam via pos penjagaan Sentra Pelayanan Kepolisian Terpadu (SPKT).",
    },
    "puskesmas_laguboti": {
        "name": "Puskesmas Laguboti",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-hospital",
        "color": "#DC2626",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Pusat kesehatan masyarakat rujukan utama kecamatan dengan layanan IGD dan rawat jalan.",
        "courier_note": "Akses ambulans harus tetap steril. Parkir motor kurir di area samping gedung pendaftaran.",
    },
    "klinik_del_sitoluama": {
        "name": "Klinik Pratama Del Sitoluama",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-hospital",
        "color": "#DC2626",
        "address": "Jl. Pematang Siantar, Sitoluama, Laguboti",
        "description": "Layanan fasilitas kesehatan pratama bagi sivitas akademika kampus Del dan warga Sitoluama.",
        "courier_note": "Pengantaran paket medis atau dokumen langsung ke meja administrasi klinik.",
    },
    "desa_sitoluama": {
        "name": "Kawasan Pemukiman Desa Sitoluama",
        "category": "Kawasan Pemukiman",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-location-dot",
        "color": "#64748B",
        "address": "Sitoluama, Laguboti",
        "description": "Kawasan hunian dan kos-kosan mahasiswa IT Del dengan jalan beraspal mulus.",
        "courier_note": "Banyak lorong hunian mahasiswa. Konfirmasi nama kos dan nomor kamar via WhatsApp.",
    },
    "desa_lumban_bagasan": {
        "name": "Desa Lumban Bagasan",
        "category": "Kawasan Pemukiman",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-location-dot",
        "color": "#64748B",
        "address": "Lumban Bagasan, Laguboti",
        "description": "Kawasan perumahan warga dan sentra kerajinan tenun kain Ulos tradisional Batak.",
        "courier_note": "Jalan lingkungan selebar 4 meter. Akses motor kurir sangat leluasa.",
    },
    "simpang_balige_jalan": {
        "name": "Simpang Tiga Dr. TB. Silalahi",
        "category": "Kawasan Pemukiman & Simpang",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-signs-post",
        "color": "#64748B",
        "address": "Simpang Jl. Dr. TB. Silalahi, Laguboti",
        "description": "Persimpangan strategis yang menghubungkan jalan poros lintas Sumatera dengan bypass Silalahi.",
        "courier_note": "Titik temu arus lalu lintas padat pagi dan sore. Perhatikan lampu lalu lintas / arus kendaraan.",
    },
    "desa_aruan": {
        "name": "Desa Aruan (Laguboti)",
        "category": "Kawasan Pemukiman",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-location-dot",
        "color": "#64748B",
        "address": "Desa Aruan, Laguboti",
        "description": "Kawasan pemukiman warga agraris di koridor selatan Laguboti arah Balige.",
        "courier_note": "Akses jalan aspal penghubung desa. Lalu lintas tenang dan minim hambatan.",
    },
    "desa_sibuea": {
        "name": "Desa Sibuea (Laguboti)",
        "category": "Kawasan Pemukiman",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-location-dot",
        "color": "#64748B",
        "address": "Desa Sibuea, Laguboti",
        "description": "Pemukiman penduduk di koridor transisi perbatasan Laguboti menuju Balige.",
        "courier_note": "Jalan lingkar desa mulus, mudah dilalui sepeda motor kurir.",
    },
    "indomaret_laguboti": {
        "name": "Indomaret Lintas Laguboti",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Minimarket retail modern dengan fasilitas ATM dan area parkir kendaraan luas di tepi jalan utama.",
        "courier_note": "Drop point pengantaran paket retail dan belanja harian warga.",
    },
    "alfamart_laguboti": {
        "name": "Alfamart Laguboti Kota",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Gerai retail modern kebutuhan sehari-hari dengan parkir motor kurir yang praktis.",
        "courier_note": "Parkir depan gerai nyaman untuk sepeda motor kurir melakukan serah terima paket.",
    },
    "bank_sumut_laguboti": {
        "name": "Bank Sumut Capem Laguboti",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Kantor cabang pembantu Bank Pembangunan Daerah Sumatera Utara dengan gerai ATM tunai.",
        "courier_note": "Penerimaan paket dokumen atau surat dinas melalui bagian administrasi lantai 1.",
    },
    "bri_laguboti": {
        "name": "Bank BRI Unit Laguboti",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Kantor unit perbankan BUMN BRI dan jaringan ATM setor-tarik tunai 24 jam.",
        "courier_note": "Serah terima paket langsung ke petugas sekuriti depan atau customer service.",
    },
    "rm_bpk_flyover": {
        "name": "RM BPK Fly Over Laguboti",
        "category": "Kuliner & Restoran",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-utensils",
        "color": "#D97706",
        "address": "Jl. Lintas Sumatera, Laguboti",
        "description": "Restoran kuliner khas tradisional Batak populer tepi jalan lintas Sumatera.",
        "courier_note": "Pesanan makanan siap saji / take away tinggi pada jam makan siang (11:30 - 14:00 WIB).",
    },
    "ambarado_cafe": {
        "name": "Ambarado Cafe & Resto",
        "category": "Kuliner & Kafe",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-utensils",
        "color": "#D97706",
        "address": "Sitoluama, Laguboti",
        "description": "Kafe dan tempat santap kuliner modern favorit mahasiswa kampus IT Del dan warga sekitar.",
        "courier_note": "Titik pengantaran makanan dan pesanan paket kurir yang sering diakses sore hingga malam.",
    },
    "gereja_hkbp_laguboti": {
        "name": "Gereja HKBP Laguboti Kota",
        "category": "Tempat Ibadah",
        "cat_key": "ibadah",
        "icon": "fa-solid fa-church",
        "color": "#0284C7",
        "address": "Jl. Gereja, Laguboti",
        "description": "Kompleks gereja Kristen Protestan historis pusat resort HKBP Laguboti dengan arsitektur klasik.",
        "courier_note": "Pengantaran surat dinas gereja atau paket pendeta drop di kantor sekretariat gereja.",
    },
    "smp_negeri_1_laguboti": {
        "name": "SMP Negeri 1 Laguboti",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Sekolah menengah pertama negeri rujukan di Laguboti dengan ratusan siswa dan tenaga pendidik.",
        "courier_note": "Drop paket di ruang tata usaha (TU) atau pos satpam saat jam sekolah (07:30 - 15:00 WIB).",
    },
    "sman_1_laguboti": {
        "name": "SMA Negeri 1 Laguboti",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Sekolah menengah atas negeri utama kecamatan Laguboti.",
        "courier_note": "Paket diserahkan ke pos penjagaan gerbang utama sekolah.",
    },
    "kantor_pos_laguboti": {
        "name": "Kantor Pos Indonesia Laguboti",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-box",
        "color": "#2563EB",
        "address": "Jl. Sisingamangaraja, Laguboti",
        "description": "Kantor pos BUMN penyedia jasa pengiriman surat kilat khusus, logistik, dan layanan wesel pos.",
        "courier_note": "Mitra logistik ekspedisi kurir resmi.",
    },
    "stasiun_laguboti": {
        "name": "Area Terminal / Eks Stasiun Laguboti",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-van-shuttle",
        "color": "#475569",
        "address": "Jl. Gereja, Laguboti",
        "description": "Titik transit pemberhentian angkutan pedesaan dan kendaraan umum rute antar-kecamatan.",
        "courier_note": "Area perputaran kendaraan angkutan umum. Cukup ramai saat pagi hari.",
    },

    # Balige
    "pusat_pasar_balige": {
        "name": "Pusat Pasar Tradisional Onan Balige",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Sisingamangaraja / Jl. Melati, Balige",
        "description": "Pasar terbesar di Kabupaten Toba dengan deretan balairung berornamen gorga Batak yang ikonik.",
        "courier_note": "Sangat padat terutama pada hari pasar (Jumat). Disarankan kurir masuk lewat koridor tepi barat.",
    },
    "mixue_balige": {
        "name": "Mixue Ice Cream & Tea Balige",
        "category": "Kuliner & Kafe",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-ice-cream",
        "color": "#D97706",
        "address": "Jl. Sisingamangaraja No. 59, Balige",
        "description": "Gerai minuman segar dan es krim populer di pusat pertokoan kota Balige dengan pesanan delivery tinggi.",
        "courier_note": "Area pick-up kurir cepat di depan kasir. Parkir motor kurir tersedia tepat di depan ruko.",
    },
    "indomaret_balige": {
        "name": "Indomaret Balige Pusat",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Minimarket retail modern pusat kota Balige.",
        "courier_note": "Akses praktis untuk serah terima paket kurir e-commerce.",
    },
    "alfamart_balige": {
        "name": "Alfamart Balige Juanda",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Minimarket retail modern dengan area parkir motor luas.",
        "courier_note": "Titik antar belanja harian dan paket drop-off.",
    },
    "rs_hkbp_balige": {
        "name": "Rumah Sakit HKBP Balige",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-hospital",
        "color": "#DC2626",
        "address": "Jl. Gereja No. 17, Balige",
        "description": "Rumah sakit rujukan utama terlengkap di Kabupaten Toba dengan fasilitas IGD 24 jam dan rawat inap.",
        "courier_note": "Akses dua arah. Pengantaran paket ke loket informasi lobi utama gedung.",
    },
    "gereja_hkbp_balige": {
        "name": "Gereja HKBP Balige Kota",
        "category": "Tempat Ibadah",
        "cat_key": "ibadah",
        "icon": "fa-solid fa-church",
        "color": "#0284C7",
        "address": "Jl. Gereja, Balige",
        "description": "Kompleks gereja bersejarah dengan menara megah di pusat jantung kota Balige.",
        "courier_note": "Pengantaran surat dinas gereja atau paket diserahkan ke kantor sekretariat distrik.",
    },
    "simpang_bypass_balige": {
        "name": "Simpang Jalur By Pass Balige",
        "category": "Kawasan Pemukiman & Simpang",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-signs-post",
        "color": "#64748B",
        "address": "Jalur Lingkar Luar By Pass, Balige",
        "description": "Jalur arteri lingkar luar pengurai kepadatan lalu lintas pusat pasar tradisional Balige.",
        "courier_note": "Jalan mulus dan lebar, kecepatan rata-rata kendaraan lebih tinggi dibanding jalur kota.",
    },
    "museum_tb_silalahi": {
        "name": "Museum TB Silalahi Center",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-camera",
        "color": "#0D9488",
        "address": "Jl. Pagar Batu No. 88, Balige",
        "description": "Pusat konservasi kebudayaan Batak dan museum militer jejak sejarah pahlawan di tepian Danau Toba.",
        "courier_note": "Drop paket kurir di loket resepsionis gerbang masuk wisata.",
    },
    "pantai_bulbul": {
        "name": "Pantai Pasir Putih Lumban Bulbul",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-umbrella-beach",
        "color": "#0D9488",
        "address": "Desa Lumban Bulbul, Balige",
        "description": "Objek wisata pantai pasir putih air tawar Danau Toba terpopuler di Balige.",
        "courier_note": "Ramai akhir pekan. Pengantaran ke kafe/pondok pantai dapat langsung ke lokasi pemilik.",
    },
    "kantor_bupati_toba": {
        "name": "Kantor Bupati Toba (Balige)",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-landmark",
        "color": "#475569",
        "address": "Jl. Sutomo No. 1, Balige",
        "description": "Kompleks pusat perkantoran pemerintahan Kabupaten Toba.",
        "courier_note": "Penerimaan berkas resmi dan paket logistik wajib melalui bagian Protokol / Bagian Umum.",
    },
    "polres_toba": {
        "name": "Polres Toba (Balige)",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-shield-halved",
        "color": "#475569",
        "address": "Jl. By Pass, Balige",
        "description": "Markas komando kepolisian resor penjagaan kamtibmas Kabupaten Toba.",
        "courier_note": "Akses 24 jam via penjagaan gerbang utama SPKT.",
    },
    "hotel_labersa": {
        "name": "Hotel Labersa Grand Toba & Convention",
        "category": "Akomodasi & Hotel",
        "cat_key": "hotel",
        "icon": "fa-solid fa-hotel",
        "color": "#2563EB",
        "address": "Jl. Sariburaja Pagar Batu, Balige",
        "description": "Hotel bintang 4 mewah dan gedung konvensi terbesar di kawasan pesisir Danau Toba Balige.",
        "courier_note": "Pengantaran paket tamu atau staf drop di meja Front Desk / Concierge hotel.",
    },
    "hotel_niagara": {
        "name": "Hotel Niagara Balige",
        "category": "Akomodasi & Hotel",
        "cat_key": "hotel",
        "icon": "fa-solid fa-hotel",
        "color": "#2563EB",
        "address": "Jl. Sutomo, Balige",
        "description": "Hotel penginapan strategis di pusat kota Balige.",
        "courier_note": "Titik antar di meja resepsionis lobi depan hotel.",
    },
    "hotel_pardede": {
        "name": "Hotel Pardede International Balige",
        "category": "Akomodasi & Hotel",
        "cat_key": "hotel",
        "icon": "fa-solid fa-hotel",
        "color": "#2563EB",
        "address": "Jl. Pardede Pasir, Balige",
        "description": "Resor hotel legendaris tepi danau dengan pemandangan langsung ke perairan Danau Toba.",
        "courier_note": "Akses motor kurir mudah, pengantaran paket via meja resepsionis.",
    },
    "spbu_balige": {
        "name": "SPBU Pertamina Balige (Juanda)",
        "category": "Stasiun Pengisian BBM",
        "cat_key": "spbu",
        "icon": "fa-solid fa-gas-pump",
        "color": "#B91C1C",
        "address": "Jl. Ir. H. Juanda, Balige",
        "description": "Stasiun pengisian bahan bakar kendaraan di gerbang masuk barat kota Balige.",
        "courier_note": "Titik isi BBM armada kurir sebelum melakukan pengantaran jarak jauh.",
    },
    "pelabuhan_balige": {
        "name": "Pelabuhan Kapal Balige (Dermaga Mulia Raja)",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-ship",
        "color": "#475569",
        "address": "Dermaga Mulia Raja Balige",
        "description": "Dermaga penyeberangan kapal motor wisata dan logistik perairan Danau Toba menuju Pulau Samosir.",
        "courier_note": "Serah terima kargo paket antar-pulau di pos dermaga pelabuhan.",
    },
    "lapangan_sisingamangaraja_balige": {
        "name": "Lapangan Sisingamangaraja (Alun-alun Balige)",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-landmark",
        "color": "#0D9488",
        "address": "Pusat Kota Balige",
        "description": "Alun-alun terbuka pusat kegiatan olahraga, festival budaya, dan upacara kenegaraan warga Balige.",
        "courier_note": "Pusat keramaian warga sore hari. Parkir motor di sekeliling trotoar lapangan.",
    },
    "kantor_pos_balige": {
        "name": "Kantor Pos Indonesia Balige",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-box",
        "color": "#2563EB",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Kantor pos sentra distribusi logistik utama Kabupaten Toba.",
        "courier_note": "Pusat transit antar-cabang ekspedisi paket kilat.",
    },
    "bri_balige": {
        "name": "Bank BRI Cabang Balige",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Kantor cabang utama perbankan BUMN BRI di Kabupaten Toba.",
        "courier_note": "Paket diserahkan ke bagian persuratan lantai 1 melalui pos satpam.",
    },
    "mandiri_balige": {
        "name": "Bank Mandiri Balige",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Kantor bank BUMN Mandiri dengan layanan perbankan dan ATM tunai.",
        "courier_note": "Penerimaan berkas kurir via customer service / pos pengamanan.",
    },
    "bni_balige": {
        "name": "Bank BNI Balige",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Kantor bank BUMN BNI di pusat kota Balige.",
        "courier_note": "Penerimaan paket melalui pos sekuriti lobi bank.",
    },
    "bank_sumut_balige": {
        "name": "Bank Sumut Cabang Balige",
        "category": "Layanan Perbankan & ATM",
        "cat_key": "bank",
        "icon": "fa-solid fa-building-columns",
        "color": "#059669",
        "address": "Jl. Sisingamangaraja, Balige",
        "description": "Kantor cabang Bank Pembangunan Daerah Sumatera Utara di Balige.",
        "courier_note": "Serah terima dokumen kurir via bagian umum.",
    },
    "apotek_kimia_farma_balige": {
        "name": "Apotek Kimia Farma Balige",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-pills",
        "color": "#DC2626",
        "address": "Jl. Sisingamangaraja No. 62, Balige",
        "description": "Apotek farmasi resmi penyedia resep obat dan perlengkapan medis terpercaya.",
        "courier_note": "Pengantaran obat atau pesanan farmasi langsung ke meja apoteker.",
    },
    "sman_1_balige": {
        "name": "SMA Negeri 1 Balige",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Jl. Gereja Soposurung, Balige",
        "description": "Sekolah menengah atas negeri unggulan di Balige.",
        "courier_note": "Pengantaran paket drop di pos jaga satpam gerbang sekolah.",
    },
    "sman_2_balige": {
        "name": "SMA Negeri 2 Balige (Soposurung)",
        "category": "Pendidikan & Kampus",
        "cat_key": "pendidikan",
        "icon": "fa-solid fa-graduation-cap",
        "color": "#7C3AED",
        "address": "Kompleks Asrama Soposurung, Balige",
        "description": "Sekolah berasrama unggulan dengan prestasi akademik nasional.",
        "courier_note": "Paket siswa dan staf wajib diserahkan ke pos pengamanan asrama gerbang utama.",
    },
    "hutanta_coffee": {
        "name": "Hutanta Coffee & Resto Balige",
        "category": "Kuliner & Kafe",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-mug-hot",
        "color": "#D97706",
        "address": "Soposurung, Balige",
        "description": "Kafe modern dengan racikan kopi Arabika Toba istimewa dan suasana estetik.",
        "courier_note": "Titik pick-up dan delivery pesanan kuliner populer bagi kalangan muda.",
    },
    "tepi_danau_bistro": {
        "name": "Tepi Danau Bistro Balige",
        "category": "Kuliner & Restoran",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-utensils",
        "color": "#D97706",
        "address": "Tepi Danau Toba, Balige",
        "description": "Restoran tepi danau dengan panorama memukau Danau Toba dan menu nusantara.",
        "courier_note": "Pesanan makanan siap antar di area kasir utama.",
    },
    "lagos_cafe": {
        "name": "Lago's Cafe Balige",
        "category": "Kuliner & Kafe",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-mug-hot",
        "color": "#D97706",
        "address": "Pantai Sibolahotang, Balige",
        "description": "Kafe santai ruang terbuka dengan pemandangan semilir angin Danau Toba.",
        "courier_note": "Akses jalan wisata lancar untuk sepeda motor kurir.",
    },
    "rm_bpk_panca": {
        "name": "RM BPK Panca Balige",
        "category": "Kuliner & Restoran",
        "cat_key": "kuliner",
        "icon": "fa-solid fa-utensils",
        "color": "#D97706",
        "address": "Jl. Gereja, Balige",
        "description": "Destinasi kuliner tradisional khas Batak favorit masyarakat lokal.",
        "courier_note": "Pesanan antar ramai pada jam makan siang.",
    },

    # Porsea & Wilayah Sekitar
    "pasar_porsea": {
        "name": "Pusat Pasar & Dermaga Porsea",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Jl. Patuan Nagari, Porsea",
        "description": "Pasar tradisional dan dermaga muara Sungai Asahan sentra ekonomi kawasan timur Toba.",
        "courier_note": "Hari pekan pasar sangat ramai, disarankan parkir di jalan penghubung luar.",
    },
    "rsud_porsea": {
        "name": "RSUD Porsea",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-hospital",
        "color": "#DC2626",
        "address": "Jl. Parapat, Porsea",
        "description": "Rumah sakit umum daerah penyedia fasilitas rawat inap dan IGD wilayah timur Toba.",
        "courier_note": "Penerimaan paket via resepsionis informasi gedung rawat jalan.",
    },
    "simpang_sigumpar": {
        "name": "Simpang Sigumpar (Porsea)",
        "category": "Kawasan Pemukiman & Simpang",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-signs-post",
        "color": "#64748B",
        "address": "Sigumpar, Toba",
        "description": "Persimpangan utama rute Porsea menuju Laguboti.",
        "courier_note": "Titik penyeberangan jalan antar-kecamatan.",
    },
    "makam_sisingamangaraja": {
        "name": "Makam Sisingamangaraja XII (Sigumpar)",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-landmark",
        "color": "#0D9488",
        "address": "Kecamatan Sigumpar, Toba",
        "description": "Kompleks pemakaman Pahlawan Nasional Raja Sisingamangaraja XII yang dikeramatkan.",
        "courier_note": "Situs cagar budaya nasional.",
    },
    "pantai_parparean": {
        "name": "Pantai Pasir Putih Parparean (Porsea)",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-umbrella-beach",
        "color": "#0D9488",
        "address": "Desa Parparean II, Porsea",
        "description": "Objek rekreasi pesisir pantai pasir putih alami di teluk Danau Toba.",
        "courier_note": "Akses jalan beton mulus hingga pintu gerbang objek wisata.",
    },
    "jembatan_sungai_asahan": {
        "name": "Jembatan Sungai Asahan (Porsea)",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-bridge",
        "color": "#475569",
        "address": "Porsea, Toba",
        "description": "Infrastruktur jembatan strategis hulu Sungai Asahan penghubung lintas Sumatera.",
        "courier_note": "Jembatan dua jalur utama lalu lintas antarkota.",
    },

    # Siborong-borong & Tarutung
    "bandara_silangit": {
        "name": "Bandara Internasional Silangit (DTB)",
        "category": "Instansi & Layanan Publik",
        "cat_key": "publik",
        "icon": "fa-solid fa-plane-departure",
        "color": "#2563EB",
        "address": "Siborong-borong, Tapanuli Utara",
        "description": "Bandara internasional gerbang penerbangan utama Kawasan Strategis Pariwisata Danau Toba.",
        "courier_note": "Penerimaan kargo paket di terminal kargo bandara (Gate Cargo Logistik).",
    },
    "pasar_siborongborong": {
        "name": "Pusat Pasar Siborong-borong",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Siborong-borong, Tapanuli Utara",
        "description": "Pusat transaksi niaga ternak dan komoditas kuliner ombus-ombus dataran tinggi Silindung.",
        "courier_note": "Sangat padat pada hari pekan pasar tradisional.",
    },
    "jl_dolok_sanggul": {
        "name": "Jl. Dolok Sanggul - Siborong-borong",
        "category": "Kawasan Pemukiman & Simpang",
        "cat_key": "wilayah",
        "icon": "fa-solid fa-road",
        "color": "#64748B",
        "address": "Siborong-borong",
        "description": "Jalan penghubung poros utama Kabupaten Humbang Hasundutan dan Tapanuli Utara.",
        "courier_note": "Jalan lintas pegunungan beraspal halus.",
    },
    "pusat_kota_tarutung": {
        "name": "Pusat Kota / Pasar Tarutung",
        "category": "Perbelanjaan & Niaga",
        "cat_key": "niaga",
        "icon": "fa-solid fa-store",
        "color": "#EA580C",
        "address": "Tarutung, Tapanuli Utara",
        "description": "Ibukota Kabupaten Tapanuli Utara, pusat niaga dan denyut ekonomi Lembah Silindung.",
        "courier_note": "Akses jalan perkotaan ramai dengan ruko-ruko perdagangan.",
    },
    "rsud_tarutung": {
        "name": "RSUD Swadana Tarutung",
        "category": "Fasilitas Kesehatan",
        "cat_key": "kesehatan",
        "icon": "fa-solid fa-hospital",
        "color": "#DC2626",
        "address": "Jl. Sisingamangaraja, Tarutung",
        "description": "Rumah sakit rujukan tipe B terlengkap di kawasan Tapanuli Bagian Utara.",
        "courier_note": "Akses IGD 24 jam dan lobi utama pendaftaran pasien.",
    },
    "salib_kasih": {
        "name": "Wisata Salib Kasih Siatas Barita",
        "category": "Destinasi Wisata & Budaya",
        "cat_key": "wisata",
        "icon": "fa-solid fa-camera",
        "color": "#0D9488",
        "address": "Siatas Barita, Tarutung",
        "description": "Monumen wisata rohani bersejarah Dr. I.L. Nommensen di puncak bukit pinus dengan pemandangan spektakuler.",
        "courier_note": "Akses jalan menanjak beraspal. Pengantaran paket drop di pos karcis gerbang bawah.",
    },
    "kantor_pusat_hkbp": {
        "name": "Kantor Pusat HKBP Pearaja Tarutung",
        "category": "Tempat Ibadah",
        "cat_key": "ibadah",
        "icon": "fa-solid fa-church",
        "color": "#0284C7",
        "address": "Pearaja, Tarutung",
        "description": "Kantor pusat Ephorus dan pimpinan sinode gereja HKBP sedunia.",
        "courier_note": "Pengantaran berkas dinas drop di meja administrasi kantor pusat sinode.",
    },
}


def get_landmark_info(key: str) -> dict[str, Any]:
    """
    Kembalikan data metadata lengkap dari suatu landmark.
    Jika tidak ada di kamus LANDMARK_DETAILS, lakukan derivasi otomatis berbasis kategori.
    """
    if key in LANDMARK_DETAILS:
        return LANDMARK_DETAILS[key]

    label = LANDMARK_LABELS.get(key, key.replace("_", " ").title())
    k_lower = key.lower()

    if any(w in k_lower for w in ("del", "sma", "smp", "sekolah", "kampus")):
        return {
            "name": label,
            "category": "Pendidikan & Kampus",
            "cat_key": "pendidikan",
            "icon": "fa-solid fa-graduation-cap",
            "color": "#7C3AED",
            "address": "Kawasan Pendidikan Toba",
            "description": "Lembaga pendidikan formal dan sarana belajar terpadu.",
            "courier_note": "Penerimaan paket via pos satpam gerbang utama.",
        }
    elif any(w in k_lower for w in ("rs_", "rsud", "puskesmas", "klinik", "apotek")):
        return {
            "name": label,
            "category": "Fasilitas Kesehatan",
            "cat_key": "kesehatan",
            "icon": "fa-solid fa-hospital",
            "color": "#DC2626",
            "address": "Fasilitas Layanan Medis",
            "description": "Pusat pelayanan kesehatan dan penanganan medis masyarakat.",
            "courier_note": "Akses darurat steril, parkir motor kurir tersedia.",
        }
    elif any(w in k_lower for w in ("pasar", "mixue", "indomaret", "alfamart")):
        return {
            "name": label,
            "category": "Perbelanjaan & Niaga",
            "cat_key": "niaga",
            "icon": "fa-solid fa-store",
            "color": "#EA580C",
            "address": "Pusat Perbelanjaan & Niaga",
            "description": "Pusat jual beli barang kebutuhan dan aktivitas perdagangan.",
            "courier_note": "Parkir motor kurir di depan gerai pertokoan.",
        }
    elif any(w in k_lower for w in ("spbu", "pertamina")):
        return {
            "name": label,
            "category": "Stasiun Pengisian BBM",
            "cat_key": "spbu",
            "icon": "fa-solid fa-gas-pump",
            "color": "#B91C1C",
            "address": "Stasiun SPBU",
            "description": "Fasilitas pengisian bahan bakar resmi kendaraan bermotor.",
            "courier_note": "Titik pengisian bahan bakar armada kurir operasional.",
        }
    elif any(w in k_lower for w in ("bank", "bri", "bni", "mandiri", "sumut")):
        return {
            "name": label,
            "category": "Layanan Perbankan & ATM",
            "cat_key": "bank",
            "icon": "fa-solid fa-building-columns",
            "color": "#059669",
            "address": "Kantor Perbankan",
            "description": "Layanan transaksi keuangan dan jaringan ATM tunai.",
            "courier_note": "Serah terima paket melalui pos pengamanan atau resepsionis.",
        }
    elif any(w in k_lower for w in ("hotel", "penginapan")):
        return {
            "name": label,
            "category": "Akomodasi & Hotel",
            "cat_key": "hotel",
            "icon": "fa-solid fa-hotel",
            "color": "#2563EB",
            "address": "Akomodasi Penginapan",
            "description": "Fasilitas hotel dan penginapan pengunjung.",
            "courier_note": "Titik serah terima di meja resepsionis lobi hotel.",
        }
    elif any(w in k_lower for w in ("cafe", "kopi", "coffee", "bistro", "bpk", "resto")):
        return {
            "name": label,
            "category": "Kuliner & Restoran",
            "cat_key": "kuliner",
            "icon": "fa-solid fa-utensils",
            "color": "#D97706",
            "address": "Destinasi Kuliner",
            "description": "Tempat bersantap makanan dan minuman khas daerah.",
            "courier_note": "Pesanan makanan siap saji atau delivery.",
        }
    elif any(w in k_lower for w in ("gereja", "hkbp")):
        return {
            "name": label,
            "category": "Tempat Ibadah",
            "cat_key": "ibadah",
            "icon": "fa-solid fa-church",
            "color": "#0284C7",
            "address": "Kompleks Peribadatan",
            "description": "Sarana ibadah dan pembinaan rohani jemaat.",
            "courier_note": "Pengantaran surat dinas ke kantor sekretariat gereja.",
        }
    elif any(w in k_lower for w in ("pantai", "museum", "salib", "makam", "wisata", "pelabuhan", "lapangan")):
        return {
            "name": label,
            "category": "Destinasi Wisata & Budaya",
            "cat_key": "wisata",
            "icon": "fa-solid fa-camera",
            "color": "#0D9488",
            "address": "Objek Wisata Daerah",
            "description": "Destinasi rekreasi panorama alam dan warisan budaya Toba.",
            "courier_note": "Drop paket di loket informasi atau gerbang masuk wisata.",
        }
    else:
        return {
            "name": label,
            "category": "Kawasan Wilayah",
            "cat_key": "wilayah",
            "icon": "fa-solid fa-location-dot",
            "color": "#64748B",
            "address": "Wilayah Toba",
            "description": "Titik lokasi jalan dan pemukiman terverifikasi di peta OSM.",
            "courier_note": "Akses jalan aspal terhubung ke jaringan rute kurir.",
        }



def parse_coordinates(text: str) -> tuple[float, float] | None:
    """
    Ekstrak pasangan koordinat (latitude, longitude) dari teks input.
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
    "desa_aruan", "desa_sibuea", "indomaret_laguboti", "alfamart_laguboti",
    "bank_sumut_laguboti", "bri_laguboti", "rm_bpk_flyover", "ambarado_cafe",
    "gereja_hkbp_laguboti", "smp_negeri_1_laguboti", "sman_1_laguboti",
    "klinik_del_sitoluama", "kantor_pos_laguboti",
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

        # x=lon, y=lat sesuai konvensi models.py (heuristik Great-Circle / Euclidean)
        graph.add_node(Node(node_id, name, x=lon, y=lat, is_hub=is_hub, is_goal=is_goal, is_geo=True))

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
        graph.add_node(Node(node_id, name, x=lon, y=lat, is_hub=is_hub, is_goal=is_goal, is_geo=True))

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

    # Tempat Terkenal & Fasilitas Penting Koridor Toba
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
    "gereja_hkbp_laguboti": [
        "gereja hkbp laguboti", "hkbp laguboti kota", "hkbp laguboti",
    ],
    "smp_negeri_1_laguboti": [
        "smp negeri 1 laguboti", "smp 1 laguboti", "smpn 1 laguboti",
    ],
    "sman_1_laguboti": [
        "sma negeri 1 laguboti", "sma 1 laguboti", "sman 1 laguboti",
    ],
    "klinik_del_sitoluama": [
        "klinik del", "klinik pratama del", "klinik del sitoluama", "klinik it del",
    ],
    "kantor_pos_laguboti": [
        "kantor pos laguboti", "pos laguboti",
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
