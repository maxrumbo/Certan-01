"""
test_chatbot.py
---------------
Suite pengujian unit untuk modul Chatbot Asisten Kurir (CourierChatbot).
Memverifikasi:
  1. Ekstraksi entitas kendaraan (sinonim, nama motor, angka).
  2. Ekstraksi entitas lokasi asal dan tujuan.
  3. Alur percakapan bertahap (step-by-step dialogue).
  4. Pengenalan kalimat penuh (one-shot slot filling).
  5. Penanganan perintah khusus (reset, ganti kendaraan, bantuan, kasus batas).
"""

from __future__ import annotations

import pytest

from certan_01.chatbot import CourierChatbot


@pytest.fixture
def bot() -> CourierChatbot:
    """Fixture instance baru CourierChatbot."""
    return CourierChatbot()


# ---------------------------------------------------------------------------
# Test 1: Ekstraksi Slot Kendaraan
# ---------------------------------------------------------------------------

def test_extract_vehicle_by_name(bot: CourierChatbot):
    assert bot.extract_vehicle("saya pakai motor vario 125") == "motor_matik_kurir"
    assert bot.extract_vehicle("bawa beat karbu") == "motor_matik_kurir"
    assert bot.extract_vehicle("motor bebek supra x") == "motor_bebek_kurir"
    assert bot.extract_vehicle("naik honda revo fit") == "motor_bebek_kurir"
    assert bot.extract_vehicle("motor sport cbr150") == "motor_premium_kurir"
    assert bot.extract_vehicle("pakai mobil pick up granmax") == "mobil_box_kurir"
    assert bot.extract_vehicle("bawa mobil box l300 solar") == "mobil_box_diesel"


def test_extract_vehicle_by_number(bot: CourierChatbot):
    assert bot.extract_vehicle("1") == "motor_matik_kurir"
    assert bot.extract_vehicle("2") == "motor_bebek_kurir"
    assert bot.extract_vehicle("3") == "motor_premium_kurir"
    assert bot.extract_vehicle("4") == "mobil_box_kurir"
    assert bot.extract_vehicle("5") == "mobil_box_diesel"


# ---------------------------------------------------------------------------
# Test 2: Ekstraksi Slot Asal & Tujuan
# ---------------------------------------------------------------------------

def test_extract_landmarks_dari_ke(bot: CourierChatbot):
    orig, dest = bot.extract_landmarks("mau antar dari jne ke it del")
    assert orig == "hub_jne_laguboti"
    assert dest == "it_del_sitoluama"


def test_extract_landmarks_ke_dari(bot: CourierChatbot):
    orig, dest = bot.extract_landmarks("kirim paket ke sma unggul del dari pasar")
    assert orig == "pasar_laguboti"
    assert dest == "sma_unggul_del"


def test_extract_landmarks_combined_with_vehicle(bot: CourierChatbot):
    orig, dest = bot.extract_landmarks("antar dari jne laguboti ke it del naik motor bebek")
    assert orig == "hub_jne_laguboti"
    assert dest == "it_del_sitoluama"


# ---------------------------------------------------------------------------
# Test 3: Alur Percakapan Bertahap (Multi-Turn Step-by-Step)
# ---------------------------------------------------------------------------

def test_conversational_step_by_step_flow(bot: CourierChatbot):
    # Turn 1: Sapaan awal
    resp1 = bot.handle_message("Halo selamat pagi")
    assert "kendaraan apa yang kamu pakai" in resp1.lower()
    assert bot.state.vehicle is None

    # Turn 2: Tentukan kendaraan (Motor Bebek)
    resp2 = bot.handle_message("Saya bawa motor Supra")
    assert bot.state.vehicle == "motor_bebek_kurir"
    assert "titik asal" in resp2.lower()

    # Turn 3: Tentukan titik asal
    resp3 = bot.handle_message("JNE Laguboti")
    assert bot.state.origin == "hub_jne_laguboti"
    assert "tujuan" in resp3.lower()

    # Turn 4: Tentukan titik tujuan
    resp4 = bot.handle_message("IT Del")
    assert bot.state.destination == "it_del_sitoluama"
    assert "RUTE PENGIRIMAN PAKET" in resp4
    assert "Motor Bebek" in resp4
    assert "ROUTE HEMAT BBM" in resp4
    assert "Rp" in resp4


def test_select_vehicle_digit_does_not_set_origin(bot: CourierChatbot):
    # User selects vehicle using option number "1"
    resp = bot.handle_message("1")
    assert bot.state.vehicle == "motor_matik_kurir"
    assert bot.state.origin is None  # MUST NOT set origin to landmark 1
    assert "titik asal" in resp.lower()

    # Next turn, user inputs origin as "1" (Hub JNE)
    resp2 = bot.handle_message("1")
    assert bot.state.origin == "hub_jne_laguboti"
    assert bot.state.destination is None  # MUST NOT set destination to landmark 1
    assert "titik tujuan" in resp2.lower()



# ---------------------------------------------------------------------------
# Test 4: One-Shot Sentence (Kalimat Utuh Sekaligus)
# ---------------------------------------------------------------------------

def test_one_shot_sentence(bot: CourierChatbot):
    msg = "Halo, saya bawa motor bebek mau antar dari JNE ke IT Del"
    resp = bot.handle_message(msg)

    assert bot.state.vehicle == "motor_bebek_kurir"
    assert bot.state.origin == "hub_jne_laguboti"
    assert bot.state.destination == "it_del_sitoluama"
    assert "RUTE PENGIRIMAN PAKET" in resp
    assert "Motor Bebek" in resp


# ---------------------------------------------------------------------------
# Test 5: Perintah Khusus & Kasus Batas
# ---------------------------------------------------------------------------

def test_command_help_and_exit(bot: CourierChatbot):
    help_resp = bot.handle_message("bantuan")
    assert "BANTUAN ASISTEN KURIR" in help_resp

    exit_resp = bot.handle_message("keluar")
    assert "Sampai jumpa" in exit_resp


def test_command_reset_and_change_vehicle(bot: CourierChatbot):
    bot.state.vehicle = "motor_matik_kurir"
    bot.state.origin = "hub_jne_laguboti"
    bot.state.destination = "it_del_sitoluama"

    # 'ganti kendaraan' = hapus kendaraan saja
    resp = bot.handle_message("ganti kendaraan")
    assert bot.state.vehicle is None
    assert "ganti kendaraanmu" in resp.lower()

    # 'antar lagi' = hanya hapus rute, kendaraan tetap
    bot.state.vehicle = "motor_bebek_kurir"
    bot.state.origin = "hub_jne_laguboti"
    resp_antar = bot.handle_message("antar lagi")
    assert bot.state.vehicle == "motor_bebek_kurir"   # kendaraan tetap
    assert bot.state.origin is None
    assert bot.state.destination is None
    assert "antar lagi" in resp_antar.lower() or "kendaraan masih" in resp_antar.lower()

    # 'reset' = hapus semua termasuk kendaraan
    bot.state.vehicle = "motor_bebek_kurir"
    resp_reset = bot.handle_message("reset")
    assert bot.state.vehicle is None                  # kendaraan ikut direset
    assert bot.state.origin is None
    assert bot.state.destination is None
    assert "direset penuh" in resp_reset.lower()



def test_same_origin_and_destination(bot: CourierChatbot):
    bot.state.vehicle = "motor_matik_kurir"
    resp = bot.handle_message("antar dari JNE ke JNE")
    assert "sama" in resp.lower()
    assert bot.state.destination is None


def test_unrecognized_location_message(bot: CourierChatbot):
    bot.state.vehicle = "motor_matik_kurir"
    bot.state.origin = "hub_jne_laguboti"

    # User inputs 'x' which is not in the landmark list
    resp = bot.handle_message("x")
    assert "tidak ditemukan" in resp.lower()
    assert bot.state.destination is None


def test_direct_street_tracking_without_prior_vehicle(bot: CourierChatbot):
    # Langsung ketik keyword nama jalan 'gereja' di awal tanpa pilih kendaraan dulu
    resp = bot.handle_message("gereja")
    assert bot.state.vehicle == "motor_matik_kurir"
    assert bot.state.origin == "jl_gereja_balige"
    assert "Jl. Gereja (Balige)" in resp
    assert "titik tujuan" in resp.lower()


def test_direct_street_to_street_route(bot: CourierChatbot):
    # Langsung input rute antar jalan nyata di Balige
    resp = bot.handle_message("antar dari jl sisingamangaraja ke jl gereja")
    assert bot.state.vehicle == "motor_matik_kurir"
    assert bot.state.origin == "jl_sisingamangaraja"
    assert bot.state.destination == "jl_gereja_balige"
    assert "RUTE PENGIRIMAN PAKET" in resp
    assert "Jl. Sisingamangaraja (Balige)" in resp
    assert "Jl. Gereja (Balige)" in resp


# ---------------------------------------------------------------------------
# Test 6: Koridor Regional (Porsea - Balige - Silangit - Tarutung)
# ---------------------------------------------------------------------------

def test_extract_regional_landmarks(bot: CourierChatbot):
    orig, dest = bot.extract_landmarks("antar paket dari Bandara Silangit ke IT Del")
    assert orig == "bandara_silangit"
    assert dest == "it_del_sitoluama"

    orig2, dest2 = bot.extract_landmarks("kirim dari pasar Balige ke Porsea")
    assert orig2 == "pusat_pasar_balige"
    assert dest2 == "pasar_porsea"


def test_regional_route_solving(bot: CourierChatbot):
    # Uji coba pencarian rute regional Balige -> IT Del (Laguboti)
    res = bot.solve_route(
        vehicle="motor_matik_kurir",
        origin="pusat_pasar_balige",
        destination="it_del_sitoluama",
    )
    assert res["astar_fuel"].found is True
    assert res["astar_fuel"].total_distance > 5.0
    assert "maps/dir" in res["gmaps_url"]
    assert "del" in res["gmaps_url"].lower() or "balige" in res["gmaps_url"].lower()


def test_disambiguate_del_destination(bot: CourierChatbot):
    # Step 1: User sets vehicle
    bot.handle_message("1")
    # Step 2: User sets origin as Mixue
    bot.handle_message("mixue")
    assert bot.state.origin == "mixue_balige"

    # Step 3: User inputs 'del' as destination
    resp = bot.handle_message("del")
    assert "pilihan lokasi" in resp.lower()
    assert "Institut Teknologi Del" in resp
    assert "SMA Unggul Del" in resp
    assert bot.state.destination is None
    assert bot.state.pending_clarification is not None

    # Step 4: User selects option 1 (IT Del)
    resp2 = bot.handle_message("1")
    assert bot.state.destination == "it_del_sitoluama"
    assert bot.state.pending_clarification is None
    assert "RUTE PENGIRIMAN PAKET" in resp2
    assert "Mixue Balige" in resp2
    assert "Institut Teknologi Del" in resp2
    assert "origin=Mixue+Balige" in resp2
    assert "destination=Institut+Teknologi+Del" in resp2


def test_disambiguate_del_to_sma_unggul_del(bot: CourierChatbot):
    bot.handle_message("1")
    bot.handle_message("mixue")
    bot.handle_message("del")
    resp = bot.handle_message("2")
    assert bot.state.destination == "sma_unggul_del"
    assert "SMA Unggul Del" in resp


def test_gmaps_url_place_names(bot: CourierChatbot):
    res = bot.solve_route("motor_matik_kurir", "mixue_balige", "it_del_sitoluama")
    assert "maps/dir" in res["gmaps_url"]
    assert "origin=Mixue+Balige" in res["gmaps_url"]
    assert "destination=Institut+Teknologi+Del" in res["gmaps_url"]


def test_chatbot_search_command(bot: CourierChatbot):
    resp = bot.handle_message("cari balige")
    assert "Hasil pencarian" in resp
    assert "Balige" in resp

    resp2 = bot.handle_message("cari gereja")
    assert "Hasil pencarian" in resp2
    assert "Gereja" in resp2


def test_chatbot_street_input(bot: CourierChatbot):
    orig, dest = bot.extract_landmarks("antar dari jl sisingamangaraja ke it del")
    assert orig == "jl_sisingamangaraja"
    assert dest == "it_del_sitoluama"


def test_chatbot_popular_poi_mixue_and_indomaret(bot: CourierChatbot):
    # Test pengenalan Mixue Balige dan Indomaret
    bot.state.vehicle = "motor_matik_kurir"
    bot.state.origin = "it_del_sitoluama"
    resp = bot.handle_message("mixue")
    assert bot.state.destination == "mixue_balige"
    assert "Mixue Balige" in resp

    # Test pencarian dengan 'cari mixue'
    resp_search = bot.handle_message("cari mixue")
    assert "Mixue Balige" in resp_search


