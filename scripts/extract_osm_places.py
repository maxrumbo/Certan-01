"""
scripts/extract_osm_places.py
-----------------------------
Ekstrak seluruh lokasi bernama (POI, fasilitas umum, kantor desa, kantor camat,
tempat ibadah, sekolah, toko, rumah sakit, tempat wisata, dan nama jalan)
dari OpenStreetMap koridor Toba - Tapanuli Utara ke file lokal:
data/osm_places_toba_taput.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import osmnx as ox


def extract_all_osm_locations() -> None:
    root = Path(__file__).resolve().parents[1]
    bbox = (98.9201, 1.9901, 99.1800, 2.4797)  # (min_lon, min_lat, max_lon, max_lat)

    # Konfigurasi OSMnx Overpass
    ox.settings.timeout = 180

    tags = {
        "amenity": True,
        "tourism": True,
        "shop": True,
        "office": True,
        "place": ["town", "village", "suburb", "hamlet"],
    }

    endpoints = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    gdf = None
    for ep in endpoints:
        try:
            print(f"Mencoba Overpass endpoint: {ep} ...")
            ox.settings.overpass_endpoint = ep
            gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
            if gdf is not None and len(gdf) > 0:
                print(f"Berhasil mengunduh {len(gdf)} fitur dari {ep}")
                break
        except Exception as e:
            print(f"Gagal dari {ep}: {e}")

    places: list[dict] = []
    seen: set[tuple[str, float, float]] = set()

    if gdf is not None and not gdf.empty:
        named = gdf[gdf["name"].notnull()].copy()
        named["lat"] = named.geometry.centroid.y
        named["lon"] = named.geometry.centroid.x

        for _, row in named.iterrows():
            name = str(row["name"]).strip()
            if not name or name.lower() == "nan":
                continue
            lat = round(float(row["lat"]), 6)
            lon = round(float(row["lon"]), 6)
            key = (name.lower(), round(lat, 3), round(lon, 3))
            if key in seen:
                continue
            seen.add(key)

            cat = "umum"
            amenity = str(row.get("amenity", ""))
            office = str(row.get("office", ""))
            tourism = str(row.get("tourism", ""))
            shop = str(row.get("shop", ""))
            place = str(row.get("place", ""))

            if amenity and amenity != "nan":
                cat = amenity
            elif office and office != "nan":
                cat = f"kantor_{office}"
            elif tourism and tourism != "nan":
                cat = f"wisata_{tourism}"
            elif shop and shop != "nan":
                cat = f"toko_{shop}"
            elif place and place != "nan":
                cat = f"wilayah_{place}"

            places.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "category": cat,
            })

    # Ekstrak semua nama jalan terdaftar dari toba_taput_network.graphml
    graph_files = [
        root / "data" / "toba_taput_network.graphml",
        root / "data" / "laguboti_network.graphml",
    ]
    for gf in graph_files:
        if gf.exists():
            print(f"Mengekstrak nama jalan dari {gf.name}...")
            G = ox.load_graphml(gf)
            for u, v, k, d in G.edges(keys=True, data=True):
                rname = d.get("name")
                if rname:
                    rnames = rname if isinstance(rname, list) else [rname]
                    for rn in rnames:
                        rn = str(rn).strip()
                        if not rn or rn.lower() == "nan":
                            continue
                        u_lat = G.nodes[u]["y"]
                        u_lon = G.nodes[u]["x"]
                        key = (rn.lower(), round(u_lat, 3), round(u_lon, 3))
                        if key not in seen:
                            seen.add(key)
                            places.append({
                                "name": rn,
                                "lat": round(float(u_lat), 6),
                                "lon": round(float(u_lon), 6),
                                "category": "jalan",
                            })

    places.sort(key=lambda p: p["name"].lower())
    out_path = root / "data" / "osm_places_toba_taput.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(places, f, indent=2, ensure_ascii=False)

    print(f"SELESAI! Berhasil menyimpan {len(places)} lokasi ke {out_path}")


if __name__ == "__main__":
    extract_all_osm_locations()
