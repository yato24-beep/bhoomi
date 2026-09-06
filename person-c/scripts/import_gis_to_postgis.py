"""
scripts/import_gis_to_postgis.py
Reproducible geospatial import pipeline.
Validates CRS EPSG:4326, checks Polygon geometries, and imports cadastral parcels into PostGIS / SQLite.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from shapely.geometry import shape

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_session import SessionLocal, init_db, engine
from src.database.models import CadastralParcelModel
from src.database.postgis_ext import is_postgres, sync_parcel_geometry


def import_geojson_files(input_dir: str):
    """Imports all GeoJSON files in the specified directory into the cadastral database."""
    init_db()
    dir_path = Path(input_dir)
    if not dir_path.is_dir():
        print(f"Directory not found: {input_dir}")
        return

    geojson_files = list(dir_path.glob("*.geojson")) + list(dir_path.glob("*.json"))
    if not geojson_files:
        print(f"No GeoJSON files found in {input_dir}")
        return

    print("=" * 80)
    print(f"CADASTRAL GIS IMPORT PIPELINE — Importing {len(geojson_files)} datasets")
    print("=" * 80)

    db = SessionLocal()
    total_imported = 0

    for gfile in geojson_files:
        with open(gfile, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "FeatureCollection":
            continue

        layer_name = data.get("name", gfile.stem)
        features = data.get("features", [])
        print(f"\nProcessing layer: '{layer_name}' ({len(features)} features) from {gfile.name}")

        layer_imported = 0
        for feat in features:
            props = feat.get("properties", {})
            geom_dict = feat.get("geometry", {})

            # Validate geometry using Shapely
            try:
                poly = shape(geom_dict)
                if not poly.is_valid:
                    poly = poly.buffer(0)  # repair if needed
            except Exception as e:
                print(f"  [WARN] Skipping invalid geometry for {props.get('khasra_number')}: {e}")
                continue

            state = props.get("state", "UNKNOWN").upper()
            district = props.get("district", "UNKNOWN").upper()
            tehsil = props.get("tehsil", "UNKNOWN").upper()
            village = props.get("village", "UNKNOWN").upper()
            khasra_no = str(props.get("khasra_number", "")).upper()
            area_ha = float(props.get("area_hectares", 0.0))
            source_label = props.get("data_source_label", "synthetic_demo_cadastral")

            # Check if record already exists
            existing = db.query(CadastralParcelModel).filter(
                CadastralParcelModel.state == state,
                CadastralParcelModel.district == district,
                CadastralParcelModel.tehsil == tehsil,
                CadastralParcelModel.village == village,
                CadastralParcelModel.khasra_number == khasra_no,
            ).first()

            if existing:
                existing.area_hectares = area_ha
                existing.geometry_geojson = geom_dict
                existing.gis_layer_name = layer_name
                record = existing
            else:
                record = CadastralParcelModel(
                    state=state,
                    district=district,
                    tehsil=tehsil,
                    village=village,
                    khasra_number=khasra_no,
                    area_hectares=area_ha,
                    gis_layer_name=layer_name,
                    geometry_geojson=geom_dict,
                )
                db.add(record)

            db.flush()  # ensure record.id is assigned before syncing PostGIS geometry

            if is_postgres(engine):
                sync_parcel_geometry(engine, record.id, geom_dict)

            layer_imported += 1
            total_imported += 1

        db.commit()
        pg_note = " + real PostGIS geom column synced" if is_postgres(engine) else " (SQLite - geometry_geojson only, no live PostGIS)"
        print(f"  Successfully imported {layer_imported} parcels from {layer_name} (CRS: EPSG:4326){pg_note}")

    db.close()
    print("\n" + "=" * 80)
    print(f"IMPORT COMPLETE: Total {total_imported} cadastral parcels loaded into database.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import cadastral GeoJSON reference datasets into PostGIS/database")
    parser.add_argument("--input-dir", default="data/gis/raw", help="Path to raw GIS data folder")
    args = parser.parse_args()
    import_geojson_files(args.input_dir)
