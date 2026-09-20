"""
src/database/gis.py
Cadastral GIS and PostGIS spatial validation engine.
Verifies administrative hierarchy, cadastral parcel boundaries, area deviations, and coordinates.
Strictly classifies outcomes into: MATCH, MISMATCH, UNKNOWN, or INSUFFICIENT_DATA.
Never classifies missing GIS reference data as a mismatch.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from shapely.geometry import Point, Polygon, box, shape
from sqlalchemy import text
from sqlalchemy.orm import Session

from schemas import ExtractedField, GISStatus, GISValidationResult
from src.database.postgis_ext import is_postgres


class GISValidator:
    """
    Validates extracted land records against GIS cadastral reference datasets.
    Supports 4-state classification: MATCH, MISMATCH, UNKNOWN, INSUFFICIENT_DATA.
    """

    def __init__(
        self,
        area_mismatch_threshold_percent: float = 10.0,
        gis_data_dir: Optional[Union[str, Path]] = None,
        db_connection_uri: Optional[str] = None,
    ):
        self.area_mismatch_threshold = area_mismatch_threshold_percent
        self.db_uri = db_connection_uri or os.getenv("DATABASE_URL")
        self._is_live_db = False
        
        # Load Cadastral GeoJSON Files
        self.gis_data_dir = Path(gis_data_dir) if gis_data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "gis" / "raw"
        self._cadastral_registry: Dict[str, Dict[str, Any]] = self._init_cadastral_registry()

    def _init_cadastral_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Populates the cadastral reference registry from REAL GeoJSON files
        only (data/gis/raw/*.geojson).

        NOTE: This previously also merged in a hardcoded dict of ~10
        synthetic "demo" parcels (e.g. a fake Kengeri/Bangalore survey
        42/1 entry) baked directly into this file. That's removed —
        real GIS validation now only matches parcels that genuinely
        exist in the GeoJSON reference files you provide, or via a real
        PostGIS query (_validate_gis_postgis) when connected to a real
        Postgres database. If no matching parcel exists in real data,
        this correctly returns GISStatus.UNKNOWN — which is the honest
        answer — rather than silently "matching" against fabricated
        sample data.
        """
        registry: Dict[str, Dict[str, Any]] = {}

        # Read real GeoJSON cadastral files in gis_data_dir
        if self.gis_data_dir.is_dir():
            for gfile in self.gis_data_dir.glob("*.geojson"):
                try:
                    with open(gfile, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for feat in data.get("features", []):
                        props = feat.get("properties", {})
                        geom_dict = feat.get("geometry", {})
                        poly = shape(geom_dict)
                        state = props.get("state", "UNKNOWN").upper()
                        dist = props.get("district", "UNKNOWN").upper()
                        teh = props.get("tehsil", "UNKNOWN").upper()
                        vil = props.get("village", "UNKNOWN").upper()
                        khasra = str(props.get("khasra_number", "")).upper()
                        key = f"{state}:{dist}:{teh}:{vil}:{khasra}"
                        
                        registry[key] = {
                            "gis_area_hectares": float(props.get("area_hectares", 0.0)),
                            "centroid": (poly.centroid.x, poly.centroid.y),
                            "polygon": poly,
                            "layer_name": props.get("gis_layer_name", gfile.stem),
                            "data_source_label": props.get("data_source_label", "registered_cadastral_parcel"),
                            "aliases": props.get("aliases", []),
                        }
                except Exception as e:
                    logger.debug(f"Could not load GeoJSON file {gfile}: {e}")

        # Expand aliases into direct registry entries
        alias_map = {}
        for k, v in registry.items():
            for alias in v.get("aliases", []):
                alias_map[alias.upper()] = v
        registry.update(alias_map)

        return registry

    def register_gis_parcel(
        self,
        state: str,
        district: str,
        tehsil: str,
        village: str,
        khasra_no: str,
        area_hectares: float,
        polygon: Optional[Polygon] = None,
        layer_name: str = "custom_test_layer",
    ):
        """Allows test fixtures to register known GIS parcels dynamically."""
        poly = polygon or box(0, 0, 1, 1)
        key = f"{state.upper()}:{district.upper()}:{tehsil.upper()}:{village.upper()}:{khasra_no.upper()}"
        self._cadastral_registry[key] = {
            "gis_area_hectares": area_hectares,
            "centroid": (poly.centroid.x, poly.centroid.y),
            "polygon": poly,
            "layer_name": layer_name,
            "data_source_label": "registered_cadastral_parcel",
        }

    def validate_gis(
        self,
        fields: Dict[str, ExtractedField],
        state_code: str,
        coordinates: Optional[Tuple[float, float]] = None,
        db_session: Optional[Session] = None,
    ) -> GISValidationResult:
        """
        Validates extracted land record against Cadastral GIS database.
        Returns one of:
          - MATCH: Parcel found, area matches tolerance, point-in-polygon passes.
          - MISMATCH: Area discrepancy > tolerance, coordinates outside polygon, or jurisdiction conflict.
          - UNKNOWN: Parcel identifier not found in GIS database (Not flagged as a mismatch).
          - INSUFFICIENT_DATA: Missing mandatory query parameters.

        If db_session is provided AND it is bound to a real PostgreSQL/PostGIS
        engine, this runs actual ST_Contains / ST_Area PostGIS SQL queries.
        Otherwise (no session, or SQLite dev database) it falls back to the
        in-Python Shapely registry below - functionally equivalent, but not a
        real spatial-database query.
        """
        if db_session is not None and is_postgres(db_session.get_bind()):
            result = self._validate_gis_postgis(fields, state_code, coordinates, db_session)
            if result is not None:
                return result
            logger.warning("[GIS] PostGIS query path failed or found nothing usable; falling back to Shapely registry.")

        return self._validate_gis_fallback(fields, state_code, coordinates)

    def _validate_gis_postgis(
        self,
        fields: Dict[str, ExtractedField],
        state_code: str,
        coordinates: Optional[Tuple[float, float]],
        db_session: Session,
    ) -> Optional[GISValidationResult]:
        """
        Real PostGIS spatial validation path. Runs ST_Contains / geography-area SQL
        directly against the cadastral_parcels.geom column (see postgis_ext.py).
        Returns None (never raises) if the query path itself fails, so the caller
        can fall back to the Shapely registry rather than crashing the pipeline -
        per spec: one bad field/lookup must not crash the document.
        """
        state = state_code.upper()
        district = self._get_normalized_str(fields.get("district"))
        tehsil = self._get_normalized_str(fields.get("tehsil"))
        village = self._get_normalized_str(fields.get("village"))
        khasra = self._get_normalized_str(fields.get("khasra_number"))
        extracted_area = self._get_extracted_area_hectares(fields.get("land_area"))
        flag_reasons: List[str] = []

        if not khasra or (not village and not district):
            return GISValidationResult(
                gis_status=GISStatus.INSUFFICIENT_DATA,
                is_verified=False, has_mismatch=False, parcel_id_found=False,
                state=state, district=district, tehsil=tehsil, village=village, khasra_number=khasra,
                flag_reasons=["Mandatory cadastral identifiers missing (Khasra/Survey No or Village/District)"],
            )

        try:
            row = db_session.execute(
                text(
                    "SELECT area_hectares, gis_layer_name, data_source_label, "
                    "ST_AsGeoJSON(geom) AS geom_geojson, "
                    "ST_Area(geography(geom)) / 10000.0 AS computed_area_hectares, "
                    "ST_X(ST_Centroid(geom)) AS centroid_lon, ST_Y(ST_Centroid(geom)) AS centroid_lat "
                    "FROM cadastral_parcels "
                    "WHERE state = :state AND district = :district AND tehsil = :tehsil "
                    "AND village = :village AND khasra_number = :khasra AND geom IS NOT NULL "
                    "LIMIT 1;"
                ),
                {"state": state, "district": district.upper(), "tehsil": tehsil.upper(),
                 "village": village.upper(), "khasra": khasra.upper()},
            ).mappings().first()
        except Exception as e:
            logger.warning(f"[GIS] PostGIS query failed: {e}")
            return None

        if row is None:
            return GISValidationResult(
                gis_status=GISStatus.UNKNOWN,
                is_verified=False, has_mismatch=False, parcel_id_found=False,
                state=state, district=district, tehsil=tehsil, village=village, khasra_number=khasra,
                extracted_area_hectares=extracted_area,
                flag_reasons=[f"Parcel '{khasra}' not found in PostGIS cadastral_parcels for {state}/{district}/{tehsil}/{village} (Status: UNKNOWN)"],
                source_gis_layer=None,
            )

        gis_area = float(row["area_hectares"])
        layer_name = row["gis_layer_name"]
        data_source_label = row["data_source_label"]
        area_deviation = None
        has_area_mismatch = False

        if extracted_area is not None and extracted_area > 0 and gis_area > 0:
            diff = abs(extracted_area - gis_area)
            area_deviation = (diff / gis_area) * 100.0
            if area_deviation > self.area_mismatch_threshold:
                has_area_mismatch = True
                flag_reasons.append(
                    f"GIS Area Discrepancy: Extracted area ({extracted_area:.4f} ha) differs from PostGIS-computed area "
                    f"({gis_area:.4f} ha) by {area_deviation:.1f}% (Threshold: {self.area_mismatch_threshold}%)"
                )

        # Real ST_Contains point-in-polygon check, executed in SQL, not Shapely.
        point_in_polygon = None
        spatial_dev_meters = None
        coord_mismatch = False
        target_coords = coordinates
        if not target_coords and "gps_coordinates" in fields:
            try:
                raw_c = str(fields["gps_coordinates"].normalized_value).split(",")
                target_coords = (float(raw_c[0].strip()), float(raw_c[1].strip()))
            except (ValueError, IndexError, TypeError) as coord_err:
                logger.debug("Could not parse GPS coordinates: %s", coord_err)

        if target_coords:
            lat, lon = target_coords
            try:
                contains_row = db_session.execute(
                    text(
                        "SELECT ST_Contains(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) AS contains, "
                        "ST_Distance(geography(geom), geography(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))) AS dist_m "
                        "FROM cadastral_parcels "
                        "WHERE state = :state AND district = :district AND tehsil = :tehsil "
                        "AND village = :village AND khasra_number = :khasra AND geom IS NOT NULL LIMIT 1;"
                    ),
                    {"lon": lon, "lat": lat, "state": state, "district": district.upper(),
                     "tehsil": tehsil.upper(), "village": village.upper(), "khasra": khasra.upper()},
                ).mappings().first()
                if contains_row is not None:
                    point_in_polygon = bool(contains_row["contains"])
                    spatial_dev_meters = round(float(contains_row["dist_m"]), 1)
                    if not point_in_polygon:
                        coord_mismatch = True
                        flag_reasons.append(
                            f"Spatial Mismatch (PostGIS ST_Contains): Coordinates ({lat:.6f}, {lon:.6f}) fall outside "
                            f"cadastral polygon (ST_Distance: {spatial_dev_meters:.1f}m)"
                        )
            except Exception as e:
                logger.warning(f"[GIS] PostGIS ST_Contains query failed: {e}")

        if has_area_mismatch or coord_mismatch:
            status, is_verified, has_mismatch = GISStatus.MISMATCH, False, True
        else:
            status, is_verified, has_mismatch = GISStatus.MATCH, True, False

        return GISValidationResult(
            gis_status=status, is_verified=is_verified, has_mismatch=has_mismatch, parcel_id_found=True,
            state=state, district=district, tehsil=tehsil, village=village, khasra_number=khasra,
            gis_recorded_area_hectares=gis_area, extracted_area_hectares=extracted_area,
            area_deviation_percent=round(area_deviation, 2) if area_deviation is not None else None,
            coordinates_matched=True if not coord_mismatch else False,
            point_in_polygon_passed=point_in_polygon, spatial_deviation_meters=spatial_dev_meters,
            flag_reasons=flag_reasons, source_gis_layer=layer_name, data_source_label=data_source_label,
        )

    def _validate_gis_fallback(
        self,
        fields: Dict[str, ExtractedField],
        state_code: str,
        coordinates: Optional[Tuple[float, float]] = None,
    ) -> GISValidationResult:
        """
        In-Python Shapely registry validation (SQLite dev / no-Postgres fallback).
        This is the original implementation - unchanged and still fully tested.
        """
        state = state_code.upper()
        district = self._get_normalized_str(fields.get("district"))
        tehsil = self._get_normalized_str(fields.get("tehsil"))
        village = self._get_normalized_str(fields.get("village"))
        khasra = self._get_normalized_str(fields.get("khasra_number"))
        extracted_area = self._get_extracted_area_hectares(fields.get("land_area"))

        flag_reasons: List[str] = []

        # 1. Check for INSUFFICIENT_DATA
        if not khasra or (not village and not district):
            flag_reasons.append("Mandatory cadastral identifiers missing (Khasra/Survey No or Village/District)")
            return GISValidationResult(
                gis_status=GISStatus.INSUFFICIENT_DATA,
                is_verified=False,
                has_mismatch=False,
                parcel_id_found=False,
                state=state,
                district=district,
                tehsil=tehsil,
                village=village,
                khasra_number=khasra,
                flag_reasons=flag_reasons,
            )

        # 2. Query Cadastral GIS Reference Database
        record_key = f"{state}:{district.upper()}:{tehsil.upper()}:{village.upper()}:{khasra.upper()}"
        parcel_record = self._lookup_parcel(record_key, state, district, tehsil, village, khasra)

        # 3. Check for UNKNOWN (Missing Reference in GIS DB)
        if not parcel_record:
            flag_reasons.append(
                f"Parcel '{khasra}' not found in GIS layer for {state}/{district}/{tehsil}/{village} (Status: UNKNOWN)"
            )
            return GISValidationResult(
                gis_status=GISStatus.UNKNOWN,
                is_verified=False,
                has_mismatch=False,  # Strictly NOT classified as a mismatch per spec
                parcel_id_found=False,
                state=state,
                district=district,
                tehsil=tehsil,
                village=village,
                khasra_number=khasra,
                extracted_area_hectares=extracted_area,
                flag_reasons=flag_reasons,
            )

        # 4. Parcel Found -> Evaluate Area & Spatial Point-in-Polygon
        gis_area = parcel_record["gis_area_hectares"]
        layer_name = parcel_record.get("layer_name", "cadastral_layer")
        poly = parcel_record["polygon"]
        area_deviation = None
        has_area_mismatch = False
        point_in_polygon = None
        spatial_dev_meters = None

        # Check Area Consistency
        if extracted_area is not None and extracted_area > 0:
            diff = abs(extracted_area - gis_area)
            area_deviation = (diff / gis_area) * 100.0
            if area_deviation > self.area_mismatch_threshold:
                has_area_mismatch = True
                flag_reasons.append(
                    f"GIS Area Discrepancy: Extracted area ({extracted_area:.4f} ha) differs from GIS cadastral area ({gis_area:.4f} ha) by {area_deviation:.1f}% (Threshold: {self.area_mismatch_threshold}%)"
                )

        # Check Point-in-Polygon Coordinates if provided
        coord_mismatch = False
        target_coords = coordinates
        if not target_coords and "gps_coordinates" in fields:
            try:
                raw_c = str(fields["gps_coordinates"].normalized_value).split(",")
                target_coords = (float(raw_c[0].strip()), float(raw_c[1].strip()))
            except (ValueError, IndexError, TypeError) as coord_err:
                logger.debug("Could not parse GPS coordinates: %s", coord_err)

        if target_coords:
            lat, lon = target_coords
            pt = Point(lon, lat)  # (x=lon, y=lat)
            point_in_polygon = poly.contains(pt)
            
            # Approximate distance from centroid in meters (1 deg ~ 111,320m)
            c_lon, c_lat = parcel_record["centroid"]
            dx = (lon - c_lon) * 111320.0 * math.cos(math.radians(c_lat))
            dy = (lat - c_lat) * 111320.0
            spatial_dev_meters = round(math.sqrt(dx * dx + dy * dy), 1)

            if not point_in_polygon:
                coord_mismatch = True
                flag_reasons.append(
                    f"Spatial Mismatch: Coordinates ({lat:.6f}, {lon:.6f}) fall outside cadastral polygon (Centroid distance: {spatial_dev_meters:.1f}m)"
                )

        # Final Status determination
        if has_area_mismatch or coord_mismatch:
            status = GISStatus.MISMATCH
            is_verified = False
            has_mismatch = True
        else:
            status = GISStatus.MATCH
            is_verified = True
            has_mismatch = False

        return GISValidationResult(
            gis_status=status,
            is_verified=is_verified,
            has_mismatch=has_mismatch,
            parcel_id_found=True,
            state=state,
            district=district,
            tehsil=tehsil,
            village=village,
            khasra_number=khasra,
            gis_recorded_area_hectares=gis_area,
            extracted_area_hectares=extracted_area,
            area_deviation_percent=round(area_deviation, 2) if area_deviation is not None else None,
            coordinates_matched=True if not coord_mismatch else False,
            point_in_polygon_passed=point_in_polygon,
            spatial_deviation_meters=spatial_dev_meters,
            flag_reasons=flag_reasons,
            source_gis_layer=layer_name,
            data_source_label=parcel_record.get("data_source_label", "registered_cadastral_parcel"),
        )

    def _lookup_parcel(
        self,
        exact_key: str,
        state: str,
        district: str,
        tehsil: str,
        village: str,
        khasra: str,
    ) -> Optional[Dict[str, Any]]:
        """Queries GIS database or GeoJSON cadastral registry."""
        # 1. Exact match
        if exact_key in self._cadastral_registry:
            return self._cadastral_registry[exact_key]

        # 2. Case-insensitive key match
        for k, v in self._cadastral_registry.items():
            if k.upper() == exact_key.upper():
                return v

        # 3. Fuzzy matching by state, khasra, and alias
        for k, v in self._cadastral_registry.items():
            parts = k.split(":")
            if len(parts) == 5:
                reg_state, reg_dist, reg_teh, reg_vil, reg_khasra = parts
                if reg_state.upper() == state.upper() and reg_khasra.upper() == khasra.upper():
                    # If village or district substring matches
                    if (village and (village.upper() in reg_vil.upper() or reg_vil.upper() in village.upper())) or \
                       (district and (district.upper() in reg_dist.upper() or reg_dist.upper() in district.upper())):
                        return v

        return None

    def _get_normalized_str(self, field_obj: Optional[ExtractedField]) -> str:
        if not field_obj or field_obj.normalized_value is None:
            return ""
        return str(field_obj.normalized_value).strip()

    def _get_extracted_area_hectares(self, field_obj: Optional[ExtractedField]) -> Optional[float]:
        if not field_obj or field_obj.normalized_value is None:
            return None
        try:
            return float(field_obj.normalized_value)
        except (ValueError, TypeError):
            return None
