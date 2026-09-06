# Cadastral GIS Dataset Metadata & Data Source Provenance

This document establishes the official data provenance, Coordinate Reference System (CRS), schema definitions, identifier mappings, and transparent data source labels for the GIS validation layer.

---

## 1. Data Source Honesty & Transparency Label

| Dataset Name | State / Authority | Data Source Label | Legal & Technical Status | Source CRS |
|---|---|---|---|---|
| `karnataka_bhoomi_cadastral_demo.geojson` | Karnataka (Bhoomi Project / Survey Settlement) | `synthetic_demo_cadastral` | Conforms to official Bhoomi RTC schema; synthetic polygons for prototype demonstration. | `EPSG:4326` (WGS84) |
| `tamilnadu_eservices_cadastral_demo.geojson` | Tamil Nadu (AnyTamilNilam / e-Services) | `synthetic_demo_cadastral` | Conforms to official TamilNilam Patta schema; synthetic polygons for prototype demonstration. | `EPSG:4326` (WGS84) |
| `maharashtra_mahabhumi_cadastral_demo.geojson` | Maharashtra (Mahabhumi / MahaBhulekh) | `synthetic_demo_cadastral` | Conforms to official 7/12 Satbara schema; synthetic polygons for prototype demonstration. | `EPSG:4326` (WGS84) |
| `up_bhulekh_cadastral_demo.geojson` | Uttar Pradesh (Bhulekh / Revenue Board) | `synthetic_demo_cadastral` | Conforms to official Khatauni Gata schema; synthetic polygons for prototype demonstration. | `EPSG:4326` (WGS84) |

> [!IMPORTANT]
> **Data Integrity Notice:** In strict compliance with architectural guidelines, synthetic/demo datasets are explicitly marked with `data_source_type: "synthetic_demo_cadastral"` and are never misrepresented as certified government shapefiles.

---

## 2. Parcel Identifier Mapping Matrix

| State Code | Primary Document | Document Field | Cadastral GIS Field | Secondary Identifier |
|---|---|---|---|---|
| **KA** (Karnataka) | Bhoomi RTC / Pahani | `khasra_number` (Survey No `ಸರ್ವೆ ನಂ`) | `survey_number` | `hissa_number` (`ಹಿಸ್ಸಾ ನಂ`) |
| **TN** (Tamil Nadu) | Patta / Chitta / 'A' Reg | `khasra_number` (Survey No `புல எண்`) | `survey_number` | `subdivision_number` (`உட்பிரிவு எண்`) |
| **MH** (Maharashtra) | 7/12 Satbara Extract | `khasra_number` (Gat No `गट क्रमांक`) | `khasra_number` | `survey_number` (`सर्व्हे नंबर`) |
| **UP** (Uttar Pradesh) | Khatauni / Khasra | `khasra_number` (Gata `गाटा संख्या`) | `khasra_number` | `khatauni_number` (`खाता संख्या`) |

---

## 3. Cadastral Reference GeoJSON Schema

Every GeoJSON feature conforms to the standard GeoJSON specification (`RFC 7946`) with the following feature properties:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
  },
  "properties": {
    "state": "KA",
    "district": "BENGALURU URBAN",
    "tehsil": "BANGALORE SOUTH",
    "village": "KENGERI",
    "khasra_number": "42/1",
    "area_hectares": 0.8094,
    "gis_layer_name": "ka_bhoomi_cadastral_demo",
    "data_source_label": "synthetic_demo_cadastral",
    "last_updated": "2026-08-28"
  }
}
```
