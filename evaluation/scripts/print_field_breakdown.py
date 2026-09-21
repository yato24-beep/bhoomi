import json
from pathlib import Path

p = Path("evaluation/real_ocr_semantic_evaluation_report.json")
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

for doc in d["documents"]:
    print(f"\n=== {doc['document_id']} ({doc['detected_document_type']}) ===")
    print(f"Total Regions: {doc['total_regions']} | Conf: {doc['overall_confidence']:.2f} | Review: {doc['requires_human_review']}")
    for fe in doc["field_evaluations"]:
        print(f"  {fe['field_name']:<20}: val={str(fe['predicted_value'])[:30]:<30} | {fe['classification']:<12} | prov={str(fe['provenance_region_id']):<15} | valid={fe['validation_status']}")
