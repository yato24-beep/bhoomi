"""Unit Tests for Dataset Bootstrap and Multilingual Manifest Utilities."""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.bootstrap import (
    LANGUAGE_SCRIPT_MAP,
    bootstrap_multilingual_datasets,
    ensure_dataset_directories,
    generate_annotation_template,
    get_canonical_script,
    scan_language_directory,
)


class TestDatasetBootstrap(unittest.TestCase):
    """Test suite for dataset bootstrapping, language scanning, and template generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)

        self.data_dir = self.root_dir / "data"
        self.raw_kannada = self.data_dir / "raw" / "kannada"
        self.raw_hindi = self.data_dir / "raw" / "hindi"
        self.raw_tamil = self.data_dir / "raw" / "tamil"
        self.ann_dir = self.data_dir / "annotations"
        self.man_dir = self.root_dir / "training" / "datasets"

        self.raw_kannada.mkdir(parents=True, exist_ok=True)
        self.raw_hindi.mkdir(parents=True, exist_ok=True)
        self.raw_tamil.mkdir(parents=True, exist_ok=True)
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.man_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy images
        self.kn_img1 = self.raw_kannada / "kannada_01.png"
        self.kn_img2 = self.raw_kannada / "kannada_02.png"
        self.hi_img1 = self.raw_hindi / "hindi_01.png"

        Image.new("RGB", (100, 30), color=255).save(self.kn_img1)
        Image.new("RGB", (100, 30), color=255).save(self.kn_img2)
        Image.new("RGB", (100, 30), color=255).save(self.hi_img1)

        # Create an annotation for kannada_01 only
        with open(self.ann_dir / "kannada.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "image": str(self.kn_img1),
                    "text": "ಕರ್ನಾಟಕ ಕಂದಾಯ ಇಲಾಖೆ",
                    "language": "kannada",
                    "script": "Kannada",
                    "metadata": {"source_type": "printed"},
                }) + "\n"
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_canonical_script_mapping(self):
        """Tests that Indic languages map to correct canonical scripts."""
        self.assertEqual(get_canonical_script("kannada"), "Kannada")
        self.assertEqual(get_canonical_script("hindi"), "Devanagari")
        self.assertEqual(get_canonical_script("tamil"), "Tamil")
        self.assertEqual(get_canonical_script("telugu"), "Telugu")
        self.assertEqual(get_canonical_script("malayalam"), "Malayalam")
        self.assertEqual(get_canonical_script("english"), "Latin")

    def test_ensure_dataset_directories(self):
        """Tests creation of all standard data directories."""
        new_temp = tempfile.TemporaryDirectory()
        base = Path(new_temp.name)

        data_base, subdirs = ensure_dataset_directories(
            data_root=base,
            languages=["kannada", "hindi", "tamil", "telugu", "malayalam"],
        )

        self.assertTrue((data_base / "raw" / "kannada").exists())
        self.assertTrue((data_base / "raw" / "hindi").exists())
        self.assertTrue((data_base / "raw" / "tamil").exists())
        self.assertTrue((data_base / "raw" / "telugu").exists())
        self.assertTrue((data_base / "raw" / "malayalam").exists())
        self.assertTrue((data_base / "annotations").exists())
        self.assertTrue((data_base / "processed").exists())
        self.assertTrue((data_base / "synthetic").exists())
        new_temp.cleanup()

    def test_scan_language_directory_detects_annotated_and_unannotated(self):
        """Tests that scanning accurately distinguishes annotated pairs from unannotated images."""
        scan_res = scan_language_directory(
            data_root=self.data_dir,
            language="kannada",
            annotations_dir=self.ann_dir,
            manifests_dir=self.man_dir,
        )

        self.assertEqual(scan_res.total_images_found, 2)
        self.assertEqual(scan_res.annotated_count, 1)
        self.assertEqual(scan_res.unannotated_count, 1)

        # Verified sample
        sample = scan_res.annotated_samples[0]
        self.assertEqual(sample.text, "ಕರ್ನಾಟಕ ಕಂದಾಯ ಇಲಾಖೆ")
        self.assertEqual(sample.language, "kannada")
        self.assertEqual(sample.script, "Kannada")

        # Unannotated image
        self.assertEqual(len(scan_res.unannotated_images), 1)
        self.assertEqual(scan_res.unannotated_images[0], self.kn_img2)

    def test_unannotated_images_do_not_fabricate_labels(self):
        """Tests that unannotated images are not given fake transcriptions."""
        scan_res = scan_language_directory(
            data_root=self.data_dir,
            language="hindi",
            annotations_dir=self.ann_dir,
            manifests_dir=self.man_dir,
        )

        self.assertEqual(scan_res.total_images_found, 1)
        self.assertEqual(scan_res.annotated_count, 0)
        self.assertEqual(scan_res.unannotated_count, 1)
        self.assertEqual(len(scan_res.annotated_samples), 0)

    def test_generate_annotation_template_creates_empty_text_records(self):
        """Tests that template generation outputs empty 'text' placeholders for human annotation."""
        out_tmpl = self.ann_dir / "hindi_template.jsonl"
        count = generate_annotation_template(
            unannotated_paths=[self.hi_img1],
            language="hindi",
            output_file=out_tmpl,
            source_type="real_handwriting",
            root_dir=self.root_dir,
        )

        self.assertEqual(count, 1)
        self.assertTrue(out_tmpl.exists())

        with open(out_tmpl, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline().strip())

        self.assertEqual(rec["language"], "hindi")
        self.assertEqual(rec["script"], "Devanagari")
        self.assertEqual(rec["text"], "")  # Must be empty placeholder
        self.assertEqual(rec["metadata"]["status"], "unannotated")
        self.assertEqual(rec["metadata"]["source_type"], "real_handwriting")

    def test_bootstrap_multilingual_datasets_end_to_end(self):
        """Tests end-to-end bootstrap orchestration across multiple Indic languages."""
        summary = bootstrap_multilingual_datasets(
            project_root=self.root_dir,
            languages=["kannada", "hindi", "tamil"],
            generate_templates=True,
        )

        self.assertEqual(summary.total_images, 3)
        self.assertEqual(summary.total_annotated, 1)
        self.assertEqual(summary.total_unannotated, 2)

        # Template for Kannada (since 1 unannotated image exists)
        self.assertTrue((self.ann_dir / "kannada_template.jsonl").exists())
        # Template for Hindi
        self.assertTrue((self.ann_dir / "hindi_template.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
