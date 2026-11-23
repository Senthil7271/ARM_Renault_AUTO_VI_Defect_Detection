"""Test Suite for Official AutoVI Benchmark Integration."""

import json
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestOfficialBenchmarkIntegration(unittest.TestCase):
    def setUp(self):
        self.out_base = PROJECT_ROOT / "outputs" / "official_autovi"
        self.reports_dir = PROJECT_ROOT / "reports"

    def test_all_six_categories_evaluated(self):
        """Verifies that all 6 categories have official metrics generated."""
        summary_file = self.out_base / "evaluation" / "all_categories_official_summary.json"
        self.assertTrue(summary_file.exists(), "Consolidated official summary JSON must exist.")
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_cats = [
            "pipe_clip",
            "pipe_staple",
            "engine_wiring",
            "tank_screw",
            "underbody_pipes",
            "underbody_screw",
        ]
        for cat in expected_cats:
            self.assertIn(cat, data, f"Category {cat} missing from official evaluation summary.")
            cat_data = data[cat]
            self.assertIn("localization", cat_data)
            self.assertIn("classification", cat_data)
            spros = cat_data["localization"]["auc_spro"]
            for fpr in ["0.01", "0.05", "0.1", "0.3", "1.0"]:
                self.assertIn(fpr, spros)
                val = spros[fpr]
                self.assertTrue(0.0 <= val <= 1.0, f"sPRO @ {fpr} for {cat} must be in [0, 1]")

    def test_comparison_table_completeness(self):
        """Verifies that previous vs official comparison tables are generated."""
        csv_file = self.reports_dir / "previous_vs_official_metrics.csv"
        md_file = self.reports_dir / "previous_vs_official_metrics.md"
        self.assertTrue(csv_file.exists(), "Comparison CSV must exist.")
        self.assertTrue(md_file.exists(), "Comparison MD must exist.")

    def test_operational_matrices_calibration_audit(self):
        """Verifies that operational confusion matrices reflect the 95th percentile calibration audit."""
        ops_json = self.reports_dir / "operational_confusion_matrices.json"
        ops_md = self.reports_dir / "operational_confusion_matrices.md"
        self.assertTrue(ops_json.exists(), "Operational confusion matrices JSON must exist.")
        self.assertTrue(ops_md.exists(), "Operational confusion matrices MD must exist.")
        with open(ops_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cat in ["pipe_clip", "pipe_staple", "engine_wiring", "tank_screw", "underbody_pipes", "underbody_screw"]:
            self.assertIn(cat, data)
            cal_meta = data[cat].get("calibration_metadata", {})
            self.assertEqual(cal_meta.get("percentile"), 95.0, "Audited percentile must be 95.0")
            self.assertEqual(cal_meta.get("target_nominal_fpr"), 0.05)
            self.assertGreater(cal_meta.get("validation_sample_size", 0), 0)

    def test_defect_subtypes_breakdown(self):
        """Verifies that per-defect-subtype breakdown report exists and contains valid metrics."""
        sub_md = self.reports_dir / "defect_subtypes_breakdown.md"
        self.assertTrue(sub_md.exists(), "Defect subtype breakdown must exist.")

    def test_engine_wiring_leakage_diagnostic(self):
        """Verifies exact denominators and calculations for engine_wiring train/test leakage."""
        leak_json = self.reports_dir / "engine_wiring_leakage_diagnostic.json"
        self.assertTrue(leak_json.exists(), "Leakage diagnostic JSON must exist.")
        with open(leak_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["category"], "engine_wiring")
        self.assertEqual(data["total_official_test_samples"], 607)
        self.assertEqual(data["official_normal_samples_count"], 285)
        self.assertEqual(data["official_defective_samples_count"], 322)
        self.assertEqual(data["unique_contaminated_test_normal_samples"], 60)
        self.assertEqual(data["clean_test_samples_count"], 547)
        self.assertEqual(data["clean_normal_samples_count"], 225)
        self.assertEqual(data["clean_defective_samples_count"], 322)
        self.assertAlmostEqual(data["full_split_binary_image_auroc"], 0.8044, places=3)
        self.assertAlmostEqual(data["clean_split_binary_image_auroc"], 0.7564, places=3)
        self.assertAlmostEqual(data["binary_auroc_delta"], -0.0480, places=3)

    def test_latency_benchmark_backbone(self):
        """Verifies that the latency benchmark correctly documents WideResNet-50-2 architecture."""
        lat_json = self.reports_dir / "inference_latency_benchmark.json"
        self.assertTrue(lat_json.exists(), "Latency benchmark JSON must exist.")
        with open(lat_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cat in ["pipe_clip", "pipe_staple", "engine_wiring", "tank_screw", "underbody_pipes", "underbody_screw"]:
            self.assertIn(cat, data)
            self.assertEqual(data[cat]["backbone"], "wide_resnet50_2")
            self.assertEqual(data[cat]["feature_dim"], 1536)

    def test_visualizations_exist(self):
        """Verifies that representative 4-panel figures and portfolio curves exist."""
        viz_dir = self.out_base / "visualizations"
        self.assertTrue((viz_dir / "portfolio_roc_curves.png").exists())
        self.assertTrue((viz_dir / "portfolio_pr_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
