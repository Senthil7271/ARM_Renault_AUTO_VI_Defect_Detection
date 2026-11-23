"""Unit and integration tests for SageMaker AWS deployment scripts and entry point."""

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure aws module and root autovi_project are importable
AWS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AWS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(AWS_DIR))

import yaml
from aws import sagemaker_entry
from aws.upload_code import create_source_tarball


class TestSageMakerEntry(unittest.TestCase):
    """Tests for SageMaker entry point argument parsing and configuration."""

    def test_parse_args_defaults(self):
        """Tests parse_args with default environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "argv", ["sagemaker_entry.py"]):
                args = sagemaker_entry.parse_args()
                self.assertEqual(args.data_dir, "/opt/ml/input/data/train")
                self.assertEqual(args.model_dir, "/opt/ml/model")
                self.assertEqual(args.output_dir, "/opt/ml/output/data")
                self.assertEqual(args.category, "pipe_clip")
                self.assertEqual(args.coreset_ratio, 0.05)
                self.assertEqual(args.chunk_size, 8192)
                self.assertEqual(args.batch_size, 16)
                self.assertEqual(args.device, "auto")

    def test_parse_args_with_sagemaker_env_vars(self):
        """Tests that standard SageMaker container environment variables are respected."""
        test_env = {
            "SM_CHANNEL_TRAIN": "/custom/input/train",
            "SM_MODEL_DIR": "/custom/model",
            "SM_OUTPUT_DATA_DIR": "/custom/output",
        }
        with patch.dict(os.environ, test_env, clear=True):
            with patch.object(sys, "argv", ["sagemaker_entry.py"]):
                args = sagemaker_entry.parse_args()
                self.assertEqual(args.data_dir, "/custom/input/train")
                self.assertEqual(args.model_dir, "/custom/model")
                self.assertEqual(args.output_dir, "/custom/output")

    def test_parse_args_cli_overrides(self):
        """Tests that explicit CLI arguments take precedence."""
        cli_args = [
            "sagemaker_entry.py",
            "--data-dir", "/cli/data",
            "--model-dir", "/cli/models",
            "--output-dir", "/cli/outputs",
            "--category", "engine_wiring",
            "--coreset-ratio", "0.10",
            "--chunk-size", "4096",
            "--device", "cuda",
            "--visualize-count", "4",
        ]
        with patch.object(sys, "argv", cli_args):
            args = sagemaker_entry.parse_args()
            self.assertEqual(args.data_dir, "/cli/data")
            self.assertEqual(args.model_dir, "/cli/models")
            self.assertEqual(args.output_dir, "/cli/outputs")
            self.assertEqual(args.category, "engine_wiring")
            self.assertAlmostEqual(args.coreset_ratio, 0.10)
            self.assertEqual(args.chunk_size, 4096)
            self.assertEqual(args.device, "cuda")
            self.assertEqual(args.visualize_count, 4)

    def test_environment_banner_runs(self):
        """Tests that print_environment_banner prints formatted diagnostics without throwing."""
        with patch.object(sys, "argv", ["sagemaker_entry.py"]):
            args = sagemaker_entry.parse_args()
            import torch
            device = torch.device("cpu")
            # Should run cleanly
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                sagemaker_entry.print_environment_banner(args, device)
            output = captured.getvalue()
            self.assertIn("AutoVI PatchCore - AWS SageMaker GPU Training", output)
            self.assertIn("Execution Target: cpu", output)


class TestSageMakerConfig(unittest.TestCase):
    """Tests sagemaker.yaml configuration schema and required settings."""

    def test_config_structure(self):
        config_path = AWS_DIR / "configs" / "sagemaker.yaml"
        self.assertTrue(config_path.is_file(), f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("aws", cfg)
        self.assertIn("sagemaker", cfg)
        self.assertIn("training", cfg)

        # Validate AWS block
        self.assertEqual(cfg["aws"]["region"], "us-east-1")
        self.assertTrue(cfg["aws"]["s3_bucket"].startswith("sagemaker-us-east-1-"))
        self.assertEqual(cfg["aws"]["s3_data_prefix"], "autovi")

        # Validate SageMaker block
        self.assertIn("pytorch-training", cfg["sagemaker"]["image_uri"])
        self.assertEqual(cfg["sagemaker"]["instance_type"], "ml.g4dn.xlarge")
        self.assertGreaterEqual(cfg["sagemaker"]["volume_size_gb"], 30)

        # Validate Training block
        self.assertEqual(cfg["training"]["category"], "pipe_clip")
        self.assertGreater(cfg["training"]["coreset_ratio"], 0.0)
        self.assertLessEqual(cfg["training"]["coreset_ratio"], 1.0)
        # Validate Role ARN format
        self.assertTrue(cfg["sagemaker"]["role_arn"].startswith("arn:aws:iam::077578771161:role/"))


class TestCodePackaging(unittest.TestCase):
    """Tests packaging the local repository into sourcedir.tar.gz for SageMaker."""

    def test_create_tarball(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tar = Path(tmp_dir) / "test_sourcedir.tar.gz"
            create_source_tarball(output_tar_path=out_tar)
            self.assertTrue(out_tar.is_file())
            self.assertGreater(out_tar.stat().st_size, 1024)

            # Inspect tarball contents
            with tarfile.open(out_tar, "r:gz") as tar:
                names = tar.getnames()
                self.assertIn("sagemaker_entry.py", names)
                self.assertIn("requirements.txt", names)
                # Check that src files are bundled
                has_model = any("src/models/patchcore_model.py" in n for n in names)
                has_loader = any("src/dataset/loader.py" in n for n in names)
                self.assertTrue(has_model, "patchcore_model.py missing in tarball")
                self.assertTrue(has_loader, "loader.py missing in tarball")

                # Verify requirements.txt pins NumPy 1.x and OpenCV <4.10
                req_member = tar.extractfile("requirements.txt")
                req_content = req_member.read().decode("utf-8")
                self.assertIn("numpy>=1.24.4,<2.0.0", req_content)
                self.assertIn("opencv-python-headless>=4.8.0.76,<4.10.0.0", req_content)


if __name__ == "__main__":
    unittest.main()
