"""Unit and smoke tests for Trainable Deep Anomaly Detection Models (Phase 3).

Verifies:
1. Synthetic pseudo-anomaly generator mechanics.
2. Reverse Distillation frozen teacher vs trainable student parameters.
3. Nontrivial gradient computation and parameter updates across training steps.
4. Decreasing training loss on small nominal batches.
5. Full serialization and reloading fidelity.
6. DRAEM reconstructive and discriminative forward/backward passes.
"""

import gc
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.synthetic_anomalies import SyntheticAnomalyGenerator, generate_synthetic_mask
from src.models.draem import DRAEMModel, focal_dice_loss, ssim_loss
from src.models.reverse_distillation import ReverseDistillationModel


class TestDeepAnomalyModels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        gc.collect()

    def tearDown(self):
        self.temp_dir.cleanup()
        gc.collect()

    def test_synthetic_anomaly_generator(self):
        # Create dummy normal image: (128, 128, 3) in [0, 1]
        dummy_nominal = np.ones((128, 128, 3), dtype=np.float32) * 0.5
        generator = SyntheticAnomalyGenerator(nominal_pool=[dummy_nominal], p_anomaly=1.0, seed=42)

        corrupted, mask, is_anomaly = generator.generate(dummy_nominal)

        self.assertEqual(is_anomaly, 1)
        self.assertEqual(corrupted.shape, (128, 128, 3))
        self.assertEqual(mask.shape, (128, 128))
        self.assertGreater(np.sum(mask), 0)
        diff = np.abs(corrupted - dummy_nominal)
        self.assertGreater(np.sum(diff), 0.0)

    def test_reverse_distillation_gradients_and_updates(self):
        model = ReverseDistillationModel(backbone_name="resnet18", bottleneck_dim=64)
        model.eval()

        # 1. Verify frozen teacher vs trainable student
        for p in model.teacher_parameters():
            self.assertFalse(p.requires_grad, "Teacher parameter must be frozen")
        
        trainable_params = list(model.trainable_parameters())
        self.assertGreater(len(trainable_params), 0)
        for p in trainable_params:
            self.assertTrue(p.requires_grad, "Student parameter must be trainable")

        initial_weight = trainable_params[0].clone().detach()

        # 2. Forward pass with dummy batch
        x = torch.randn(2, 3, 128, 128)
        t_feats, s_feats = model(x)
        loss = model.compute_cosine_discrepancy_loss(t_feats, s_feats)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        # 3. Backward pass & parameter update
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()

        self.assertIsNotNone(trainable_params[0].grad)
        self.assertGreater(torch.norm(trainable_params[0].grad).item(), 0.0)

        for p in model.teacher_parameters():
            self.assertIsNone(p.grad)

        optimizer.step()

        updated_weight = trainable_params[0].clone().detach()
        weight_diff = torch.norm(updated_weight - initial_weight).item()
        self.assertGreater(weight_diff, 1e-6, "Student weights must update after optimizer step")

    def test_reverse_distillation_loss_decrease(self):
        """Verify that training on a tiny batch reduces discrepancy loss."""
        torch.manual_seed(42)
        model = ReverseDistillationModel(backbone_name="resnet18", bottleneck_dim=64)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=5e-3)

        x = torch.randn(2, 3, 128, 128)
        initial_loss = None
        final_loss = None

        for step in range(5):
            optimizer.zero_grad()
            t_feats, s_feats = model(x)
            loss = model.compute_cosine_discrepancy_loss(t_feats, s_feats)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss, "Training steps must decrease loss on fixed batch")

    def test_reverse_distillation_serialization(self):
        model = ReverseDistillationModel(backbone_name="resnet18", bottleneck_dim=64)
        model.threshold = 0.42
        model.calibrated = True

        save_p = self.root / "rd_test.pth"
        model.save(save_p)
        self.assertTrue(save_p.exists())

        reloaded = ReverseDistillationModel.load(save_p, device="cpu")
        self.assertEqual(reloaded.threshold, 0.42)
        self.assertTrue(reloaded.calibrated)

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            s1, m1 = model.predict_anomaly(x)
            s2, m2 = reloaded.predict_anomaly(x)

        np.testing.assert_allclose(s1.numpy(), s2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(m1.numpy(), m2.numpy(), rtol=1e-5, atol=1e-5)

    def test_draem_gradients_and_updates(self):
        model = DRAEMModel(base_reconstruct_c=16, base_discriminative_c=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        clean = torch.rand(2, 3, 128, 128)
        corrupted = torch.rand(2, 3, 128, 128)
        mask = (torch.rand(2, 1, 128, 128) > 0.8).float()

        first_param = next(model.parameters())
        initial_weight = first_param.clone().detach()

        optimizer.zero_grad()
        loss, metrics = model.compute_total_loss(clean, corrupted, mask)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_rec", metrics)
        self.assertIn("loss_disc", metrics)

        loss.backward()
        self.assertIsNotNone(first_param.grad)
        self.assertGreater(torch.norm(first_param.grad).item(), 0.0)

        optimizer.step()
        weight_diff = torch.norm(first_param.detach() - initial_weight).item()
        self.assertGreater(weight_diff, 1e-6)

    def test_draem_serialization(self):
        model = DRAEMModel(base_reconstruct_c=16, base_discriminative_c=16)
        model.threshold = 0.75
        model.calibrated = True

        save_p = self.root / "draem_test.pth"
        model.save(save_p)
        self.assertTrue(save_p.exists())

        reloaded = DRAEMModel.load(save_p, device="cpu")
        self.assertEqual(reloaded.threshold, 0.75)
        self.assertTrue(reloaded.calibrated)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            s1, m1 = model.predict_anomaly(x)
            s2, m2 = reloaded.predict_anomaly(x)

        np.testing.assert_allclose(s1.numpy(), s2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(m1.numpy(), m2.numpy(), rtol=1e-5, atol=1e-5)


class TestSageMakerDeepPreflight(unittest.TestCase):
    """Regression tests preventing avoidable SageMaker script failures."""

    def test_sagemaker_train_deep_no_undefined_names(self):
        """Audits sagemaker_train_deep.py with symtable to prove 0 undefined variables exist."""
        import builtins
        import symtable
        entry_script = PROJECT_ROOT / "aws" / "sagemaker_train_deep.py"
        self.assertTrue(entry_script.exists())

        with open(entry_script, "r", encoding="utf-8") as f:
            src = f.read()

        tbl = symtable.symtable(src, str(entry_script), "exec")
        undefined = []

        def check_tbl(t):
            for sym in t.get_symbols():
                if sym.is_global() and not sym.is_declared_global():
                    name = sym.get_name()
                    if name not in dir(builtins) and name not in ["__file__", "__name__", "__doc__"]:
                        try:
                            top_sym = tbl.lookup(name)
                            if not top_sym.is_assigned() and not top_sym.is_imported():
                                undefined.append(name)
                        except KeyError:
                            undefined.append(name)
            for child in t.get_children():
                check_tbl(child)

        check_tbl(tbl)
        self.assertEqual(undefined, [], f"Undefined names detected in sagemaker_train_deep.py: {undefined}")

    def test_sagemaker_train_deep_cli_parsing(self):
        """Verifies sagemaker_train_deep CLI argument parser."""
        from aws.sagemaker_train_deep import parse_args
        args = parse_args([
            "--category", "pipe_clip",
            "--model-type", "reverse_distillation",
            "--epochs", "5",
            "--batch-size", "4",
            "--lr", "0.0002"
        ])
        self.assertEqual(args.category, "pipe_clip")
        self.assertEqual(args.model_type, "reverse_distillation")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.0002)

    def test_recovered_checkpoint_reloading(self):
        """Verifies that the recovered Pilot 2 checkpoint loads and infers cleanly on CPU."""
        rec_ckpt = PROJECT_ROOT / "outputs" / "phase3" / "recovered_pilot2" / "pipe_clip" / "reverse_distillation" / "best_model.pth"
        if rec_ckpt.exists():
            model = ReverseDistillationModel.load(rec_ckpt, device="cpu")
            x = torch.randn(1, 3, 256, 256)
            score, amap = model.predict_anomaly(x)
            self.assertGreater(score.item(), 0.0)
            self.assertEqual(amap.shape, (1, 256, 256))


if __name__ == "__main__":
    unittest.main()
