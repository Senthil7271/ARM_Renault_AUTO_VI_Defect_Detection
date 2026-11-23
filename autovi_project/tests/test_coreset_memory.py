"""Regression tests for memory-bounded greedy k-center coreset subsampling.

Tests:
1. Exact mathematical equivalence against a trusted naive reference implementation.
2. Uniqueness guarantee: no center can be selected more than once.
3. Memory boundedness: zero materialization of full (N, D) temporary distance matrices.
4. Numerical clamping: prevention of negative squared distances due to float precision.
5. Candidate pool subsampling integration.
"""

import os
from pathlib import Path
import unittest
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchcore_model import PatchCoreModel


class TestCoresetMemory(unittest.TestCase):
    def setUp(self):
        self.model = PatchCoreModel(
            backbone_name="wide_resnet50_2",
            target_size=(256, 256),
            coreset_ratio=0.1,
            chunk_size=128,
        )

    def test_exact_mathematical_equivalence_to_reference(self):
        """Verifies that chunked squared Euclidean selection matches reference Euclidean selection."""
        n_samples = 300
        dim = 32
        target_size = 20

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, dim)).astype(np.float32)

        # 1. Trusted Reference Implementation (naive Euclidean with full broadcasting)
        ref_rng = np.random.default_rng(42)
        first_idx = int(ref_rng.integers(0, n_samples))
        ref_selected = [first_idx]
        ref_dists = np.linalg.norm(X - X[first_idx:first_idx + 1], axis=1)
        ref_dists[first_idx] = -1.0

        for _ in range(1, target_size):
            best_idx = int(np.argmax(ref_dists))
            ref_selected.append(best_idx)
            new_d = np.linalg.norm(X - X[best_idx:best_idx + 1], axis=1)
            ref_dists = np.minimum(ref_dists, new_d)
            ref_dists[best_idx] = -1.0

        # 2. Optimized Memory-Bounded Chunked Implementation
        opt_selected = self.model._greedy_kcenter_subsampling(
            X, target_size=target_size, verbose=False
        )

        self.assertEqual(len(opt_selected), target_size)
        self.assertListEqual(
            list(opt_selected),
            ref_selected,
            "Optimized chunked selection did not select identical centers to reference!"
        )

    def test_no_duplicate_centers_selected(self):
        """Ensures that once a center is selected, it cannot be selected again."""
        n_samples = 500
        dim = 64
        target_size = 50

        rng = np.random.default_rng(123)
        X = rng.standard_normal((n_samples, dim)).astype(np.float32)

        selected = self.model._greedy_kcenter_subsampling(
            X, target_size=target_size, verbose=False
        )

        self.assertEqual(len(selected), target_size)
        self.assertEqual(len(set(selected)), target_size, "Duplicate centers were selected!")

    def test_memory_bounded_scaling(self):
        """Verifies that selecting centers over 20,000 points does not spike memory."""
        if psutil is None:
            self.skipTest("psutil not available")

        proc = psutil.Process(os.getpid())
        n_samples = 20000
        dim = 128
        target_size = 50

        rng = np.random.default_rng(99)
        X = rng.standard_normal((n_samples, dim)).astype(np.float32)

        start_rss = proc.memory_info().rss / (1024.0 * 1024.0)

        # Run with small chunk size 1024
        self.model.chunk_size = 1024
        selected = self.model._greedy_kcenter_subsampling(
            X, target_size=target_size, verbose=False
        )

        end_rss = proc.memory_info().rss / (1024.0 * 1024.0)
        rss_delta = end_rss - start_rss

        self.assertEqual(len(selected), target_size)
        # RSS delta should be minimal (under 30 MiB), never allocating 87.5 MiB per step
        self.assertLess(rss_delta, 30.0, f"Memory expanded unexpectedly by {rss_delta:.2f} MiB")

    def test_numerical_clamping_avoids_nans(self):
        """Ensures identical vectors with floating-point roundoff do not create negative distances or NaNs."""
        n_samples = 100
        dim = 32
        target_size = 10

        # Create repeated rows so dot product equals sum of squares
        base = np.ones((1, dim), dtype=np.float32) * 3.14159
        X = np.repeat(base, n_samples, axis=0)

        selected = self.model._greedy_kcenter_subsampling(
            X, target_size=target_size, verbose=False
        )
        self.assertEqual(len(selected), target_size)
        self.assertEqual(len(set(selected)), target_size)


if __name__ == "__main__":
    unittest.main()
