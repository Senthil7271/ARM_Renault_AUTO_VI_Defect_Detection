"""Training script for AutoVI PatchCore models.

Usage:
    python scripts/train.py --category pipe_clip
    python scripts/train.py --category all
"""

import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import PatchCoreTrainer

ALL_CATEGORIES = [
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]


def train_single_category(category: str, args, default_cfg: dict):
    cfg = dict(default_cfg)
    if args.coreset_ratio is not None:
        cfg["model"]["coreset_sampling_ratio"] = args.coreset_ratio
    if args.chunk_size is not None:
        cfg["model"]["chunk_size"] = args.chunk_size
    if args.candidate_pool_size is not None:
        cfg["model"]["candidate_pool_size"] = args.candidate_pool_size
    if args.device is not None:
        cfg["training"]["device"] = args.device

    out_dir = Path(args.output) / category
    trainer = PatchCoreTrainer(
        workspace_root=args.workspace,
        category=category,
        config=cfg,
        output_dir=out_dir,
    )
    model = trainer.train()
    print(f"\n[DONE] Training complete for category '{category}'. Artifacts in: {out_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore model on AutoVI category")
    parser.add_argument(
        "--category",
        type=str,
        default="pipe_clip",
        help="Category name (or 'all' for all 6 categories)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(PROJECT_ROOT.parent),
        help="Workspace root containing categories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "experiments"),
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
        help="Override coreset sampling ratio (e.g. 0.01, 0.05, or 0.1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for memory-bounded distance computation (default: 8192)",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Optional candidate pool subsampling size for faster CPU experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cpu', 'cuda', 'auto')",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)

    if args.category.lower() == "all":
        for cat in ALL_CATEGORIES:
            print(f"\n==================================================")
            print(f"Processing Category: {cat}")
            print(f"==================================================")
            train_single_category(cat, args, default_cfg)
    else:
        train_single_category(args.category, args, default_cfg)


if __name__ == "__main__":
    main()
