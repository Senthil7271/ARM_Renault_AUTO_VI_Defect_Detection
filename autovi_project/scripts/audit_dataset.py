"""Dataset Audit Script for AutoVI.

Usage:
    python scripts/audit_dataset.py [--workspace ROOT] [--output OUT]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.audit import DatasetAuditor


def main():
    parser = argparse.ArgumentParser(description="Audit AutoVI Dataset structure and integrity")
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(PROJECT_ROOT.parent),
        help="Path to workspace root containing AutoVI categories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "dataset_audit"),
        help="Directory to save audit CSVs and summary",
    )
    args = parser.parse_args()

    workspace_p = Path(args.workspace).resolve()
    output_p = Path(args.output).resolve()

    print(f"==================================================")
    print(f"Starting AutoVI Dataset Audit")
    print(f"Workspace: {workspace_p}")
    print(f"Output:    {output_p}")
    print(f"==================================================")

    auditor = DatasetAuditor(workspace_p)
    results = auditor.run_full_audit(output_p)

    print("\nDataset Summary:")
    print(results["summary"].to_string(index=False))

    print(f"\nAudit complete. Reports saved to: {output_p}")
    print(f"Summary markdown: {output_p / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
