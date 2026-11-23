"""AutoVI Edge: Cryptographic SHA-256 Manifest Generator.

Computes SHA-256 hashes for all deployed edge files (models, memory banks, code, configs)
and saves them to manifest.sha256 for deployment integrity verification.
"""

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_ROOT = PROJECT_ROOT / "autovi-edge"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    print("=" * 80)
    print("GENERATING SHA-256 INTEGRITY MANIFEST FOR AUTOVI-EDGE")
    print("=" * 80)

    rows = []
    for p in sorted(EDGE_ROOT.rglob("*")):
        if p.is_file() and p.name != "manifest.sha256" and "__pycache__" not in p.parts:
            sha = compute_sha256(p)
            rel_path = p.relative_to(EDGE_ROOT).as_posix()
            rows.append(f"{sha}  {rel_path}")
            print(f"  {sha[:16]}...  {rel_path}")

    manifest_file = EDGE_ROOT / "manifest.sha256"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"\n[SAVED] Manifest: {manifest_file} ({len(rows)} files hashed)")


if __name__ == "__main__":
    generate_manifest()
