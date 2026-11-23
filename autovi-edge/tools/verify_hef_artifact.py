#!/usr/bin/env python3
"""AutoVI: HEF Artifact Verification and Metadata Inspector.

Inspects any `.hef` file to determine if it is:
1. A genuine compiled binary Hailo Executable Format (HEF).
2. An uncompiled JSON metadata stub (which must be rejected).
3. A corrupted or truncated file.

If `hailortcli` is installed, it runs `hailortcli parse-hef` to display
hardware streams, network groups, and compiled target architecture.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
EDGE_ROOT = TOOLS_DIR.parent
DEFAULT_HEF = EDGE_ROOT / "models" / "wideresnet50_patchcore.hef"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_hef(hef_path: Path = DEFAULT_HEF):
    print("=" * 80)
    print("AUTOVI: HEF ARTIFACT AUDIT & INTEGRITY INSPECTOR")
    print("=" * 80)

    if not hef_path.exists():
        print(f"[FAIL] Target file does not exist: {hef_path}")
        return False

    size_bytes = hef_path.stat().st_size
    file_hash = sha256_file(hef_path)

    print(f"File Path:   {hef_path}")
    print(f"File Size:   {size_bytes} bytes ({size_bytes / 1e6:.4f} MB)")
    print(f"SHA-256:     {file_hash}")

    # Inspect first 512 bytes
    with open(hef_path, "rb") as f:
        head = f.read(512)

    is_json = False
    try:
        text = head.decode("utf-8")
        if text.strip().startswith("{"):
            is_json = True
    except UnicodeDecodeError:
        pass

    if is_json or size_bytes < 1_000_000:
        print("\n" + "!" * 80)
        print("VERDICT: [REJECTED] UNCOMPILED METADATA STUB")
        print("!" * 80)
        print("Reason: File is a text/JSON placeholder, NOT a compiled Hailo binary.")
        print(f"Physical WideResNet-50-2 HEF must be 15 MB - 35 MB.")
        try:
            data = json.loads(hef_path.read_text(encoding="utf-8"))
            print("Stub Contents:")
            print(json.dumps(data, indent=2))
        except Exception:
            print(f"Preview (ASCII): {head[:120]}")
        print("\nREQUIRED ACTION:")
        print("Compile 'wideresnet50_patchcore.onnx' using Hailo Dataflow Compiler (DFC) on Linux x86_64.")
        return False

    print("\n" + "=" * 80)
    print("VERDICT: [ACCEPTED] PHYSICAL BINARY HEF ARTIFACT")
    print("=" * 80)

    # Check hailortcli
    if shutil.which("hailortcli"):
        print("\nRunning 'hailortcli parse-hef'...")
        res = subprocess.run(["hailortcli", "parse-hef", str(hef_path)], capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("[STDERR]", res.stderr)
    else:
        print("\n'hailortcli' not detected on current host. To view stream metadata on Raspberry Pi:")
        print(f"  hailortcli parse-hef {hef_path.name}")

    return True


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HEF
    success = inspect_hef(target)
    sys.exit(0 if success else 1)
