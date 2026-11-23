"""Packages training source code and uploads it to S3 deployment prefix.

Creates:
  sourcedir.tar.gz containing:
    ├── sagemaker_entry.py
    ├── requirements.txt
    ├── src/
    └── configs/

Uploads to:
  s3://sagemaker-us-east-1-077578771161/autovi/deployment/sourcedir.tar.gz
"""

import argparse
import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_source_tarball(output_tar_path: Path) -> str:
    """Packages the code files into a gzipped tar archive."""
    output_tar_path = Path(output_tar_path).resolve()
    output_tar_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Packaging source directory from: {PROJECT_ROOT}")
    with tarfile.open(output_tar_path, "w:gz") as tar:
        # Add entry script as root 'sagemaker_entry.py'
        entry_script = PROJECT_ROOT / "aws" / "sagemaker_entry.py"
        tar.add(entry_script, arcname="sagemaker_entry.py")

        # Add requirements-training.txt as root 'requirements.txt'
        req_file = PROJECT_ROOT / "aws" / "requirements-training.txt"
        tar.add(req_file, arcname="requirements.txt")

        # Add src/ directory
        src_dir = PROJECT_ROOT / "src"
        for file_p in src_dir.rglob("*.py"):
            arcname = file_p.relative_to(PROJECT_ROOT).as_posix()
            tar.add(file_p, arcname=arcname)

        # Add configs/ directory
        cfg_dir = PROJECT_ROOT / "configs"
        for file_p in cfg_dir.rglob("*.yaml"):
            arcname = file_p.relative_to(PROJECT_ROOT).as_posix()
            tar.add(file_p, arcname=arcname)

    # Compute SHA-256
    hasher = hashlib.sha256()
    with open(output_tar_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()
    print(f"Package created: {output_tar_path} ({output_tar_path.stat().st_size / 1024:.1f} KiB)")
    print(f"SHA-256: {sha256_hash}")
    return sha256_hash


def upload_to_s3(
    tar_path: Path,
    bucket: str = "sagemaker-us-east-1-077578771161",
    key: str = "autovi/deployment/sourcedir.tar.gz",
    profile: str = "autovi",
    region: str = "us-east-1",
) -> str:
    """Uploads tarball to S3."""
    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    print(f"Uploading to s3://{bucket}/{key} using profile '{profile}'...")
    s3.upload_file(str(tar_path), bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"[SUCCESS] Uploaded code package to: {s3_uri}")
    return s3_uri


def main():
    parser = argparse.ArgumentParser(description="Package and upload AutoVI code to S3")
    parser.add_argument("--profile", type=str, default="autovi", help="AWS profile")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", type=str, default="sagemaker-us-east-1-077578771161", help="S3 bucket")
    parser.add_argument("--key", type=str, default="autovi/deployment/sourcedir.tar.gz", help="S3 key")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "sourcedir.tar.gz"
        create_source_tarball(tar_path)
        upload_to_s3(tar_path, bucket=args.bucket, key=args.key, profile=args.profile, region=args.region)


if __name__ == "__main__":
    main()
