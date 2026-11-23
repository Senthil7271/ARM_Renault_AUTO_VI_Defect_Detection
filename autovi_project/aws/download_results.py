"""Downloads and extracts SageMaker training job outputs from S3.

Usage:
    python autovi_project/aws/download_results.py --job-name <job-name>
"""

import argparse
import os
import tarfile
from pathlib import Path
import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_sagemaker_results(
    job_name: str,
    bucket: str = "sagemaker-us-east-1-077578771161",
    output_prefix: str = "autovi/outputs",
    profile: str = "autovi",
    region: str = "us-east-1",
    dest_dir: Path = None,
):
    dest_dir = Path(dest_dir or (PROJECT_ROOT / "outputs" / "sagemaker_runs" / job_name))
    dest_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")

    s3_prefix = f"{output_prefix}/{job_name}/output"
    print(f"Checking for results in s3://{bucket}/{s3_prefix}...")

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    contents = resp.get("Contents", [])

    if not contents:
        print(f"No objects found under s3://{bucket}/{s3_prefix}. Has the job finished?")
        return False

    for item in contents:
        key = item["Key"]
        filename = Path(key).name
        local_file = dest_dir / filename
        print(f"Downloading {key} -> {local_file}...")
        s3.download_file(bucket, key, str(local_file))

        # If tar.gz, extract it
        if filename.endswith(".tar.gz"):
            extract_folder = dest_dir / filename.replace(".tar.gz", "")
            extract_folder.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {filename} to {extract_folder}...")
            with tarfile.open(local_file, "r:gz") as tar:
                tar.extractall(extract_folder)

    print(f"\n[SUCCESS] Results successfully downloaded to: {dest_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SageMaker Job outputs from S3")
    parser.add_argument("--job-name", type=str, required=True, help="SageMaker training job name")
    parser.add_argument("--bucket", type=str, default="sagemaker-us-east-1-077578771161", help="S3 bucket")
    parser.add_argument("--prefix", type=str, default="autovi/outputs", help="S3 output prefix")
    parser.add_argument("--profile", type=str, default="autovi", help="AWS profile")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--dest", type=str, default=None, help="Local destination directory")
    args = parser.parse_args()

    download_sagemaker_results(
        job_name=args.job_name,
        bucket=args.bucket,
        output_prefix=args.prefix,
        profile=args.profile,
        region=args.region,
        dest_dir=args.dest,
    )


if __name__ == "__main__":
    main()
