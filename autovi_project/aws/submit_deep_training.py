"""Submits Phase 3 Deep Anomaly Training Job to Amazon SageMaker with Quota and Cost Safeguards.

Usage:
    # Dry-run validation (default / safe mode, zero AWS charges):
    python autovi_project/aws/submit_deep_training.py --category pipe_clip --model-type reverse_distillation --dry-run

    # Live execution (ONLY after explicit approval):
    python autovi_project/aws/submit_deep_training.py --category pipe_clip --model-type reverse_distillation --execute
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_deep_source_tarball(output_tar_path: Path) -> str:
    """Packages Phase 3 code files and validated manifests into a gzipped tar archive."""
    output_tar_path = Path(output_tar_path).resolve()
    output_tar_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PACKAGING] Building Phase 3 source tarball from {PROJECT_ROOT}...")
    with tarfile.open(output_tar_path, "w:gz") as tar:
        # 1. Entry scripts
        entry_script = PROJECT_ROOT / "aws" / "sagemaker_train_deep.py"
        tar.add(entry_script, arcname="sagemaker_train_deep.py")
        tar.add(entry_script, arcname="aws/sagemaker_train_deep.py")

        # 2. Requirements
        req_file = PROJECT_ROOT / "aws" / "requirements-training.txt"
        tar.add(req_file, arcname="requirements.txt")

        # 3. Source directory
        src_dir = PROJECT_ROOT / "src"
        for file_p in src_dir.rglob("*.py"):
            arcname = file_p.relative_to(PROJECT_ROOT).as_posix()
            tar.add(file_p, arcname=arcname)

        # 4. Configs
        cfg_dir = PROJECT_ROOT / "configs"
        for file_p in cfg_dir.rglob("*.yaml"):
            arcname = file_p.relative_to(PROJECT_ROOT).as_posix()
            tar.add(file_p, arcname=arcname)

        # 5. Validated split manifests
        manifest_dir = PROJECT_ROOT / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "output"
        if manifest_dir.exists():
            for m_file in manifest_dir.rglob("*_split_manifest.json"):
                arcname = m_file.relative_to(PROJECT_ROOT).as_posix()
                tar.add(m_file, arcname=arcname)
                tar.add(m_file, arcname=f"manifests/{m_file.name}")

    hasher = hashlib.sha256()
    with open(output_tar_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()
    print(f"[PACKAGING] Archive created: {output_tar_path} ({output_tar_path.stat().st_size / 1024:.1f} KiB, SHA-256: {sha256_hash[:16]}...)")
    return sha256_hash


def upload_tarball_to_s3(local_path: Path, bucket: str, key: str, s3_client) -> str:
    """Uploads tarball to S3."""
    print(f"[S3] Uploading {local_path.name} to s3://{bucket}/{key}...")
    s3_client.upload_file(str(local_path), bucket, key)
    print(f"[S3] Upload complete: s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


def check_active_sagemaker_jobs(sm_client) -> int:
    """Checks the number of currently running SageMaker training jobs."""
    try:
        response = sm_client.list_training_jobs(StatusEquals="InProgress", MaxResults=10)
        jobs = response.get("TrainingJobSummaries", [])
        return len(jobs)
    except Exception as e:
        print(f"[WARN] Unable to list running training jobs: {e}")
        return 0


def wait_for_job_completion(sm_client, job_name: str, poll_interval: int = 30) -> dict:
    """Monitors SageMaker training job until completion and logs status."""
    print(f"\n[MONITOR] Monitoring SageMaker training job '{job_name}'...")
    start_time = time.time()
    last_status = None

    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary_status = desc.get("SecondaryStatus", "")

        if status != last_status or secondary_status != desc.get("_last_sec", ""):
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status} ({secondary_status}) | Elapsed: {elapsed:.0f}s")
            last_status = status
            desc["_last_sec"] = secondary_status

        if status in ["Completed", "Failed", "Stopped"]:
            elapsed = time.time() - start_time
            billable_seconds = desc.get("BillableTimeInSeconds", int(elapsed))
            print(f"\n[MONITOR] Training job finished with status: {status}")
            print(f"  Wall-clock Duration: {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")
            print(f"  Billable Time:       {billable_seconds}s ({billable_seconds / 60.0:.1f} min)")
            billable_cost = (billable_seconds / 3600.0) * 0.7364
            print(f"  Billable Cost:       ~${billable_cost:.4f} USD")

            if status == "Failed":
                failure_reason = desc.get("FailureReason", "Unknown error")
                print(f"\n[ERROR] Training job failed: {failure_reason}")
                raise RuntimeError(f"SageMaker training job failed: {failure_reason}")

            return desc

        time.sleep(poll_interval)


def submit_deep_sagemaker_job(
    category: str = "pipe_clip",
    model_type: str = "reverse_distillation",
    backbone: str = "wide_resnet50_2",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-4,
    instance_type: str = "ml.g4dn.xlarge",
    role_arn: str = "arn:aws:iam::077578771161:role/service-role/AmazonSageMaker-ExecutionRole-20261115T084659",
    profile: str = "autovi",
    region: str = "us-east-1",
    bucket: str = "sagemaker-us-east-1-077578771161",
    dry_run: bool = True,
    wait: bool = False,
):
    sm_client = None
    s3_client = None
    try:
        session = boto3.Session(profile_name=profile, region_name=region)
        sm_client = session.client("sagemaker")
        s3_client = session.client("s3")
    except Exception as e:
        if dry_run:
            print(f"[DRY-RUN NOTICE] AWS credentials not mounted in current environment: {e}")
            print("                 Proceeding with offline schema, payload, and cost verification.")
        else:
            raise

    s3_data_prefix = "autovi"
    s3_output_prefix = "autovi/phase3_outputs"
    s3_deploy_prefix = "autovi/deployment"

    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"

    # Estimate runtime and cost: ml.g4dn.xlarge is $0.7364/hr ($0.0123/min)
    est_minutes = 8 if model_type == "reverse_distillation" else 12
    est_cost_usd = (est_minutes / 60.0) * 0.7364

    print("\n" + "=" * 75)
    print("      AutoVI Phase 3 — SageMaker Deep Anomaly Training Submission      ")
    print("=" * 75)
    print(f"  Mode:            {'[DRY-RUN VALIDATION (Zero Cost)]' if dry_run else '[LIVE SUBMISSION (Billable)]'}")
    print(f"  Model Type:      {model_type.upper()}")
    print(f"  Category:        {category}")
    print(f"  Teacher/Backbone:{backbone if model_type == 'reverse_distillation' else 'U-Net'}")
    print(f"  Epochs / Batch:  {epochs} epochs / batch size {batch_size}")
    print(f"  Instance Type:   {instance_type} (1x NVIDIA Tesla T4 GPU, 16 GiB VRAM)")
    print(f"  Execution Role:  {role_arn}")
    print(f"  Container URI:   {image_uri}")
    print(f"  S3 Data Input:   s3://{bucket}/{s3_data_prefix}/")
    print(f"  S3 Output Path:  s3://{bucket}/{s3_output_prefix}/")
    print(f"  Estimated Time:  ~{est_minutes} minutes")
    print(f"  Estimated Cost:  ~${est_cost_usd:.3f} USD (Rate: $0.7364/hr)")
    print("=" * 75 + "\n")

    # Step 1: Concurrency and Quota check
    if sm_client is not None:
        active_jobs = check_active_sagemaker_jobs(sm_client)
        if active_jobs > 0:
            print(f"[QUOTA ALERT] Found {active_jobs} active SageMaker training job(s).")
            print("              The account quota is 1 ml.g4dn.xlarge instance.")
            if not dry_run:
                raise RuntimeError("Cannot launch training job: another training job is currently in progress.")
        else:
            print("[QUOTA CHECK] 0 active training jobs. 1 ml.g4dn.xlarge instance slot available.")
    else:
        print("[QUOTA CHECK] Offline dry-run: Account concurrency limit is 1 ml.g4dn.xlarge instance.")

    # Step 2: Deployment package check / creation
    code_s3_key = f"{s3_deploy_prefix}/p3_deep_sourcedir.tar.gz"
    code_s3_uri = f"s3://{bucket}/{code_s3_key}"

    if dry_run:
        if s3_client is not None:
            print("[DRY-RUN] Validating S3 prefix access...")
            try:
                s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{s3_data_prefix}/", MaxKeys=1)
                print(f"[PASS] S3 prefix 's3://{bucket}/{s3_data_prefix}/' is accessible.")
            except Exception as e:
                print(f"[FAIL] S3 prefix inaccessible: {e}")
                raise
        else:
            print(f"[DRY-RUN] Target S3 bucket '{bucket}' configured.")
    else:
        tar_local = PROJECT_ROOT / "p3_deep_sourcedir.tar.gz"
        create_deep_source_tarball(tar_local)
        code_s3_uri = upload_tarball_to_s3(tar_local, bucket=bucket, key=code_s3_key, s3_client=s3_client)
        if tar_local.exists():
            tar_local.unlink()

    # Step 3: Build Training Job Spec
    job_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    short_model = "rd" if model_type == "reverse_distillation" else "draem"
    cat_short = category.replace("_", "")
    job_name = f"autovi-p3-{short_model}-{cat_short}-{job_timestamp}"

    hyperparameters = {
        "model-type": model_type,
        "category": category,
        "backbone": backbone,
        "epochs": str(epochs),
        "batch-size": str(batch_size),
        "lr": str(lr),
        "seed": "42",
        "sagemaker_program": "sagemaker_train_deep.py",
        "sagemaker_submit_directory": code_s3_uri,
        "sagemaker_container_log_level": "20",
    }

    job_spec = {
        "TrainingJobName": job_name,
        "HyperParameters": hyperparameters,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": False,
        },
        "RoleArn": role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{s3_data_prefix}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{s3_output_prefix}/",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600,  # 1 hour max safety cutoff
        },
        "EnableNetworkIsolation": False,
    }

    if dry_run:
        print("[DRY-RUN] Job specification constructed successfully:")
        print(f"  TrainingJobName: {job_name}")
        print(f"  Entry Point:     aws/sagemaker_train_deep.py")
        print(f"  Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
        print("\n[SUCCESS] Dry run passed! Zero AWS charges incurred.")
        print(f"To launch paid training after user approval, execute:")
        print(f"python autovi_project/aws/submit_deep_training.py --category {category} --model-type {model_type} --execute")
        return None

    # Step 4: Live Submission
    print(f"\n[SUBMISSION] Launching live SageMaker training job: {job_name}...")
    try:
        response = sm_client.create_training_job(**job_spec)
        print(f"[SUCCESS] Training job launched! ARN: {response['TrainingJobArn']}")
        if wait:
            return wait_for_job_completion(sm_client, job_name, poll_interval=20)
        return job_name
    except ClientError as e:
        print(f"[ERROR] Failed to submit SageMaker training job ({e.response['Error']['Code']}):")
        print(f"        {e.response['Error']['Message']}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Phase 3 Deep Anomaly Training Job")
    parser.add_argument("--category", type=str, default="pipe_clip", choices=["pipe_clip", "tank_screw", "engine_wiring"])
    parser.add_argument("--model-type", type=str, default="reverse_distillation", choices=["reverse_distillation", "draem"])
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--execute", action="store_true", help="Launch live billable job (requires user approval)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Perform dry-run checks only")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for job completion")

    args = parser.parse_args()
    is_live = args.execute

    submit_deep_sagemaker_job(
        category=args.category,
        model_type=args.model_type,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=(not is_live),
        wait=(not args.no_wait),
    )
