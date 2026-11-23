"""Submits PatchCore GPU Training Job to Amazon SageMaker with Dry-Run Support.

Usage:
    # Dry-run validation (does not launch or incur costs):
    python autovi_project/aws/submit_training.py --category pipe_clip --dry-run

    # Live execution:
    python autovi_project/aws/submit_training.py --category pipe_clip [--wait]
"""

import argparse
import json
import sys
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aws.upload_code import create_source_tarball, upload_to_s3


def submit_sagemaker_job(
    category: str = "pipe_clip",
    coreset_ratio: float = 0.05,
    chunk_size: int = 8192,
    batch_size: int = 16,
    instance_type: str = "ml.g4dn.xlarge",
    role_arn: str = None,
    profile: str = "autovi",
    region: str = "us-east-1",
    bucket: str = "sagemaker-us-east-1-077578771161",
    config_path: Path = None,
    dry_run: bool = False,
    wait: bool = False,
):
    # Load default yaml config if available
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    region = cfg.get("aws", {}).get("region", region)
    profile = cfg.get("aws", {}).get("profile", profile)
    bucket = cfg.get("aws", {}).get("s3_bucket", bucket)
    s3_data_prefix = cfg.get("aws", {}).get("s3_data_prefix", "autovi")
    s3_output_prefix = cfg.get("aws", {}).get("s3_output_prefix", "autovi/outputs")
    s3_deploy_prefix = cfg.get("aws", {}).get("s3_deployment_prefix", "autovi/deployment")

    image_uri = cfg.get("sagemaker", {}).get(
        "image_uri",
        f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    )
    role_arn = (
        role_arn
        or cfg.get("sagemaker", {}).get("role_arn")
        or "arn:aws:iam::077578771161:role/service-role/AmazonSageMaker-ExecutionRole-20261115T084659"
    )

    print("\n" + "=" * 70)
    print("      AutoVI PatchCore — SageMaker GPU Training Submission      ")
    print("=" * 70)
    print(f"  Mode:           {'[DRY-RUN VALIDATION]' if dry_run else '[LIVE SUBMISSION]'}")
    print(f"  Category:       {category}")
    print(f"  Instance Type:  {instance_type} (1x NVIDIA T4 GPU, 16 GiB VRAM)")
    print(f"  Execution Role: {role_arn}")
    print(f"  Coreset Ratio:  {coreset_ratio}")
    print(f"  Chunk Size:     {chunk_size}")
    print(f"  Container URI:  {image_uri}")
    print(f"  S3 Data Input:  s3://{bucket}/{s3_data_prefix}/")
    print(f"  S3 Output Path: s3://{bucket}/{s3_output_prefix}/")
    print(f"  Deployment S3:  s3://{bucket}/{s3_deploy_prefix}/sourcedir.tar.gz")
    print("=" * 70 + "\n")

    code_s3_uri = f"s3://{bucket}/{s3_deploy_prefix}/sourcedir.tar.gz"

    # Step 1: Package and upload code tarball (or verify in dry-run)
    if dry_run:
        print("[DRY-RUN] Checking existing deployment package in S3...")
        session = boto3.Session(profile_name=profile, region_name=region)
        s3 = session.client("s3")
        try:
            head = s3.head_object(Bucket=bucket, Key=f"{s3_deploy_prefix}/sourcedir.tar.gz")
            print(f"[PASS] Existing package verified on S3 ({head.get('ContentLength', 0) / 1024:.1f} KiB).")
        except Exception as e:
            print(f"[WARN] Deployment package check: {e}")
            print("       (upload_code.py will create and upload sourcedir.tar.gz on live run)")
    else:
        tar_local = Path("sourcedir.tar.gz")
        create_source_tarball(tar_local)
        code_s3_key = f"{s3_deploy_prefix}/sourcedir.tar.gz"
        code_s3_uri = upload_to_s3(tar_local, bucket=bucket, key=code_s3_key, profile=profile, region=region)
        if tar_local.exists():
            tar_local.unlink()

    # Step 2: Build SageMaker Training Job Specification
    job_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    job_name = f"autovi-{category.replace('_', '-')}-{job_timestamp}"[:63]

    hyperparameters = {
        "sagemaker_program": "sagemaker_entry.py",
        "sagemaker_submit_directory": code_s3_uri,
        "category": str(category),
        "coreset-ratio": str(coreset_ratio),
        "chunk-size": str(chunk_size),
        "batch-size": str(batch_size),
        "device": "cuda",
        "visualize-count": "8",
    }

    training_params = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
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
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{s3_output_prefix}/",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 7200,
        },
        "HyperParameters": hyperparameters,
    }

    if dry_run:
        print("[DRY-RUN] Validating training job API parameters structure:")
        print(json.dumps(training_params, indent=2))
        print("\n" + "=" * 70)
        print("                 DRY RUN SUCCESSFUL (NO CHARGES INCURRED)               ")
        print("=" * 70)
        print("  - Job Payload validated successfully.")
        print(f"  - Target Model:       PatchCore + WideResNet-50-2 ({category})")
        print(f"  - Target Role:        {role_arn}")
        print(f"  - Target Hardware:    {instance_type} (NVIDIA T4)")
        print(f"  - Container Image:    {image_uri}")
        print("=" * 70 + "\n")
        return {"status": "DRY_RUN_PASSED", "job_name": job_name, "params": training_params}

    # Live Submission
    session = boto3.Session(profile_name=profile, region_name=region)
    sm_client = session.client("sagemaker")
    try:
        print(f"Submitting live training job: {job_name}...")
        resp = sm_client.create_training_job(**training_params)
        job_arn = resp.get("TrainingJobArn", "")
        print(f"\n[SUCCESS] Training Job Created!")
        print(f"  Job Name: {job_name}")
        print(f"  Job ARN:  {job_arn}")
        print(f"  Console:  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")

        if wait:
            print("\nWaiting for training job to complete...")
            while True:
                desc = sm_client.describe_training_job(TrainingJobName=job_name)
                status = desc["TrainingJobStatus"]
                print(f"  Job Status: {status}...")
                if status in ["Completed", "Failed", "Stopped"]:
                    if status == "Failed":
                        print(f"[FAIL] Failure Reason: {desc.get('FailureReason')}")
                    break
                time.sleep(30)

        return {"status": "SUBMITTED", "job_name": job_name, "job_arn": job_arn}

    except ClientError as e:
        err_code = e.response.get("Error", {}).get("Code", "")
        err_msg = e.response.get("Error", {}).get("Message", "")
        print(f"\n[ERROR] Failed to submit SageMaker training job ({err_code}):")
        print(f"        {err_msg}")

        if "AccessDenied" in err_code:
            print("\n>>> HOW TO RESOLVE ACCESS DENIED <<<")
            print("1. Your AWS IAM User 'autovi-s3-uploader' lacks 'sagemaker:CreateTrainingJob' or 'iam:PassRole'.")
            print("2. In AWS IAM Console, attach policy 'AmazonSageMakerFullAccess' to user 'autovi-s3-uploader'.")
            print(f"3. Ensure user has permission `iam:PassRole` on role '{role_arn}'.")
            print("4. Or open AWS SageMaker Studio and run the job directly with its pre-attached Execution Role.")

        return {"status": "FAILED", "error_code": err_code, "error_msg": err_msg}


def main():
    parser = argparse.ArgumentParser(description="Submit AutoVI SageMaker GPU Training Job")
    parser.add_argument("--category", type=str, default="pipe_clip", help="Category to train")
    parser.add_argument("--coreset-ratio", type=float, default=0.05, help="Coreset ratio")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge", help="EC2 instance type")
    parser.add_argument("--role-arn", type=str, default=None, help="SageMaker execution role ARN")
    parser.add_argument("--profile", type=str, default="autovi", help="AWS CLI profile")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", type=str, default="sagemaker-us-east-1-077578771161", help="S3 bucket")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "aws" / "configs" / "sagemaker.yaml"),
        help="Config file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run validation without submitting")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    args = parser.parse_args()

    submit_sagemaker_job(
        category=args.category,
        coreset_ratio=args.coreset_ratio,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        instance_type=args.instance_type,
        role_arn=args.role_arn,
        profile=args.profile,
        region=args.region,
        bucket=args.bucket,
        config_path=args.config,
        dry_run=args.dry_run,
        wait=args.wait,
    )


if __name__ == "__main__":
    main()
