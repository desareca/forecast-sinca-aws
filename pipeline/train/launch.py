"""Lanzador manual del Training Job del baseline (SageMaker, CPU `ml.m5.large`).

Sigue el patrón del blueprint `compute`: la imagen ECR lleva **solo el runtime**
(LightGBM + pandas + numpy + scikit-learn + mlflow + boto3 + pyarrow), y el
código viaja por `source_dir`/`entry_point` desde `pipeline/` (el SDK empaqueta
`pipeline/` y lo sube a `code_location`, que apunta al bucket de tracking).

Disparo **manual** (Fase 3): no hay pipeline automatizado todavía.

Requiere **SageMaker Python SDK v2** (`sagemaker>=2,<3`): la v3 (3.x) eliminó
`Estimator`/`Model`/`Predictor`. Las deps del ambiente local están en
`requirements-dev.txt` (raíz del repo, fuera de `pipeline/` para que el toolkit
`sagemaker-training` no las instale en el contenedor).

Uso:

    python pipeline/train/launch.py \
        --profile $AWS_PROFILE \
        --role-arn <arn de training_role_arn (output de Terraform)> \
        --image-uri <url de train_ecr_repository_url>:latest \
        --wait

Los ARNs salen de los outputs de Terraform (`training_role_arn`,
`train_ecr_repository_url`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
ENTRY_POINT = "train/train.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lanza el Training Job del baseline (Fase 3).")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--role-arn", default=os.environ.get("TRAINING_ROLE_ARN"))
    parser.add_argument("--image-uri", default=os.environ.get("TRAIN_IMAGE_URI"))
    parser.add_argument("--instance-type", default=os.environ.get("TRAIN_INSTANCE_TYPE", "ml.m5.large"))
    parser.add_argument("--data-bucket", default=os.environ.get("DATA_BUCKET", "sinca-data"))
    parser.add_argument("--mlflow-bucket", default=os.environ.get("MLFLOW_BUCKET", "sinca-mlflow"))
    parser.add_argument("--mlflow-prefix", default=os.environ.get("MLFLOW_PREFIX", "_mlflow"))
    parser.add_argument("--search-trials", type=int, default=20)
    parser.add_argument("--search-windows", type=int, default=12)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--max-run", type=int, default=7200, help="Timeout del job en segundos (2 h).")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--wait", action="store_true", help="Esperar a que el job termine.")
    parser.add_argument("--no-logs", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Armar el Estimator y mostrar la config, sin lanzar el job.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # En Windows el stdout suele ser cp1252 y el streaming de logs de CloudWatch
    # incluye caracteres no representables -> UnicodeEncodeError. Forzamos UTF-8
    # para que el launcher no muera mientras el job sigue corriendo.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.role_arn:
        raise SystemExit("Falta --role-arn (o la env TRAINING_ROLE_ARN).")
    if not args.image_uri:
        raise SystemExit("Falta --image-uri (o la env TRAIN_IMAGE_URI).")

    import boto3
    from sagemaker.estimator import Estimator
    from sagemaker.session import Session

    boto_session = boto3.Session(profile_name=args.profile or None, region_name=args.region)
    session = Session(boto_session=boto_session)

    code_location = f"s3://{args.mlflow_bucket}/{args.mlflow_prefix}/code"
    output_path = f"s3://{args.mlflow_bucket}/model-artifacts"

    estimator = Estimator(
        image_uri=args.image_uri,
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        entry_point=ENTRY_POINT,
        source_dir=str(PIPELINE_DIR),
        code_location=code_location,
        output_path=output_path,
        base_job_name="sinca-baseline-train",
        hyperparameters={
            "data-bucket": args.data_bucket,
            "mlflow-bucket": args.mlflow_bucket,
            "mlflow-prefix": args.mlflow_prefix,
            "search-trials": args.search_trials,
            "search-windows": args.search_windows,
            "version": args.version,
        },
        max_run=args.max_run,
        volume_size=30,
        disable_profiler=True,
        sagemaker_session=session,
    )

    print(f"Training Job: image={args.image_uri} instance={args.instance_type}")
    print(f"  code_location={code_location} output_path={output_path}")

    if args.dry_run:
        hyperparameters = estimator.hyperparameters()
        print("  role_arn=" + args.role_arn)
        print("  entry_point=" + ENTRY_POINT + f"  source_dir={PIPELINE_DIR}")
        print("  hyperparameters=" + str(hyperparameters))
        print("  max_run=" + str(args.max_run))
        print("(--dry-run: no se lanzó el job)")
        return 0

    estimator.fit(job_name=args.job_name, wait=args.wait, logs=not args.no_logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
