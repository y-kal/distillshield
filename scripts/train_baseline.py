from __future__ import annotations

from _bootstrap import bootstrap_repo_packages

bootstrap_repo_packages()

from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_models.ml import BaselineModelTrainer
from distillshield_storage.database import create_db_and_tables, get_session
from distillshield_storage.repository import save_model_version
from distillshield_synthetic_data.generator import SyntheticDataGenerator


def main() -> None:
    create_db_and_tables()
    generator = SyntheticDataGenerator(seed=7)
    feature_pipeline = FeaturePipeline()
    rows = [feature_pipeline.to_frame_row(session) for session in generator.dataset_splits(num_users=60, sessions_per_user=4)["train"]]
    trainer = BaselineModelTrainer()
    artifacts = trainer.train_all(rows, seed=7)
    with get_session() as db:
        for artifact in artifacts:
            if artifact.available:
                save_model_version(db, artifact.name, "0.1.0", artifact.artifact_path, artifact.metrics)
                print(f"Stored model {artifact.name} at {artifact.artifact_path}")
            else:
                print(f"Skipped optional model {artifact.name}")


if __name__ == "__main__":
    main()
