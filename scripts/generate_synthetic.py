from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import bootstrap_repo_packages

bootstrap_repo_packages()

from distillshield_core.config import get_settings
from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_storage.database import create_db_and_tables, get_session
from distillshield_storage.repository import upsert_session
from distillshield_synthetic_data.generator import SyntheticDataGenerator


def main() -> None:
    settings = get_settings()
    create_db_and_tables()
    generator = SyntheticDataGenerator(seed=7)
    splits = generator.dataset_splits(num_users=40, sessions_per_user=3)
    output_dir = Path(settings.data_dir) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    with get_session() as db:
        for split_name, sessions in splits.items():
            for session in sessions:
                upsert_session(db, session)
            (output_dir / f"{split_name}.json").write_text(json.dumps([session.model_dump(mode="json") for session in sessions], indent=2))
    print(f"Saved synthetic data to {output_dir}")


if __name__ == "__main__":
    main()
