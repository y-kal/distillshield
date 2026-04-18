from __future__ import annotations

import json

from _bootstrap import bootstrap_repo_packages

bootstrap_repo_packages()

from distillshield_eval.runner import EvaluationRunner
from distillshield_storage.database import create_db_and_tables, get_session
from distillshield_storage.repository import save_experiment_run


def main() -> None:
    create_db_and_tables()
    runner = EvaluationRunner()
    result = runner.run(seed=11, num_users=40, sessions_per_user=3)
    with get_session() as db:
        save_experiment_run(db, result["experiment_id"], result["metrics"], result["artifact_paths"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
