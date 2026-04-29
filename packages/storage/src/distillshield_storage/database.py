from contextlib import contextmanager

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from distillshield_core.config import get_settings


settings = get_settings()
engine = create_engine(settings.db_url, echo=False)


def get_engine():
    return engine


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_risk_assessment_entity()


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session


def _migrate_risk_assessment_entity() -> None:
    inspector = inspect(engine)
    if "riskassessmententity" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("riskassessmententity")}
    required_columns = {
        "explainability": "JSON DEFAULT '{}'",
        "category_scores": "JSON DEFAULT '{}'",
        "top_reasons": "JSON DEFAULT '[]'",
        "triggered_rules": "JSON DEFAULT '[]'",
        "risk_reducers": "JSON DEFAULT '[]'",
    }

    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE riskassessmententity ADD COLUMN {column_name} {column_type}"))
