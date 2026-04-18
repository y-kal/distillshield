from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from distillshield_core.config import get_settings


settings = get_settings()
engine = create_engine(settings.db_url, echo=False)


def get_engine():
    return engine


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
