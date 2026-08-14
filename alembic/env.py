import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env -- pydantic-settings loads .env
                # internally for Settings only, it never touches os.environ itself

# Make app/ importable, matching the pattern used everywhere else in this project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from config import settings  # noqa: E402
from db import Base  # noqa: E402
import db  # noqa: E402  (imports AgentSessionORM, AgentRunORM, AgentStepORM so they register on Base.metadata)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Pick which database this migration run targets, via a command-line flag
# instead of editing .env by hand (which is exactly the step that's caused
# repeated "forgot to migrate the other database" bugs):
#
#   alembic upgrade head              -> targets DATABASE_URL (production/Neon)
#   alembic -x db=test upgrade head   -> targets TEST_DATABASE_URL (local test db)
x_args = context.get_x_argument(as_dictionary=True)
target = x_args.get("db", "main")

if target == "test":
    db_url = os.environ.get("TEST_DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "TEST_DATABASE_URL is not set in .env -- cannot target the test database."
        )
else:
    db_url = settings.database_url

config.set_main_option("sqlalchemy.url", db_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()