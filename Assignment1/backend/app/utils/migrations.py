"""Database migration utilities for schema updates."""
import logging
from typing import List, Callable
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal
from app.config import settings

logger = logging.getLogger(__name__)


class Migration:
    """
    Represents a database migration.

    Each migration has an ID, description, and up/down functions.
    """

    def __init__(
        self,
        migration_id: str,
        description: str,
        up: Callable[[Session], None],
        down: Callable[[Session], None],
    ):
        """
        Initialize migration.

        Args:
            migration_id: Unique migration identifier (e.g., "001_add_session_fields")
            description: Human-readable description
            up: Function to apply migration
            down: Function to rollback migration
        """
        self.migration_id = migration_id
        self.description = description
        self.up = up
        self.down = down


class MigrationManager:
    """
    Manage database migrations and versioning.

    Tracks applied migrations and provides methods to apply/rollback changes.
    """

    def __init__(self):
        """Initialize migration manager."""
        self.migrations: List[Migration] = []
        self._ensure_migrations_table()

    def _ensure_migrations_table(self) -> None:
        """Create migrations table if it doesn't exist."""
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_id TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rolled_back_at TIMESTAMP NULL
                )
            """))
            conn.commit()
        logger.info("Migrations table ensured")

    def register(self, migration: Migration) -> None:
        """
        Register a migration.

        Args:
            migration: Migration to register
        """
        self.migrations.append(migration)
        logger.debug(f"Registered migration: {migration.migration_id}")

    def is_applied(self, migration_id: str) -> bool:
        """
        Check if migration has been applied.

        Args:
            migration_id: Migration identifier

        Returns:
            True if applied, False otherwise
        """
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE migration_id = :migration_id AND rolled_back_at IS NULL"
                ),
                {"migration_id": migration_id},
            )
            count = result.scalar()
            return count > 0

    def apply_migration(self, migration: Migration) -> None:
        """
        Apply a migration.

        Args:
            migration: Migration to apply

        Raises:
            Exception: If migration fails
        """
        if self.is_applied(migration.migration_id):
            logger.info(f"Migration {migration.migration_id} already applied, skipping")
            return

        logger.info(f"Applying migration {migration.migration_id}: {migration.description}")

        db = SessionLocal()
        try:
            # Run migration
            migration.up(db)
            db.commit()

            # Record migration
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO schema_migrations (migration_id, description) "
                        "VALUES (:migration_id, :description)"
                    ),
                    {
                        "migration_id": migration.migration_id,
                        "description": migration.description,
                    },
                )
                conn.commit()

            logger.info(f"Migration {migration.migration_id} applied successfully")

        except Exception as e:
            db.rollback()
            logger.error(f"Migration {migration.migration_id} failed: {e}")
            raise
        finally:
            db.close()

    def rollback_migration(self, migration: Migration) -> None:
        """
        Rollback a migration.

        Args:
            migration: Migration to rollback

        Raises:
            Exception: If rollback fails
        """
        if not self.is_applied(migration.migration_id):
            logger.info(f"Migration {migration.migration_id} not applied, nothing to rollback")
            return

        logger.info(f"Rolling back migration {migration.migration_id}")

        db = SessionLocal()
        try:
            # Run rollback
            migration.down(db)
            db.commit()

            # Mark as rolled back
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "UPDATE schema_migrations "
                        "SET rolled_back_at = CURRENT_TIMESTAMP "
                        "WHERE migration_id = :migration_id"
                    ),
                    {"migration_id": migration.migration_id},
                )
                conn.commit()

            logger.info(f"Migration {migration.migration_id} rolled back successfully")

        except Exception as e:
            db.rollback()
            logger.error(f"Rollback of {migration.migration_id} failed: {e}")
            raise
        finally:
            db.close()

    def apply_all(self) -> int:
        """
        Apply all pending migrations.

        Returns:
            Number of migrations applied
        """
        applied_count = 0

        for migration in self.migrations:
            if not self.is_applied(migration.migration_id):
                self.apply_migration(migration)
                applied_count += 1

        logger.info(f"Applied {applied_count} migrations")
        return applied_count

    def get_migration_status(self) -> List[dict]:
        """
        Get status of all migrations.

        Returns:
            List of migration status dictionaries
        """
        status_list = []

        for migration in self.migrations:
            is_applied = self.is_applied(migration.migration_id)

            status_list.append({
                "migration_id": migration.migration_id,
                "description": migration.description,
                "applied": is_applied,
            })

        return status_list


# Create global migration manager
migration_manager = MigrationManager()


# Define migrations
def migration_001_add_session_fields_up(db: Session) -> None:
    """Add session management fields to users table."""
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("users")]

    # Add columns if they don't exist
    if "last_activity" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ))
        logger.info("Added last_activity column")

    if "session_expires_at" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN session_expires_at TIMESTAMP NULL"
        ))
        logger.info("Added session_expires_at column")

    if "is_active" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1"
        ))
        logger.info("Added is_active column")

    if "api_key_created_at" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN api_key_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ))
        logger.info("Added api_key_created_at column")

    if "api_key_expires_at" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN api_key_expires_at TIMESTAMP NULL"
        ))
        logger.info("Added api_key_expires_at column")

    if "is_admin" not in columns:
        db.execute(text(
            "ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0"
        ))
        logger.info("Added is_admin column")


def migration_001_add_session_fields_down(db: Session) -> None:
    """Remove session management fields from users table."""
    # SQLite doesn't support DROP COLUMN directly, so we'd need to recreate the table
    # For now, just log that rollback isn't fully supported
    logger.warning("Rollback for this migration not fully implemented (SQLite limitation)")


# Register migrations
migration_manager.register(
    Migration(
        migration_id="001_add_session_fields",
        description="Add session management and API key fields to users table",
        up=migration_001_add_session_fields_up,
        down=migration_001_add_session_fields_down,
    )
)


def run_migrations() -> int:
    """
    Run all pending migrations.

    Returns:
        Number of migrations applied
    """
    logger.info("Running database migrations...")
    try:
        applied = migration_manager.apply_all()
        logger.info(f"Migrations complete. Applied {applied} migrations.")
        return applied
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


def get_migration_status() -> List[dict]:
    """
    Get status of all migrations.

    Returns:
        List of migration status dictionaries
    """
    return migration_manager.get_migration_status()
