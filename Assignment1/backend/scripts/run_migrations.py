#!/usr/bin/env python3
"""Database migration script for applying schema updates."""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.migrations import run_migrations, get_migration_status
from app.utils.backup import DatabaseBackup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run database migrations."""
    logger.info("Starting database migrations...")

    # Create backup before migrations
    try:
        logger.info("Creating pre-migration backup...")
        backup_manager = DatabaseBackup()
        backup_path = backup_manager.create_backup(compress=True)
        logger.info(f"Pre-migration backup created: {backup_path}")
    except Exception as e:
        logger.warning(f"Pre-migration backup failed: {e}")
        response = input("Continue without backup? (y/N): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled")
            return 1

    # Show current migration status
    logger.info("Current migration status:")
    status = get_migration_status()
    for migration in status:
        applied_str = "✓ Applied" if migration["applied"] else "✗ Pending"
        logger.info(f"  {applied_str}: {migration['migration_id']} - {migration['description']}")

    # Run migrations
    try:
        applied = run_migrations()
        logger.info(f"Migrations completed successfully. Applied {applied} migrations.")
        return 0
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        logger.error("You may need to restore from backup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
