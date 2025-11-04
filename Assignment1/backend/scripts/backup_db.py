#!/usr/bin/env python3
"""Database backup script for manual or scheduled backups."""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.backup import DatabaseBackup
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Create a database backup."""
    logger.info("Starting database backup...")

    try:
        backup_manager = DatabaseBackup()

        # Create backup
        backup_path = backup_manager.create_backup(compress=True)
        logger.info(f"Backup created successfully: {backup_path}")

        # Get backup stats
        stats = backup_manager.get_backup_stats()
        logger.info(f"Total backups: {stats['total_backups']}")
        logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")

        # Cleanup old backups
        deleted = backup_manager.cleanup_old_backups()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old backup files")

        logger.info("Backup completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
