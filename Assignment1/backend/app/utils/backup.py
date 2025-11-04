"""Database backup and restore utilities."""
import shutil
import logging
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """
    Handle SQLite database backup and restore operations.

    Supports full backups, compression, and automated cleanup.
    """

    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize database backup manager.

        Args:
            backup_dir: Directory to store backups (uses config if not provided)
        """
        self.backup_dir = Path(backup_dir or settings.BACKUP_DIRECTORY)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self._get_db_path()

    def _get_db_path(self) -> Path:
        """
        Get database file path from DATABASE_URL.

        Returns:
            Path to database file
        """
        db_url = settings.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            # Handle relative paths
            if not db_path.startswith("/"):
                db_path = Path(".") / db_path
            return Path(db_path)
        else:
            raise ValueError(f"Unsupported database URL: {db_url}")

    def create_backup(self, compress: bool = True) -> Path:
        """
        Create a backup of the database.

        Args:
            compress: Whether to compress the backup with gzip

        Returns:
            Path to the backup file

        Raises:
            FileNotFoundError: If database file doesn't exist
            IOError: If backup fails
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        # Generate backup filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"ollama_web_backup_{timestamp}.db"

        if compress:
            backup_name += ".gz"

        backup_path = self.backup_dir / backup_name

        try:
            logger.info(f"Creating database backup: {backup_path}")

            if compress:
                # Create compressed backup
                with open(self.db_path, "rb") as f_in:
                    with gzip.open(backup_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Create uncompressed backup
                shutil.copy2(self.db_path, backup_path)

            # Also backup WAL file if it exists
            wal_path = Path(str(self.db_path) + "-wal")
            if wal_path.exists():
                wal_backup_name = f"ollama_web_backup_{timestamp}.db-wal"
                if compress:
                    wal_backup_name += ".gz"
                wal_backup_path = self.backup_dir / wal_backup_name

                if compress:
                    with open(wal_path, "rb") as f_in:
                        with gzip.open(wal_backup_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(wal_path, wal_backup_path)

            logger.info(f"Database backup created successfully: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            raise IOError(f"Backup failed: {e}")

    def restore_backup(self, backup_path: Path, force: bool = False) -> None:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file
            force: Force restore even if current database exists

        Raises:
            FileNotFoundError: If backup file doesn't exist
            IOError: If restore fails
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        if self.db_path.exists() and not force:
            raise IOError(
                "Database file already exists. Use force=True to overwrite."
            )

        try:
            logger.info(f"Restoring database from backup: {backup_path}")

            # Create backup of current database if it exists
            if self.db_path.exists():
                current_backup = self.db_path.with_suffix(".db.before_restore")
                shutil.copy2(self.db_path, current_backup)
                logger.info(f"Current database backed up to: {current_backup}")

            # Restore from backup
            if backup_path.suffix == ".gz":
                # Decompress and restore
                with gzip.open(backup_path, "rb") as f_in:
                    with open(self.db_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(backup_path, self.db_path)

            logger.info("Database restored successfully")

        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            raise IOError(f"Restore failed: {e}")

    def list_backups(self) -> List[dict]:
        """
        List all available backups.

        Returns:
            List of backup information dictionaries
        """
        backups = []

        for backup_file in sorted(self.backup_dir.glob("ollama_web_backup_*.db*")):
            # Skip WAL files
            if "-wal" in backup_file.name:
                continue

            backup_info = {
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_bytes": backup_file.stat().st_size,
                "size_mb": backup_file.stat().st_size / (1024 * 1024),
                "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime),
                "compressed": backup_file.suffix == ".gz",
            }
            backups.append(backup_info)

        return sorted(backups, key=lambda x: x["created_at"], reverse=True)

    def cleanup_old_backups(self, retention_days: Optional[int] = None) -> int:
        """
        Remove backups older than retention period.

        Args:
            retention_days: Number of days to retain backups (uses config if not provided)

        Returns:
            Number of backups deleted
        """
        retention_days = retention_days or settings.BACKUP_RETENTION_DAYS
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        deleted_count = 0
        backups = self.list_backups()

        for backup in backups:
            if backup["created_at"] < cutoff_date:
                try:
                    backup_path = Path(backup["path"])
                    backup_path.unlink()
                    logger.info(f"Deleted old backup: {backup['filename']}")
                    deleted_count += 1

                    # Also delete corresponding WAL file if it exists
                    wal_path = Path(str(backup_path).replace(".db", ".db-wal"))
                    if wal_path.exists():
                        wal_path.unlink()
                        deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to delete backup {backup['filename']}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backup files")

        return deleted_count

    def get_backup_stats(self) -> dict:
        """
        Get backup statistics.

        Returns:
            Dictionary with backup statistics
        """
        backups = self.list_backups()

        if not backups:
            return {
                "total_backups": 0,
                "total_size_mb": 0,
                "oldest_backup": None,
                "newest_backup": None,
            }

        total_size = sum(b["size_bytes"] for b in backups)

        return {
            "total_backups": len(backups),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_backup": backups[-1]["created_at"],
            "newest_backup": backups[0]["created_at"],
            "backup_directory": str(self.backup_dir),
        }


def create_backup_job() -> Optional[Path]:
    """
    Create a database backup (used for scheduled jobs).

    Returns:
        Path to backup file or None if backup is disabled
    """
    if not settings.BACKUP_ENABLED:
        logger.info("Database backup is disabled")
        return None

    try:
        backup_manager = DatabaseBackup()
        backup_path = backup_manager.create_backup(compress=True)

        # Cleanup old backups
        deleted = backup_manager.cleanup_old_backups()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old backup files")

        return backup_path

    except Exception as e:
        logger.error(f"Backup job failed: {e}", exc_info=True)
        return None
