"""Database setup and session management."""
import logging
from sqlalchemy import create_engine, event, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

# Create SQLite engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.LOG_LEVEL == "DEBUG",
)


# Enable WAL mode for SQLite (better concurrency)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable SQLite WAL mode for better concurrency."""
    if settings.SQLITE_WAL_MODE:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        cursor.close()
        logger.info("SQLite WAL mode and optimizations enabled")


# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_indexes():
    """
    Create additional database indexes for performance optimization.

    This is called after initial table creation to add indexes that
    improve query performance for common operations.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)

    # Check if indexes already exist
    existing_indexes = set()
    for table_name in inspector.get_table_names():
        for index in inspector.get_indexes(table_name):
            existing_indexes.add(index['name'])

    # Define indexes to create
    indexes_to_create = [
        # Conversations indexes
        ("idx_conversations_user_updated", "conversations", ["user_id", "updated_at"]),
        ("idx_conversations_updated_at", "conversations", ["updated_at"]),

        # Messages indexes
        ("idx_messages_conversation_created", "messages", ["conversation_id", "created_at"]),
        ("idx_messages_created_at", "messages", ["created_at"]),
        ("idx_messages_role", "messages", ["role"]),

        # Settings indexes
        ("idx_settings_user_key", "settings", ["user_id", "key"]),
    ]

    # Create indexes if they don't exist
    with engine.connect() as conn:
        for index_name, table_name, columns in indexes_to_create:
            if index_name not in existing_indexes:
                try:
                    column_list = ", ".join(columns)
                    create_index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_list})"
                    conn.execute(create_index_sql)
                    conn.commit()
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to create index {index_name}: {e}")

    logger.info("Database indexes creation completed")


def init_db():
    """Initialize database tables and indexes."""
    logger.info("Initializing database...")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

    # Create indexes for optimization
    create_indexes()

    logger.info("Database initialization completed")
