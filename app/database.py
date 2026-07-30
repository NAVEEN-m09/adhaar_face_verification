from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

db_url = settings.DATABASE_URL
if db_url.startswith("sqlite"):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    if db_url.startswith("mysql"):
        try:
            # Extract database name from connection URL
            base_url, db_name = db_url.rsplit("/", 1)
            if "?" in db_name:
                db_name, _ = db_name.split("?", 1)
            
            # Connect to MySQL server instance without database name to execute DDL query
            temp_engine = create_engine(base_url)
            with temp_engine.connect() as conn:
                # Set isolation level to AUTOCOMMIT for DDL execution
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            temp_engine.dispose()
        except Exception as e:
            from app.utils.logger import logger
            logger.error(f"Database auto-creation failed: {str(e)}")
            
    engine = create_engine(db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    FastAPI dependency injection provider for transactional database sessions.
    Automatically closes session connections upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
