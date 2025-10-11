"""
Database connection and models for the chatbot backend.
Connects to the same PostgreSQL database as pvwebapp (Django).
"""
import os
from typing import Optional
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ========================
# Database Configuration
# ========================
POSTGRES_DB = os.getenv("POSTGRES_DB", "planville")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before using
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ========================
# Models (matching Django models)
# ========================
class LeadType(Base):
    """Maps to customers_leadtype table in Django"""
    __tablename__ = "customers_leadtype"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    project_type_id = Column(Integer, nullable=True)  # ForeignKey to django_content_type


class LeadChatbot(Base):
    """Maps to customers_leadchatbot table in Django"""
    __tablename__ = "customers_leadchatbot"

    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String(50), nullable=True)
    first_name = Column(Text, nullable=True)
    last_name = Column(Text, nullable=True)
    company = Column(Text, nullable=True)
    street_and_number = Column(Text, nullable=True)
    zip_and_city = Column(Text, nullable=True)
    province = Column(Text, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    email = Column(String(254), nullable=True)
    phone1 = Column(String(20), nullable=True)
    phone2 = Column(String(20), nullable=True)
    phone3 = Column(String(20), nullable=True)
    registration_datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    lead_type_id = Column(Integer, ForeignKey("customers_leadtype.id"), nullable=True)
    source = Column(Text, nullable=True, default="chatbot")

    # Processing tracking
    processed_datetime = Column(DateTime, nullable=True)
    lead_id = Column(Integer, nullable=True)  # ForeignKey to customers_lead

    # Metadata
    session_id = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)


# ========================
# Database Dependency
# ========================
def get_db() -> Session:
    """
    Dependency function to get a database session.
    Use with FastAPI Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for manual use.
    Remember to close it when done.
    """
    return SessionLocal()


# ========================
# Helper Functions
# ========================
def test_connection() -> bool:
    """Test if database connection is working"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False


def create_leadchatbot(
    db: Session,
    *,
    gender: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    company: Optional[str] = None,
    street_and_number: Optional[str] = None,
    zip_and_city: Optional[str] = None,
    province: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    email: Optional[str] = None,
    phone1: Optional[str] = None,
    phone2: Optional[str] = None,
    phone3: Optional[str] = None,
    lead_type_id: Optional[int] = None,
    source: str = "chatbot",
    session_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> LeadChatbot:
    """
    Create a new LeadChatbot entry in the database.

    Returns:
        LeadChatbot: The created database object
    """
    lead_chatbot = LeadChatbot(
        gender=gender,
        first_name=first_name,
        last_name=last_name,
        company=company,
        street_and_number=street_and_number,
        zip_and_city=zip_and_city,
        province=province,
        latitude=latitude,
        longitude=longitude,
        email=email,
        phone1=phone1,
        phone2=phone2,
        phone3=phone3,
        lead_type_id=lead_type_id,
        source=source,
        session_id=session_id,
        notes=notes,
        registration_datetime=datetime.utcnow(),
    )

    db.add(lead_chatbot)
    db.commit()
    db.refresh(lead_chatbot)

    return lead_chatbot


def get_all_lead_types(db: Session):
    """Get all available lead types"""
    return db.query(LeadType).all()
