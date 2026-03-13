#-------config.py-------
import os
from functools import lru_cache
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, AnyHttpUrl, field_validator

BACKEND_DIR = Path(__file__).resolve().parents[2]

RASTER_ROOT = Path(os.getenv("RASTER_ROOT", "/data/Mortality"))
MORTALITY_ROOT = RASTER_ROOT

# AOI bundled in repo
AOI_PATH = Path(os.getenv("AOI_PATH", "D:/AOI/AOI_diss.shp"))

# Static / generated outputs
STATIC_DIR = BACKEND_DIR / "static"
OVERLAYS_DIR = STATIC_DIR / "overlays"
EXPORTS_DIR = STATIC_DIR / "exports"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Storage
STORAGE_DIR = BACKEND_DIR / "storage"
GEOPDF_STORAGE_DIR = STORAGE_DIR / "geopdf"

class Settings(BaseModel):
    app_name: str = "VMRC Geospatial API"
    PROJECT_NAME: str = "VMRC Geospatial API"
    VERSION: str = "0.1.0"

    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    backend_cors_origins: List[AnyHttpUrl] = []

    database_url: str = "postgresql+psycopg://vmrc:vmrc123@localhost:5432/vmrc_db"

    aws_s3_bucket: Optional[str] = os.getenv("AWS_S3_BUCKET") or None
    aws_region: Optional[str] = os.getenv("AWS_REGION") or None
    gcs_bucket: Optional[str] = os.getenv("GCS_BUCKET") or None

    secret_key: str = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    access_token_expire_minutes: int = 60 * 24
    algorithm: str = "HS256"

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        if isinstance(v, list):
            return v
        return []

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()