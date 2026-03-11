# app/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings, STATIC_DIR, BACKEND_DIR
from app.api.v1.api import api_router


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
    )

    # ---------- CORS ----------
    # For Cloudflare tunnel support, set ALLOWED_ORIGINS env var (comma-separated)
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
    if allowed_origins_env:
        origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    else:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,      # Explicit origins (not "*") - required for allow_credentials=True
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- STATIC FILES ----------
    # Serves PNG overlays located in /static/overlays/*
    # Make sure "static" folder exists relative to backend working directory
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    # ---------- ROUTERS ----------
    app.include_router(api_router, prefix="/api/v1")

    # =====================================================
    # Serve frontend (Vite build)
    # Repo layout:
    #   repo/
    #     backend/
    #       app/main.py
    #     frontend/
    #       dist/
    # =====================================================
    FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

    if FRONTEND_DIST.is_dir():
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_frontend_root():
            return FileResponse(str(FRONTEND_DIST / "index.html"))

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_frontend_spa(full_path: str):
            candidate = FRONTEND_DIST / full_path
            if candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(FRONTEND_DIST / "index.html"))

    return app


app = create_application()
