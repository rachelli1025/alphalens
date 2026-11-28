from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alphalens_core.config import get_settings
from app.routers import ask, health

app = FastAPI(title="AlphaLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,  # CORS_ORIGINS env var
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Access-Key"],
)

app.include_router(health.router)
app.include_router(ask.router)
