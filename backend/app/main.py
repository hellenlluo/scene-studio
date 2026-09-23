from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import jobs, scenes
from app.config import get_settings
from app.db import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(title="SceneStudio", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(scenes.router)

# Only generated scene assets are public. Mounting `storage_dir` itself also serves
# the SQLite database, uploaded source photos and cached model responses; none of
# those are browser assets. Keeping the `/storage/scenes/...` URL prefix preserves
# the paths already stored in scene specs while narrowing the filesystem boundary.
app.mount(
    "/storage/scenes",
    StaticFiles(directory=settings.scenes_dir),
    name="scene-assets",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
