import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import predict, optimize
from api.services.predictor import predictor
from api.services.optimizer import optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    predictor.load()
    optimizer.load()
    logger.info(f"Models loaded: predictor={predictor._loaded}, optimizer={optimizer._loaded}")
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(optimize.router)


@app.get("/health")
async def health():
    ok = predictor._loaded
    logger.debug(f"Health check: models_loaded={ok}")
    return {"status": "ok", "models_loaded": ok}
