import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_model.gpu_setup import configure_gpu
from api.config import settings
from api.routers import predict
from api.services.predictor import predictor
from api.services.timefm_predictor import timefm_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_gpu()
    logger.info("Loading ANN models...")
    predictor.load()
    logger.info("ANN loaded: M1 ensemble (20 seeds), physics-based speed optimizer")
    try:
        logger.info("Loading TimeFM 2.5 inference...")
        timefm_api.load()
        if timefm_api.loaded:
            logger.info("TimeFM 2.5 loaded")
        else:
            logger.warning("TimeFM 2.5 head not found — run training to enable")
    except Exception as e:
        logger.warning(f"TimeFM 2.5 not available: {e}")
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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "ann_m1_ensemble": predictor._loaded,
            "speed_optimizer": "physics_based",
            "timefm_2_5": timefm_api.loaded,
        },
    }
