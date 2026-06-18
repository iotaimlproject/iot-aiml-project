from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import predict, optimize
from api.services.predictor import predictor
from api.services.optimizer import optimizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    optimizer.load()
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
    return {"status": "ok", "models_loaded": predictor._loaded}
