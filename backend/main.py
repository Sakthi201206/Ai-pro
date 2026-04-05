from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.config import ALLOWED_ORIGINS

app = FastAPI(
    title="AI Debate Assistant API",
    description="RAG-powered debate generator using the AI Jobs dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "AI Debate Assistant API is running 🚀", "docs": "/docs"}
