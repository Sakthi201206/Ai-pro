import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Load variables from .env file

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = BASE_DIR / "ai_jobs_debate_dataset.csv"

# OpenAI key (optional – if not set, uses smart template generation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# CORS allowed origins
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
