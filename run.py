"""Entry point — run from project root: python run.py"""
import os
import uvicorn
from backend import config

if __name__ == "__main__":
    # reload=True for local dev, False for production (Render, Docker)
    is_dev = os.getenv("ENV", "production") == "development"
    uvicorn.run(
        "backend.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=is_dev,
    )
