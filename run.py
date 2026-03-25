"""Entry point — run from project root: python run.py"""
import uvicorn
from backend import config

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
