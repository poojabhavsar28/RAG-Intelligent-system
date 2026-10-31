"""Backward-compatible ASGI entry point.

The canonical FastAPI application and backend selection live in app.main.
Keeping a second legacy application here would bypass centralized settings,
PostgreSQL tenant authentication, RLS repositories, and restricted CORS
whenever a deployment used main:app.
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, loop="asyncio", log_config=None)
