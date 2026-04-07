import os
import uvicorn

from .main import app


def main():
    """
    OpenEnv entry point.

    This module is required for `openenv validate` multi-mode deployment checks.
    """
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()

