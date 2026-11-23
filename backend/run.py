"""Start the app and open it in the default browser.

  uv run python run.py

Listens on 127.0.0.1 only: nothing is reachable from other machines.
"""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

from app.config import HOST, PORT

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="warning",
                server_header=False)
