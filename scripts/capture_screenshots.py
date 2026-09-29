"""Capture a screenshot of every dashboard page (requires: pip install playwright && playwright install chromium)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PAGES = ["overview", "vendor", "factors", "segments", "matching", "copilot", "evaluation", "method"]


def main(port: int = 8599) -> None:
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.headless", "true"], cwd=ROOT)
    try:
        time.sleep(8)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            for key in PAGES:
                page.goto(f"http://localhost:{port}/?page={key}")
                page.wait_for_timeout(90000 if key == "evaluation" else 15000)
                page.screenshot(path=str(OUT / f"{key}.png"), full_page=True)
                print(f"saved {key}.png")
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
