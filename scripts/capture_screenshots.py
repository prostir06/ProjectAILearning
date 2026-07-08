"""
Знімає скріншоти Streamlit та Flask додатків для README.

Використання:
    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --flask-only
    python scripts/capture_screenshots.py --streamlit-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"
STREAMLIT_PORT = 8502
FLASK_PORT = 5001


def wait_for_server(base_url: str, timeout: float = 60.0) -> None:
    """Чекає, поки HTTP-сервер відповість."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"Сервер не запустився на {base_url}")


def start_streamlit() -> subprocess.Popen:
    """Запускає Streamlit у фоновому режимі."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "streamlit_app.py"),
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_flask() -> subprocess.Popen:
    """Запускає Flask у фоновому режимі."""
    env = os.environ.copy()
    env["FLASK_APP"] = "app"
    env["FLASK_DEBUG"] = "0"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "flask",
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            str(FLASK_PORT),
            "--no-debugger",
            "--no-reload",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_process(process: subprocess.Popen) -> None:
    """Коректно зупиняє фоновий процес."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def capture_streamlit(page) -> None:
    """Скріншоти Streamlit-додатка."""
    base_url = f"http://localhost:{STREAMLIT_PORT}"
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_timeout(4000)

    page.screenshot(
        path=str(SCREENSHOTS_DIR / "streamlit-01-metrics-and-form.png"),
        full_page=True,
    )

    page.get_by_role("button", name="Передбачити").click()
    page.wait_for_timeout(5000)

    page.screenshot(
        path=str(SCREENSHOTS_DIR / "streamlit-02-prediction-results.png"),
        full_page=True,
    )


def capture_flask(page) -> None:
    """Скріншоти Flask-додатка."""
    base_url = f"http://127.0.0.1:{FLASK_PORT}"
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_timeout(3000)

    page.screenshot(
        path=str(SCREENSHOTS_DIR / "flask-01-metrics-and-form.png"),
        full_page=True,
    )

    page.get_by_role("button", name="Передбачити").click()
    page.wait_for_timeout(5000)

    page.screenshot(
        path=str(SCREENSHOTS_DIR / "flask-02-prediction-results.png"),
        full_page=True,
    )


def capture_apps(
    streamlit: bool = True,
    flask: bool = True,
) -> None:
    """Робить скріншоти обраних додатків."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Потрібен playwright: pip install playwright && playwright install chromium"
        ) from exc

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []

    try:
        if streamlit:
            processes.append(start_streamlit())
            wait_for_server(f"http://localhost:{STREAMLIT_PORT}")
        if flask:
            processes.append(start_flask())
            wait_for_server(f"http://127.0.0.1:{FLASK_PORT}")

        time.sleep(2)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            if streamlit:
                capture_streamlit(page)
            if flask:
                capture_flask(page)

            browser.close()
    finally:
        for process in processes:
            stop_process(process)

    print(f"Скріншоти збережено в {SCREENSHOTS_DIR}")


def parse_args() -> argparse.Namespace:
    """Парсить аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Знімає скріншоти UI для README.")
    parser.add_argument(
        "--streamlit-only",
        action="store_true",
        help="Лише Streamlit",
    )
    parser.add_argument(
        "--flask-only",
        action="store_true",
        help="Лише Flask",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входу скрипта."""
    args = parse_args()
    if args.streamlit_only and args.flask_only:
        raise SystemExit("Оберіть лише один з прапорців --streamlit-only / --flask-only.")

    capture_apps(
        streamlit=not args.flask_only,
        flask=not args.streamlit_only,
    )


if __name__ == "__main__":
    main()
