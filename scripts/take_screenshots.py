"""Retake the README screenshots from a running app.

    streamlit run app/streamlit_app.py --server.port 8599
    pip install playwright
    python scripts/take_screenshots.py

Uses the installed Google Chrome. It selects the README's worked example, then saves each
page or section whole, not just one screen of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, TimeoutError, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
DESKTOP = {"width": 1280, "height": 900}
PHONE = {"width": 390, "height": 844}


def settle(page: Page) -> None:
    """Wait until Streamlit has finished running the script and drawing charts."""
    page.wait_for_load_state("networkidle")
    page.wait_for_selector('[data-testid="stMain"]')
    page.wait_for_function('!document.querySelector(\'[data-testid="stStatusWidget"]\')', timeout=60_000)
    page.wait_for_timeout(1500)


def choose(page: Page, label: str, option: str) -> None:
    box = page.get_by_role("combobox", name=label, exact=True)
    choice = page.locator('[role="option"]', has_text=option).first
    box.scroll_into_view_if_needed()
    page.wait_for_timeout(500)  # a dropdown opened while the page still moves closes again
    box.click()
    try:
        choice.wait_for(state="visible", timeout=5000)
    except TimeoutError:
        box.click()
    choice.click()
    settle(page)


def grow(page: Page, width: int) -> None:
    """Streamlit scrolls an inner element, so the window is grown to show all of it."""
    height = page.evaluate('document.querySelector(\'[data-testid="stMain"]\').scrollHeight')
    page.set_viewport_size({"width": width, "height": max(height, 400)})
    page.wait_for_timeout(1500)


def full_page(page: Page, name: str, width: int) -> None:
    grow(page, width)
    page.screenshot(path=OUT / name)
    page.set_viewport_size({"width": width, "height": 900})
    print(name)


def section(page: Page, name: str, width: int, start: str, end: str) -> None:
    """The part of the page from the heading `start` down to the heading `end`."""
    grow(page, width)
    tops = page.evaluate(
        """([start, end]) => {
            const top = text => [...document.querySelectorAll('h2, h3')]
                .find(h => h.innerText.includes(text)).getBoundingClientRect().top;
            return [top(start), top(end)];
        }""",
        [start, end],
    )
    page.screenshot(path=OUT / name, clip={"x": 0, "y": tops[0] - 16, "width": width, "height": tops[1] - tops[0]})
    page.set_viewport_size({"width": width, "height": 900})
    print(name)


def worked_example(page: Page) -> None:
    choose(page, "Date", "2026-11-02")
    choose(page, "Snapshot", "D-1 14:00")
    choose(page, "Contract", "65–66°F")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=DESKTOP, color_scheme="dark", device_scale_factor=1)
        page = context.new_page()

        page.goto(URL)
        settle(page)
        worked_example(page)
        full_page(page, "01-contract-explorer.png", DESKTOP["width"])

        page.get_by_role("button", name="Price this contract").click()
        settle(page)
        full_page(page, "02-note-pricer.png", DESKTOP["width"])

        page.goto(f"{URL}/model_evidence")
        settle(page)
        section(page, "03-model-evidence-scores.png", DESKTOP["width"], "survive better tests", "Results for one city")
        grow(page, DESKTOP["width"])  # the whole page fits, so the selector needs no scrolling
        choose(page, "City", "Chicago (Midway)")
        section(page, "04-model-evidence-calibration.png", DESKTOP["width"], "Results for one city", "Limitations")

        fresh = browser.new_context(viewport=DESKTOP, color_scheme="dark", device_scale_factor=1).new_page()
        fresh.goto(f"{URL}/note_pricer")
        settle(fresh)
        full_page(fresh, "05-note-pricer-no-selection.png", DESKTOP["width"])

        phone = browser.new_context(viewport=PHONE, color_scheme="dark", device_scale_factor=2, is_mobile=True).new_page()
        phone.goto(URL)
        settle(phone)
        worked_example(phone)
        full_page(phone, "06-phone-width.png", PHONE["width"])
        browser.close()


if __name__ == "__main__":
    main()
