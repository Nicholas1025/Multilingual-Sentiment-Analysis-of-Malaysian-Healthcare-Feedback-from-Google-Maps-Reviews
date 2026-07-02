"""
Google Maps Hospital Reviews Scraper - Melaka, Malaysia
Powered by SeleniumBase UC Mode (undetected-chromedriver).

Google added a "limited view" restriction in Feb 2026 that hides the Reviews
tab from obvious automation. SeleniumBase UC Mode + search-based navigation
bypasses this. NOTE: UC Mode must run with a visible browser window
(headless=False) - it does NOT work headless.

Label logic: 1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive
"""

import time
import csv
import os
import random
import sys
import urllib.parse
from datetime import datetime

from seleniumbase import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, StaleElementReferenceException
)

# Allow Unicode output on Windows consoles using a non-UTF-8 codepage
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "Data", "raw", "healthcare_raw.csv"
)

# Melaka healthcare providers - Google Maps search queries.
# type "hospital" -> scrape up to HOSPITAL_MAX reviews.
# type "clinic"   -> scrape up to CLINIC_MAX; discarded if it has fewer
#                    than CLINIC_MIN_KEEP reviews (keep only well-reviewed ones).
HOSPITALS = [
    # --- hospitals ---
    {"query": "Hospital Melaka",                            "type": "hospital"},
    {"query": "Mahkota Medical Centre Melaka",              "type": "hospital"},
    {"query": "Pantai Hospital Ayer Keroh Melaka",          "type": "hospital"},
    {"query": "Columbia Asia Hospital Bukit Rambai Melaka", "type": "hospital"},
    {"query": "Oriental Melaka Straits Medical Centre",     "type": "hospital"},
    {"query": "Putra Specialist Hospital Melaka",           "type": "hospital"},
    {"query": "Hospital Jasin Melaka",                      "type": "hospital"},
    {"query": "Hospital Alor Gajah Melaka",                 "type": "hospital"},
    # --- clinics / dental / traditional medicine (kept only if >= 90 reviews) ---
    {"query": "QHC Medical Centre Melaka",                  "type": "clinic"},
    {"query": "Klinik Kesihatan Peringgit Melaka",          "type": "clinic"},
    {"query": "Klinik Kesihatan Bukit Baru Melaka",         "type": "clinic"},
    {"query": "Klinik Kesihatan Ayer Keroh",                "type": "clinic"},
    {"query": "Klinik Kesihatan Klebang Melaka",            "type": "clinic"},
    {"query": "Klinik Kesihatan Batu Berendam",             "type": "clinic"},
    {"query": "Klinik Pergigian Melaka",                    "type": "clinic"},
    {"query": "Klinik Pergigian Peringgit Melaka",          "type": "clinic"},
]

HOSPITAL_MAX    = 150   # max reviews per hospital
CLINIC_MAX      = 90    # max reviews per clinic
CLINIC_MIN_KEEP = 90    # clinics with fewer reviews than this are dropped

# ─── LABEL ───────────────────────────────────────────────────────────────────

def stars_to_label(stars: int) -> str:
    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"

# ─── DRIVER SETUP ─────────────────────────────────────────────────────────────

def build_driver() -> Driver:
    """SeleniumBase UC Mode driver. headless=False is REQUIRED for stealth."""
    driver = Driver(
        uc=True,
        headless=False,
        locale_code="en",
        window_size="1920,1080",
    )
    driver.implicitly_wait(4)
    return driver


# Debug: raw HTML of a few review cards, saved so the 'See original'
# (translation) handling can be verified after a run.
DEBUG_HTML = []
DEBUG_FILE = os.path.join(os.path.dirname(__file__), "..", "..",
                          "Data", "raw", "translation_debug.html")

# ─── PAGE NAVIGATION ──────────────────────────────────────────────────────────

def open_place(driver, query: str):
    """Navigate to the Maps search URL using UC Mode's stealth open.
    hl=en forces an English interface so the 'See original' toggle is in
    English regardless of the logged-in account's language setting."""
    url = ("https://www.google.com/maps/search/"
           + urllib.parse.quote(query) + "?hl=en")
    driver.uc_open_with_reconnect(url, reconnect_time=6)
    time.sleep(5)

    # If a results list appeared instead of a place page, click first result
    try:
        results = driver.find_elements(By.CSS_SELECTOR, "a.hfpxzc")
        if results:
            driver.execute_script("arguments[0].click();", results[0])
            time.sleep(5)
    except Exception:
        pass


def open_reviews_tab(driver) -> bool:
    """Click the Reviews tab (matched by aria-label or text). Retries once."""
    for _ in range(2):
        try:
            tabs = WebDriverWait(driver, 12).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, "button[role='tab']")
            )
            for tab in tabs:
                label = (tab.get_attribute("aria-label")
                         or tab.text or "").lower()
                if "review" in label:
                    driver.execute_script("arguments[0].click();", tab)
                    time.sleep(4)
                    return True
        except TimeoutException:
            pass
        time.sleep(4)
    return False


def wait_for_reviews(driver, timeout=12) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[data-review-id]")
            )
        )
        return True
    except TimeoutException:
        return False

# ─── REVIEW EXTRACTION ────────────────────────────────────────────────────────

def get_star_count(review_el) -> int:
    try:
        aria = review_el.find_element(
            By.CSS_SELECTOR, "span[role='img'][aria-label]"
        ).get_attribute("aria-label")
        for word in aria.replace("-", " ").split():
            if word.isdigit():
                return int(word)
    except Exception:
        pass
    return 0


def _click_buttons(driver, review_el, matchers):
    """Click every button in the card whose text/aria-label matches."""
    try:
        for btn in review_el.find_elements(By.TAG_NAME, "button"):
            try:
                label = ((btn.text or "") + " "
                         + (btn.get_attribute("aria-label") or "")).lower()
                if any(m in label for m in matchers):
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.3)
            except StaleElementReferenceException:
                continue
    except Exception:
        pass


def expand_and_get_text(driver, review_el) -> str:
    # 1. expand truncated text  ("More" / "See more")
    _click_buttons(driver, review_el, ["see more", "more"])

    # 2. if the review was auto-translated, switch to the original language
    #    (English UI forced via hl=en, so the toggle reads "See original")
    _click_buttons(driver, review_el, ["see original", "original"])

    # 3. read the (now original) review text
    try:
        return review_el.find_element(
            By.CSS_SELECTOR, "span.wiI7pd"
        ).text.strip()
    except NoSuchElementException:
        return ""


def get_review_date(review_el) -> str:
    try:
        return review_el.find_element(
            By.CSS_SELECTOR, "span.rsqaWe"
        ).text.strip()
    except NoSuchElementException:
        return ""

# ─── CORE SCRAPER ─────────────────────────────────────────────────────────────

def scrape_hospital(driver, place: dict) -> list[dict]:
    hospital = place["query"]
    ptype    = place["type"]
    cap = HOSPITAL_MAX if ptype == "hospital" else CLINIC_MAX
    print(f"\n[->] Scraping ({ptype}, max {cap}): {hospital}", flush=True)
    reviews = []

    # Retry up to 3x: a fresh reload often escapes Google's "limited view".
    opened = False
    for attempt in range(3):
        open_place(driver, hospital)
        if open_reviews_tab(driver):
            opened = True
            break
        print(f"  [..] Reviews tab missing - retry {attempt + 1}/3",
              flush=True)
        time.sleep(random.uniform(3, 6))

    if not opened:
        print("  [!] Reviews tab not found - skipping", flush=True)
        return reviews

    if not wait_for_reviews(driver):
        print("  [!] No reviews loaded - skipping", flush=True)
        return reviews

    seen = set()
    stale_rounds = 0
    prev_card_count = 0

    for _ in range(110):
        cards = driver.find_elements(By.CSS_SELECTOR, "div[data-review-id]")

        if cards:
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});",
                    cards[-1],
                )
            except StaleElementReferenceException:
                pass
        time.sleep(random.uniform(1.0, 1.6))

        for card in cards:
            if len(reviews) >= cap:
                break
            try:
                stars = get_star_count(card)
                if stars == 0:
                    continue
                text = expand_and_get_text(driver, card)
                if not text or len(text) < 10 or text in seen:
                    continue
                seen.add(text)
                reviews.append({
                    "source": "Google Maps",
                    "hospital": hospital,
                    "text": text,
                    "stars": stars,
                    "label": stars_to_label(stars),
                    "date_scraped": datetime.now().strftime("%Y-%m-%d"),
                    "review_date": get_review_date(card),
                })
            except StaleElementReferenceException:
                continue

        if len(reviews) >= cap:
            break

        if len(cards) == prev_card_count:
            stale_rounds += 1
            if stale_rounds >= 4:
                break
        else:
            stale_rounds = 0
        prev_card_count = len(cards)

    # save a few raw review cards for translation-handling diagnostics
    if len(DEBUG_HTML) < 36:
        try:
            for card in driver.find_elements(
                    By.CSS_SELECTOR, "div[data-review-id]")[:6]:
                DEBUG_HTML.append(
                    f"<!-- ===== {hospital} ===== -->\n"
                    + (card.get_attribute("outerHTML") or "")
                )
        except Exception:
            pass

    # clinics must be well-reviewed; otherwise drop them
    if ptype == "clinic" and len(reviews) < CLINIC_MIN_KEEP:
        print(f"  [skip] Clinic has only {len(reviews)} reviews "
              f"(< {CLINIC_MIN_KEEP}) - discarded", flush=True)
        return []

    print(f"  [OK] Collected {len(reviews)} reviews", flush=True)
    return reviews

# ─── SAVE ────────────────────────────────────────────────────────────────────

def save_csv(reviews: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["source", "hospital", "text", "stars", "label",
                  "date_scraped", "review_date"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n[OK] Saved {len(reviews)} reviews -> {path}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    driver = build_driver()
    all_reviews = []

    try:
        for place in HOSPITALS:
            try:
                all_reviews.extend(scrape_hospital(driver, place))
            except Exception as e:
                print(f"  [!] Error on {place['query']}: {e}", flush=True)

            from collections import Counter
            dist = Counter(r["label"] for r in all_reviews)
            print(f"  Running totals -> {dict(dist)}", flush=True)
            time.sleep(random.uniform(2, 4))
    finally:
        driver.quit()

    # write the diagnostic HTML (used to verify translation handling)
    if DEBUG_HTML:
        try:
            os.makedirs(os.path.dirname(DEBUG_FILE), exist_ok=True)
            with open(DEBUG_FILE, "w", encoding="utf-8") as f:
                f.write("\n\n".join(DEBUG_HTML))
        except OSError:
            pass

    print(f"\nTotal collected: {len(all_reviews)}")
    save_csv(all_reviews, OUTPUT_FILE)
    print("Next step: run merge_and_check.py to merge with Play Store data.")


if __name__ == "__main__":
    main()
