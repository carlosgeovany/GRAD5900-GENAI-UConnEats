import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup


NUTRITION_INDEX_URL = "https://dining.uconn.edu/nutrition/"
DATE_PARAM = "dtdate"
USER_AGENT = "UConnEatsBot/0.1 (academic project CLI prototype)"
MEAL_NAMES = {"breakfast", "lunch", "dinner", "brunch", "late night"}


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def parse_allergens(desc: str) -> List[str]:
    match = re.search(r"Allergens:\s*(.+)$", desc, re.IGNORECASE)
    if not match:
        return []
    allergens_raw = match.group(1)
    return [normalize_space(x) for x in allergens_raw.split(",") if x.strip()]


def derive_flags(desc: str, item_name: str) -> Dict[str, bool]:
    blob = f"{desc} {item_name}".lower()
    return {
        "chefs_choice": "chef's choice" in blob or "chefs choice" in blob,
        "contains_pork": "contains pork" in blob or "pork" in blob,
        "contains_alcohol": "contains alcohol" in blob or "alcohol" in blob,
        "variable_nutrition": "nutrition and allergens will vary" in blob,
    }


def parse_menu_date(soup: BeautifulSoup) -> Optional[str]:
    title = soup.select_one(".shortmenutitle")
    if not title:
        return None
    text = normalize_space(title.get_text(" ", strip=True))
    match = re.search(r"Menus for (.+)$", text)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%A, %B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def build_date_url(base_url: str, date_obj: datetime) -> str:
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query[DATE_PARAM] = [f"{date_obj.month}/{date_obj.day}/{date_obj.year}"]
    query["myaction"] = ["read"]
    query["WeeksMenus"] = ["This Week's Menus"]
    encoded = urlencode(query, doseq=True)
    return parsed._replace(query=encoded).geturl()


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def discover_halls(session: requests.Session, index_url: str) -> List[Dict[str, str]]:
    soup = fetch_soup(session, index_url)
    container = soup.select_one("#pg-60-2")
    if not container:
        raise RuntimeError("Could not find nutrition hall container '#pg-60-2'.")
    halls: List[Dict[str, str]] = []
    for a in container.select("a[href]"):
        name = normalize_space(a.get_text(" ", strip=True))
        href = a.get("href", "").strip()
        if not name or not href:
            continue
        halls.append(
            {
                "hall_id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "hall_name": name,
                "url": urljoin(index_url, href),
            }
        )
    return halls


def parse_one_hall_day(
    session: requests.Session, hall: Dict[str, str], day_url: str
) -> Optional[Dict[str, Any]]:
    soup = fetch_soup(session, day_url)
    menu_date = parse_menu_date(soup)
    if not menu_date:
        return None

    meals: Dict[str, List[Dict[str, Any]]] = {}
    current_meal = None
    current_station = None
    last_item: Optional[Dict[str, Any]] = None

    nodes = soup.select(
        ".shortmenumeals, .shortmenucats, .shortmenurecipes, .shortmenuproddesc"
    )
    for node in nodes:
        classes = node.get("class", [])
        text = normalize_space(node.get_text(" ", strip=True))
        if not text:
            continue

        if "shortmenumeals" in classes:
            if text.lower() in MEAL_NAMES:
                current_meal = text
                meals.setdefault(current_meal, [])
                current_station = None
                last_item = None
            continue

        if "shortmenucats" in classes:
            station = text.strip("- ").strip()
            current_station = station
            continue

        if "shortmenurecipes" in classes:
            if not current_meal:
                continue
            item = {
                "item_name": text,
                "station": current_station or "General",
                "allergens": [],
                "diet_tags": [],
                "flags": {
                    "chefs_choice": False,
                    "contains_pork": False,
                    "contains_alcohol": False,
                    "variable_nutrition": False,
                },
            }
            meals[current_meal].append(item)
            last_item = item
            continue

        if "shortmenuproddesc" in classes and last_item is not None:
            allergens = parse_allergens(text)
            flags = derive_flags(text, last_item["item_name"])
            last_item["allergens"] = allergens
            last_item["flags"] = flags
            if "vegan" in text.lower():
                last_item["diet_tags"].append("vegan")
            if "vegetarian" in text.lower():
                last_item["diet_tags"].append("vegetarian")

    if not meals:
        return None

    return {
        "hall_id": hall["hall_id"],
        "hall_name": hall["hall_name"],
        "source_url": day_url,
        "menu_date": menu_date,
        "meals": meals,
    }


def scrape_menus(days_ahead: int, include_halls: Optional[List[str]]) -> Dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    halls = discover_halls(session, NUTRITION_INDEX_URL)

    if include_halls:
        wanted = {x.lower() for x in include_halls}
        halls = [h for h in halls if h["hall_id"].lower() in wanted or h["hall_name"].lower() in wanted]

    start_date = datetime.now()
    menus: List[Dict[str, Any]] = []

    for hall in halls:
        base_url = hall["url"]
        for offset in range(days_ahead + 1):
            target = start_date + timedelta(days=offset)
            day_url = build_date_url(base_url, target)
            parsed = parse_one_hall_day(session, hall, day_url)
            if parsed:
                menus.append(parsed)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_index_url": NUTRITION_INDEX_URL,
        "halls": [
            {"hall_id": h["hall_id"], "hall_name": h["hall_name"], "source_url": h["url"]}
            for h in halls
        ],
        "menus": menus,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UConn Dining nutrition menus into normalized JSON.")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=7,
        help="How many days ahead to scrape (including today).",
    )
    parser.add_argument(
        "--halls",
        default="",
        help="Optional comma-separated hall_id or hall_name filter.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data" / "menus_scraped.json"),
        help="Output path for normalized JSON.",
    )
    args = parser.parse_args()

    include_halls = [normalize_space(x) for x in args.halls.split(",") if x.strip()]
    payload = scrape_menus(days_ahead=args.days_ahead, include_halls=include_halls or None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['menus'])} hall-day menu records to {out_path}")


if __name__ == "__main__":
    main()
