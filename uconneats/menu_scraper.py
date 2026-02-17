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
HOURS_URL = "https://dining.uconn.edu/hours/"
DATE_PARAM = "dtdate"
USER_AGENT = "UConnEatsBot/0.1 (academic project CLI prototype)"
MEAL_NAMES = {"breakfast", "lunch", "dinner", "brunch", "late night"}
DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


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


def parse_time_token(token: str, default_meridiem: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    cleaned = token.strip().lower().replace("*", "").replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", cleaned)
    if not m:
        return None, None
    hh = int(m.group(1))
    mm = int(m.group(2) or "00")
    meridiem = m.group(3) or default_meridiem
    if meridiem == "pm" and hh != 12:
        hh += 12
    if meridiem == "am" and hh == 12:
        hh = 0
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None, None
    return f"{hh:02d}:{mm:02d}", meridiem


def parse_time_range(raw: str) -> Optional[Tuple[str, str]]:
    text = normalize_space(raw).replace("*", "")
    if "-" not in text:
        return None
    left, right = [x.strip() for x in text.split("-", 1)]
    right_hhmm, right_meridiem = parse_time_token(right)
    if not right_hhmm:
        return None
    left_hhmm, _ = parse_time_token(left, default_meridiem=right_meridiem)
    if not left_hhmm:
        return None
    return left_hhmm, right_hhmm


def normalize_hall_label(label: str) -> List[str]:
    txt = label.replace("*", "").strip().lower()
    txt = txt.replace("connecticut hall", "connecticut")
    txt = txt.replace("north & whitney", "north, whitney")
    parts = re.split(r",|&", txt)
    halls = []
    for p in parts:
        p = normalize_space(p)
        if not p:
            continue
        if p in {"putnam", "connecticut", "mcmahon", "north", "northwest", "south", "gelfenbien", "whitney"}:
            halls.append(p)
    return halls


def is_hall_header(line: str) -> bool:
    if not line or ":" in line:
        return False
    if re.search(r"\d", line):
        return False
    return bool(normalize_hall_label(line))


def parse_hours_section(section_text: str, weekday_mode: bool) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    lines = [normalize_space(x) for x in section_text.splitlines() if normalize_space(x)]
    out: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    current_halls: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if is_hall_header(line):
            current_halls = normalize_hall_label(line)
            i += 1
            continue

        meal_match = re.match(r"^(Breakfast|Lunch|Dinner|Brunch|Late Night)\s*:\s*(.*)$", line, re.IGNORECASE)
        if not meal_match or not current_halls:
            i += 1
            continue

        meal = meal_match.group(1).title()
        rhs = meal_match.group(2).strip()
        if not rhs and (i + 1) < len(lines):
            rhs = lines[i + 1]
            i += 1

        day_targets = ["saturday", "sunday"] if not weekday_mode else DAY_NAMES[:5]
        rhs_l = rhs.lower()
        if "(saturday)" in rhs_l:
            day_targets = ["saturday"]
            rhs = rhs_l.replace("(saturday)", "").strip()
        elif "(sunday)" in rhs_l:
            day_targets = ["sunday"]
            rhs = rhs_l.replace("(sunday)", "").strip()

        parsed_range = parse_time_range(rhs)
        if parsed_range:
            start_hhmm, end_hhmm = parsed_range
            for hall in current_halls:
                for day_name in day_targets:
                    out.setdefault(hall, {}).setdefault(day_name, {})[meal] = {"start": start_hhmm, "end": end_hhmm}
        i += 1
    return out


def merge_hours(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(a)
    for hall, day_map in b.items():
        for day_name, meals in day_map.items():
            for meal, window in meals.items():
                merged.setdefault(hall, {}).setdefault(day_name, {})[meal] = window
    return merged


def scrape_official_hours(session: requests.Session) -> Dict[str, Any]:
    soup = fetch_soup(session, HOURS_URL)
    blobs = soup.select(".textwidget")
    page_text = ""
    for b in blobs:
        txt = b.get_text("\n", strip=True)
        if "MONDAY through FRIDAY HOURS" in txt and "WEEKEND HOURS" in txt:
            page_text = txt
            break
    if not page_text:
        page_text = soup.get_text("\n", strip=True)

    start_wk = page_text.find("MONDAY through FRIDAY HOURS")
    start_we = page_text.find("WEEKEND HOURS")
    end_we = page_text.find("KOSHER & HALAL")
    if start_wk < 0 or start_we < 0:
        return {}
    if end_we < 0:
        end_we = len(page_text)

    weekday_text = page_text[start_wk + len("MONDAY through FRIDAY HOURS") : start_we]
    weekend_text = page_text[start_we + len("WEEKEND HOURS") : end_we]

    weekday_hours = parse_hours_section(weekday_text, weekday_mode=True)
    weekend_hours = parse_hours_section(weekend_text, weekday_mode=False)
    return merge_hours(weekday_hours, weekend_hours)


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
    official_hours = scrape_official_hours(session)

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
        "source_hours_url": HOURS_URL,
        "official_hours": official_hours,
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
        default=str(Path(__file__).parent.parent / "data" / "menus_scraped.json"),
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
