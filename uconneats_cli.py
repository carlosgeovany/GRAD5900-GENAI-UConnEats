import argparse
import json
import math
import os
import re
import requests
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI
from menu_scraper import scrape_menus, scrape_official_hours


@dataclass
class ParsedIntent:
    craving_terms: List[str]
    cuisine_hints: List[str]
    avoid_allergens: List[str]
    preferred_diets: List[str]
    allergen_terms: List[str]
    requested_date: Optional[str]
    requested_time: Optional[str]
    requested_meal: Optional[str]
    requested_hall: Optional[str]
    menu_lookup: bool
    hours_lookup: bool
    allergen_lookup: bool


DIET_ALIASES = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "halal": "halal",
    "kosher": "kosher",
    "glutenfree": "gluten-free",
    "gluten-free": "gluten-free",
    "pescatarian": "pescatarian",
}

KNOWN_ALLERGENS = {
    "peanut": "peanuts",
    "peanuts": "peanuts",
    "tree nut": "tree nuts",
    "tree nuts": "tree nuts",
    "nuts": "tree nuts",
    "milk": "milk",
    "dairy": "milk",
    "egg": "eggs",
    "eggs": "eggs",
    "soy": "soy",
    "soybean": "soy",
    "soybeans": "soy",
    "wheat": "wheat",
    "gluten": "gluten",
    "sesame": "sesame",
    "shellfish": "shellfish",
    "fish": "fish",
}

GENERIC_QUERY_TOKENS = {
    "what",
    "whats",
    "whats",
    "is",
    "are",
    "for",
    "at",
    "on",
    "in",
    "the",
    "there",
    "option",
    "options",
    "food",
    "foods",
    "menu",
    "tonight",
    "today",
    "tomorrow",
    "breakfast",
    "lunch",
    "dinner",
}


def load_data(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_data(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_token(token: str) -> str:
    return token.strip().lower()


def normalize_word(token: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", normalize_token(token))


def parse_csv_arg(raw: str) -> List[str]:
    if not raw:
        return []
    return [normalize_token(x) for x in raw.split(",") if x.strip()]


def parse_time_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def in_window(current: time, start_str: str, end_str: str) -> bool:
    start_t = parse_time_hhmm(start_str)
    end_t = parse_time_hhmm(end_str)
    return start_t <= current <= end_t


def get_open_meals(menu_entry: Dict[str, Any], now_t: time) -> List[str]:
    hours = menu_entry.get("hours", {})
    open_meals = []
    for meal_name, window in hours.items():
        if in_window(now_t, window["start"], window["end"]):
            open_meals.append(meal_name)
    return open_meals


def openai_parse_intent(
    client: OpenAI,
    model: str,
    query: str,
    explicit_allergens: List[str],
    explicit_diets: List[str],
    now_et: datetime,
) -> ParsedIntent:
    prompt = f"""
Extract user dining intent into strict JSON.

User query: {query}
Explicit allergens to avoid: {explicit_allergens}
Explicit diet preferences: {explicit_diets}
Current Eastern time reference: {now_et.strftime('%Y-%m-%d %H:%M')}

Return JSON only with this schema:
{{
  "craving_terms": ["..."],
  "cuisine_hints": ["..."],
  "avoid_allergens": ["..."],
  "allergen_terms": ["..."],
  "preferred_diets": ["..."],
  "requested_date": "YYYY-MM-DD or empty",
  "requested_time": "HH:MM 24h or empty",
  "requested_meal": "Breakfast/Lunch/Dinner/Late Night or empty",
  "requested_hall": "South/Northwest/etc or empty",
  "menu_lookup": true,
  "hours_lookup": false,
  "allergen_lookup": false
}}
"""
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    raw = response.output_text.strip()
    parsed = safe_json_parse(raw)
    return ParsedIntent(
        craving_terms=[normalize_token(x) for x in parsed.get("craving_terms", [])],
        cuisine_hints=[normalize_token(x) for x in parsed.get("cuisine_hints", [])],
        avoid_allergens=[normalize_token(x) for x in parsed.get("avoid_allergens", [])],
        allergen_terms=[normalize_token(x) for x in parsed.get("allergen_terms", [])],
        preferred_diets=[normalize_token(x) for x in parsed.get("preferred_diets", [])],
        requested_date=(parsed.get("requested_date") or "").strip() or None,
        requested_time=(parsed.get("requested_time") or "").strip() or None,
        requested_meal=(parsed.get("requested_meal") or "").strip() or None,
        requested_hall=(parsed.get("requested_hall") or "").strip() or None,
        menu_lookup=bool(parsed.get("menu_lookup", False)),
        hours_lookup=bool(parsed.get("hours_lookup", False)),
        allergen_lookup=bool(parsed.get("allergen_lookup", False)),
    )


def local_parse_intent(
    query: str,
    explicit_allergens: List[str],
    explicit_diets: List[str],
    now_et: datetime,
) -> ParsedIntent:
    query_l = query.lower()
    tokens = [
        normalize_token(t)
        for t in query.replace("/", " ").split()
        if t.strip()
    ]
    requested_date, requested_time, requested_meal = parse_datetime_from_query_local(query_l, now_et)
    menu_lookup = any(
        x in query_l
        for x in ["what's for", "whats for", "what is for", "menu at", "what is on the menu", "what's on the menu"]
    )
    hours_lookup = any(x in query_l for x in ["hours", "open", "close", "closing", "opening", "when does"])
    allergen_lookup = any(x in query_l for x in ["allergen", "allergens", "contains", "contain", "free"])
    allergen_terms = extract_allergen_terms(query_l)
    return ParsedIntent(
        craving_terms=tokens,
        cuisine_hints=[],
        avoid_allergens=explicit_allergens,
        allergen_terms=allergen_terms,
        preferred_diets=explicit_diets,
        requested_date=requested_date,
        requested_time=requested_time,
        requested_meal=requested_meal,
        requested_hall=None,
        menu_lookup=menu_lookup,
        hours_lookup=hours_lookup,
        allergen_lookup=allergen_lookup or bool(allergen_terms),
    )


def parse_datetime_from_query_local(query_l: str, now_et: datetime) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    requested_date: Optional[str] = None
    requested_time: Optional[str] = None
    requested_meal: Optional[str] = None

    if "tomorrow" in query_l:
        requested_date = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in query_l or "tonight" in query_l:
        requested_date = now_et.strftime("%Y-%m-%d")

    date_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query_l)
    if date_iso:
        requested_date = date_iso.group(1)
    else:
        md = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", query_l)
        if md:
            month = int(md.group(1))
            day = int(md.group(2))
            year_raw = md.group(3)
            if year_raw:
                year = int(year_raw)
                if year < 100:
                    year += 2000
            else:
                year = now_et.year
            try:
                requested_date = datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                requested_date = None

    if "breakfast" in query_l:
        requested_meal = "Breakfast"
    elif "lunch" in query_l:
        requested_meal = "Lunch"
    elif "dinner" in query_l or "tonight" in query_l:
        requested_meal = "Dinner"
    elif "late night" in query_l:
        requested_meal = "Late Night"

    if "noon" in query_l:
        requested_time = "12:00"
    elif "midnight" in query_l:
        requested_time = "00:00"
    else:
        ampm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", query_l)
        if ampm:
            hh = int(ampm.group(1))
            mm = int(ampm.group(2) or "00")
            period = ampm.group(3)
            if period == "pm" and hh != 12:
                hh += 12
            if period == "am" and hh == 12:
                hh = 0
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                requested_time = f"{hh:02d}:{mm:02d}"
        else:
            hhmm = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", query_l)
            if hhmm:
                requested_time = f"{int(hhmm.group(1)):02d}:{hhmm.group(2)}"

    return requested_date, requested_time, requested_meal


def extract_allergen_terms(query_l: str) -> List[str]:
    found = set()
    for k, v in KNOWN_ALLERGENS.items():
        if k in query_l:
            found.add(v)
    return sorted(found)


def normalize_intent(intent: ParsedIntent) -> ParsedIntent:
    diets = {normalize_token(x) for x in intent.preferred_diets}
    allg = {normalize_token(x) for x in intent.allergen_terms}
    cleaned_terms: List[str] = []
    for t in intent.craving_terms:
        w = normalize_word(t)
        if not w or w in GENERIC_QUERY_TOKENS:
            continue
        if w in DIET_ALIASES:
            diets.add(DIET_ALIASES[w])
            continue
        if w in KNOWN_ALLERGENS:
            allg.add(KNOWN_ALLERGENS[w])
            continue
        cleaned_terms.append(w)
    intent.craving_terms = cleaned_terms
    intent.preferred_diets = sorted(diets)
    intent.allergen_terms = sorted(allg)
    return intent


def is_diet_options_query(intent: ParsedIntent) -> bool:
    return bool(intent.preferred_diets) and not intent.craving_terms and not intent.cuisine_hints


def safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def resolve_target_dt(intent: ParsedIntent, now_et: datetime) -> Tuple[str, time]:
    date_str = intent.requested_date or now_et.strftime("%Y-%m-%d")
    time_str = intent.requested_time or now_et.strftime("%H:%M")
    return date_str, parse_time_hhmm(time_str)


def resolve_hall_id(hall_hint: Optional[str], data: Dict[str, Any]) -> Optional[str]:
    if not hall_hint:
        return None
    hint = normalize_token(hall_hint)
    for h in data.get("halls", []):
        if hint == normalize_token(h.get("hall_id", "")):
            return h["hall_id"]
        if hint == normalize_token(h.get("hall_name", "")):
            return h["hall_id"]
        if hint in normalize_token(h.get("hall_name", "")):
            return h["hall_id"]
    return None


def extract_hall_from_query_local(query: str, data: Dict[str, Any]) -> Optional[str]:
    q = normalize_token(query)
    for h in data.get("halls", []):
        hall_id = normalize_token(h.get("hall_id", ""))
        hall_name = normalize_token(h.get("hall_name", ""))
        if hall_id and hall_id in q:
            return h.get("hall_id")
        if hall_name and hall_name in q:
            return h.get("hall_id")
        parts = [p for p in hall_name.replace("-", " ").split() if p]
        if parts and any(f"at {p}" in q for p in parts):
            return h.get("hall_id")
    return None


def choose_meal_from_official_hours(
    entry: Dict[str, Any],
    target_date: str,
    now_t: time,
    official_hours: Dict[str, Any],
) -> Optional[str]:
    hall_id = entry.get("hall_id", "")
    try:
        day_name = datetime.strptime(target_date, "%Y-%m-%d").strftime("%A").lower()
    except ValueError:
        return None

    hall_hours = official_hours.get(hall_id, {}).get(day_name, {})
    menu_meals = set(entry.get("meals", {}).keys())
    candidate_windows: List[Tuple[time, time, str]] = []

    for meal_name, window in hall_hours.items():
        if meal_name not in menu_meals:
            continue
        try:
            start_t = parse_time_hhmm(window["start"])
            end_t = parse_time_hhmm(window["end"])
        except Exception:
            continue
        candidate_windows.append((start_t, end_t, meal_name))

    if not candidate_windows:
        for meal_name in entry.get("meals", {}).keys():
            return meal_name
        return None

    for start_t, end_t, meal_name in candidate_windows:
        if start_t <= now_t <= end_t:
            return meal_name

    future_meals = sorted((start_t, meal_name) for start_t, _, meal_name in candidate_windows if start_t > now_t)
    if future_meals:
        return future_meals[0][1]

    candidate_windows.sort(key=lambda x: x[0])
    return candidate_windows[-1][2]


def cache_is_stale(data: Dict[str, Any], now_et: datetime, max_cache_hours: int) -> bool:
    generated_at_raw = (data.get("generated_at") or "").strip()
    if not generated_at_raw:
        return True
    try:
        generated_at = datetime.fromisoformat(generated_at_raw)
    except ValueError:
        return True
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=ZoneInfo("America/New_York"))
    age_hours = (now_et - generated_at.astimezone(ZoneInfo("America/New_York"))).total_seconds() / 3600.0
    return age_hours > max_cache_hours


def cache_has_today(data: Dict[str, Any], now_et: datetime) -> bool:
    today = now_et.strftime("%Y-%m-%d")
    for entry in data.get("menus", []):
        if entry.get("menu_date") == today:
            return True
    return False


def ensure_menu_cache(data_file: Path, now_et: datetime, max_cache_hours: int, days_ahead: int) -> Dict[str, Any]:
    reason = ""
    data: Dict[str, Any] = {}
    if not data_file.exists():
        reason = "cache file missing"
    else:
        try:
            data = load_data(data_file)
        except Exception:
            reason = "cache file unreadable"
        else:
            if cache_is_stale(data, now_et, max_cache_hours):
                reason = f"cache older than {max_cache_hours}h"
            elif not cache_has_today(data, now_et):
                reason = "cache missing current ET date menu"

    if reason:
        print(f"Refreshing menu cache ({reason})...")
        payload = scrape_menus(days_ahead=days_ahead, include_halls=None)
        write_data(data_file, payload)
        return payload
    return data


def food_match_score(intent: ParsedIntent, item_name: str, station: str) -> float:
    text = f"{item_name} {station}".lower()
    terms = intent.craving_terms + intent.cuisine_hints
    if not terms:
        return 0.2
    hits = sum(1 for t in terms if t in text)
    return min(1.0, hits / max(1, len(terms)))


def preference_fit_score(intent: ParsedIntent, diet_tags: List[str]) -> float:
    if not intent.preferred_diets:
        return 0.7
    tags = {normalize_token(x) for x in diet_tags}
    wanted = {normalize_token(x) for x in intent.preferred_diets}
    if wanted.issubset(tags):
        return 1.0
    return 0.0


def violates_hard_constraints(intent: ParsedIntent, item: Dict[str, Any]) -> bool:
    allergens = {normalize_token(a) for a in item.get("allergens", [])}
    requested = set(intent.avoid_allergens)
    return bool(allergens.intersection(requested))


def score_candidate(
    food_match: float,
    open_now: float,
    pref_fit: float,
) -> float:
    w1, w2, w3 = 0.65, 0.2, 0.15
    return (w1 * food_match) + (w2 * open_now) + (w3 * pref_fit)


def recommend(
    data: Dict[str, Any],
    intent: ParsedIntent,
    target_date: str,
    now_t: time,
    explicit_meal: Optional[str],
    top_k: int,
    hall_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    halls = {h["hall_id"]: h for h in data["halls"]}
    official_hours = data.get("official_hours", {})
    candidates: List[Dict[str, Any]] = []

    for entry in data["menus"]:
        if entry["menu_date"] != target_date:
            continue
        if hall_filter and entry["hall_id"] != hall_filter:
            continue

        hall = halls[entry["hall_id"]]
        open_meals = get_open_meals(entry, now_t)
        if explicit_meal:
            active_meals = [explicit_meal] if explicit_meal in entry.get("meals", {}) else []
        else:
            if open_meals:
                active_meals = open_meals
            else:
                inferred_meal = choose_meal_from_official_hours(
                    entry=entry,
                    target_date=target_date,
                    now_t=now_t,
                    official_hours=official_hours,
                )
                active_meals = [inferred_meal] if inferred_meal else []

        for meal in active_meals:
            for item in entry.get("meals", {}).get(meal, []):
                if violates_hard_constraints(intent, item):
                    continue

                fm = food_match_score(intent, item["item_name"], item.get("station", ""))
                pf = preference_fit_score(intent, item.get("diet_tags", []))
                if intent.craving_terms and fm <= 0:
                    continue
                if pf <= 0:
                    continue

                os = 1.0 if meal in open_meals else 0.2
                total = score_candidate(fm, os, pf)

                candidates.append(
                    {
                        "hall_id": entry["hall_id"],
                        "hall_name": hall["hall_name"],
                        "date": target_date,
                        "meal": meal,
                        "item_name": item["item_name"],
                        "station": item.get("station", ""),
                        "score": round(total, 4),
                        "why": [
                            f"matches item '{item['item_name']}'",
                            "currently available in selected meal window",
                        ],
                    }
                )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def list_diet_options(
    data: Dict[str, Any],
    target_date: str,
    meal: Optional[str],
    diets: List[str],
    avoid_allergens: List[str],
    hall_filter: Optional[str],
    limit: int = 25,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    diets_set = {normalize_token(x) for x in diets}
    fake_intent = ParsedIntent(
        craving_terms=[],
        cuisine_hints=[],
        avoid_allergens=avoid_allergens,
        allergen_terms=[],
        preferred_diets=diets,
        requested_date=None,
        requested_time=None,
        requested_meal=None,
        requested_hall=None,
        menu_lookup=False,
        hours_lookup=False,
        allergen_lookup=False,
    )
    for entry in data.get("menus", []):
        if entry.get("menu_date") != target_date:
            continue
        if hall_filter and entry.get("hall_id") != hall_filter:
            continue
        hall_name = entry.get("hall_name", entry.get("hall_id", "Unknown Hall"))
        for meal_name, items in entry.get("meals", {}).items():
            if meal and meal_name != meal:
                continue
            for item in items:
                if violates_hard_constraints(fake_intent, item):
                    continue
                tags = {normalize_token(x) for x in item.get("diet_tags", [])}
                if diets_set and not diets_set.issubset(tags):
                    continue
                key = (hall_name, meal_name, item.get("item_name", ""))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "hall_name": hall_name,
                        "meal": meal_name,
                        "item_name": item.get("item_name", ""),
                    }
                )
                if len(out) >= limit:
                    return out
    return out


def list_menu_items_for_lookup(
    data: Dict[str, Any],
    hall_id: str,
    target_date: str,
    meal: Optional[str],
) -> List[str]:
    out: List[str] = []
    for entry in data.get("menus", []):
        if entry.get("hall_id") != hall_id or entry.get("menu_date") != target_date:
            continue
        meals = entry.get("meals", {})
        if meal:
            items = meals.get(meal, [])
            out.extend([i.get("item_name", "").strip() for i in items if i.get("item_name")])
        else:
            for _, items in meals.items():
                out.extend([i.get("item_name", "").strip() for i in items if i.get("item_name")])
    unique = sorted({x for x in out if x})
    return unique


def list_hours_for_lookup(
    data: Dict[str, Any],
    target_date: str,
    hall_filter: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        day_name = datetime.strptime(target_date, "%Y-%m-%d").strftime("%A").lower()
    except ValueError:
        return []
    hours = data.get("official_hours", {})
    if not hours:
        try:
            session = requests.Session()
            hours = scrape_official_hours(session)
        except Exception:
            hours = {}
    halls = {h["hall_id"]: h.get("hall_name", h["hall_id"]) for h in data.get("halls", [])}
    out: List[Dict[str, Any]] = []
    for hall_id, day_map in hours.items():
        if hall_filter and hall_id != hall_filter:
            continue
        meals = day_map.get(day_name, {})
        if not meals:
            continue
        out.append(
            {
                "hall_id": hall_id,
                "hall_name": halls.get(hall_id, hall_id),
                "meals": meals,
            }
        )
    # Fallback for cache files created before official_hours was added.
    if not out:
        for entry in data.get("menus", []):
            if entry.get("menu_date") != target_date:
                continue
            if hall_filter and entry.get("hall_id") != hall_filter:
                continue
            meals = entry.get("hours", {})
            if not meals:
                continue
            out.append(
                {
                    "hall_id": entry.get("hall_id", ""),
                    "hall_name": entry.get("hall_name", entry.get("hall_id", "Unknown Hall")),
                    "meals": meals,
                }
            )
    out.sort(key=lambda x: x["hall_name"])
    return out


def print_hours_lookup(hours_rows: List[Dict[str, Any]], target_date: str) -> None:
    if not hours_rows:
        print(f"\nNo hours found for {target_date}.")
        return
    print(f"\nDining hall hours for {target_date}:")
    for row in hours_rows:
        print(f"{row['hall_name']}:")
        for meal, window in row["meals"].items():
            print(f"  - {meal}: {window['start']} - {window['end']}")
    print("\nDo you want menu options for one of these halls?")


def allergen_match(item: Dict[str, Any], allergen_terms: List[str]) -> bool:
    if not allergen_terms:
        return False
    item_allg = {normalize_token(a) for a in item.get("allergens", [])}
    query_allg = {normalize_token(a) for a in allergen_terms}
    return bool(item_allg.intersection(query_allg))


def list_allergen_answers(
    data: Dict[str, Any],
    intent: ParsedIntent,
    target_date: str,
    meal: Optional[str],
    hall_filter: Optional[str],
    limit: int = 20,
) -> Tuple[List[str], List[str]]:
    contains_lines: List[str] = []
    safe_lines: List[str] = []
    terms = set(intent.craving_terms + intent.cuisine_hints)
    for entry in data.get("menus", []):
        if entry.get("menu_date") != target_date:
            continue
        if hall_filter and entry.get("hall_id") != hall_filter:
            continue
        hall_name = entry.get("hall_name", entry.get("hall_id", "Unknown Hall"))
        for meal_name, items in entry.get("meals", {}).items():
            if meal and meal_name != meal:
                continue
            for item in items:
                item_name = item.get("item_name", "")
                text = f"{item_name} {item.get('station', '')}".lower()
                if terms and not any(t in text for t in terms):
                    continue
                if intent.allergen_terms and allergen_match(item, intent.allergen_terms):
                    contains_lines.append(
                        f"{item_name} at {hall_name} ({meal_name}) contains {', '.join(item.get('allergens', [])) or 'listed allergens'}."
                    )
                if intent.allergen_terms and not allergen_match(item, intent.allergen_terms):
                    safe_lines.append(f"{item_name} at {hall_name} ({meal_name})")
                if len(contains_lines) >= limit and len(safe_lines) >= limit:
                    return contains_lines[:limit], safe_lines[:limit]
    return contains_lines[:limit], safe_lines[:limit]


def find_next_available(
    data: Dict[str, Any],
    intent: ParsedIntent,
    start_date: datetime,
    days_ahead: int,
) -> Optional[Dict[str, Any]]:
    halls = {h["hall_id"]: h for h in data["halls"]}
    terms = set(intent.craving_terms + intent.cuisine_hints)
    if not terms:
        return None

    for i in range(1, days_ahead + 1):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for entry in data["menus"]:
            if entry["menu_date"] != date_str:
                continue
            hall = halls[entry["hall_id"]]
            for meal, items in entry.get("meals", {}).items():
                for item in items:
                    if violates_hard_constraints(intent, item):
                        continue
                    text = f"{item.get('item_name', '')} {item.get('station', '')}".lower()
                    if any(term in text for term in terms):
                        return {
                            "date": date_str,
                            "meal": meal,
                            "hall_name": hall["hall_name"],
                            "item_name": item["item_name"],
                        }
    return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def get_embedding(
    client: OpenAI,
    model: str,
    text: str,
    cache: Dict[str, List[float]],
) -> List[float]:
    key = text.strip().lower()
    if key in cache:
        return cache[key]
    res = client.embeddings.create(model=model, input=text)
    vec = res.data[0].embedding
    cache[key] = vec
    return vec


def suggest_similar_by_embedding(
    data: Dict[str, Any],
    intent: ParsedIntent,
    start_date: datetime,
    days_ahead: int,
    client: Optional[OpenAI],
    embedding_model: str,
    max_suggestions: int = 5,
) -> List[Dict[str, str]]:
    if client is None:
        return []

    query_text = " ".join(intent.craving_terms + intent.cuisine_hints).strip()
    if not query_text:
        return []
    emb_cache: Dict[str, List[float]] = {}
    query_vec = get_embedding(client, embedding_model, query_text, emb_cache)
    scored: List[Tuple[float, Dict[str, str]]] = []
    seen = set()

    for i in range(0, days_ahead + 1):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for entry in data.get("menus", []):
            if entry.get("menu_date") != date_str:
                continue
            hall_name = entry.get("hall_name", entry.get("hall_id", "Unknown Hall"))
            for meal, items in entry.get("meals", {}).items():
                for item in items:
                    if violates_hard_constraints(intent, item):
                        continue
                    item_name = item.get("item_name", "")
                    key = (date_str, meal, hall_name, item_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_text = f"{item_name} {item.get('station', '')}"
                    cand_vec = get_embedding(client, embedding_model, candidate_text, emb_cache)
                    sim = cosine_similarity(query_vec, cand_vec)
                    scored.append(
                        (
                            sim,
                            {
                                "date": date_str,
                                "meal": meal,
                                "hall_name": hall_name,
                                "item_name": item_name,
                            },
                        )
                    )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for sim, entry in scored[:max_suggestions] if sim > -1]


def requested_food_label(intent: ParsedIntent) -> str:
    terms = intent.craving_terms + intent.cuisine_hints
    if not terms:
        return "that dish"
    return terms[0]


def print_recommendations(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No eligible recommendation found for this time and constraints.")
        return
    print("\nTop recommendations:")
    for idx, r in enumerate(results, start=1):
        print(
            f"{idx}. {r['hall_name']} | {r['meal']} | {r['item_name']} | "
            f"score={r['score']}"
        )
        print(f"   because: {', '.join(r['why'])}")


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="UConn Eats CLI recommender (MVP starter).")
    parser.add_argument("--query", required=True, help='Example: "I want pho, avoid peanuts"')
    parser.add_argument("--meal", default="", help="Optional meal override (Lunch/Dinner)")
    parser.add_argument("--avoid-allergens", default="", help="Comma-separated allergens")
    parser.add_argument("--diets", default="", help="Comma-separated diets (vegetarian, vegan, halal)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations")
    parser.add_argument("--days-ahead", type=int, default=7, help="Search window for next availability")
    parser.add_argument(
        "--max-cache-hours",
        type=int,
        default=24,
        help="Auto-refresh scraped menu cache when older than this many hours",
    )
    parser.add_argument(
        "--data-file",
        default=str(Path(__file__).parent / "data" / "menus_scraped.json"),
        help="Path to normalized menu data JSON",
    )
    parser.add_argument(
        "--offline-intent",
        action="store_true",
        help="Skip OpenAI parsing and use local token parsing",
    )
    args = parser.parse_args()

    explicit_allergens = parse_csv_arg(args.avoid_allergens)
    explicit_diets = parse_csv_arg(args.diets)
    now_et = datetime.now(ZoneInfo("America/New_York"))

    data = ensure_menu_cache(
        data_file=Path(args.data_file),
        now_et=now_et,
        max_cache_hours=args.max_cache_hours,
        days_ahead=args.days_ahead,
    )
    client: Optional[OpenAI] = None
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if args.offline_intent:
        intent = local_parse_intent(args.query, explicit_allergens, explicit_diets, now_et)
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")
        if api_key.startswith("your_") or api_key.startswith("<") or "api_key_here" in api_key:
            raise RuntimeError(
                "OPENAI_API_KEY appears to be a placeholder. Put your real key in .env (not .env.example)."
            )
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        intent = openai_parse_intent(
            client=client,
            model=model,
            query=args.query,
            explicit_allergens=explicit_allergens,
            explicit_diets=explicit_diets,
            now_et=now_et,
        )
    intent = normalize_intent(intent)

    target_date, now_t = resolve_target_dt(intent, now_et)
    explicit_meal = args.meal if args.meal else (intent.requested_meal or None)
    hall_filter = resolve_hall_id(intent.requested_hall, data)
    if not hall_filter:
        hall_filter = extract_hall_from_query_local(args.query, data)
    print(f"Using target datetime (ET): {target_date} {now_t.strftime('%H:%M')}")

    if intent.menu_lookup and hall_filter:
        items = list_menu_items_for_lookup(
            data=data,
            hall_id=hall_filter,
            target_date=target_date,
            meal=explicit_meal,
        )
        hall_name = next((h["hall_name"] for h in data.get("halls", []) if h.get("hall_id") == hall_filter), hall_filter)
        meal_label = explicit_meal or "All Meals"
        if items:
            print(f"\n{hall_name} | {target_date} | {meal_label}")
            for i, item in enumerate(items[:25], start=1):
                print(f"{i}. {item}")
            if len(items) > 25:
                print(f"...and {len(items) - 25} more items.")
        else:
            print(f"\nNo menu items found for {hall_name} on {target_date} ({meal_label}).")
        print("\nIs there something special you want to eat?")
        return

    if intent.hours_lookup:
        rows = list_hours_for_lookup(
            data=data,
            target_date=target_date,
            hall_filter=hall_filter,
        )
        print_hours_lookup(rows, target_date)
        return

    if intent.allergen_lookup and intent.allergen_terms:
        contains_lines, safe_lines = list_allergen_answers(
            data=data,
            intent=intent,
            target_date=target_date,
            meal=explicit_meal,
            hall_filter=hall_filter,
        )
        focus = ", ".join(intent.allergen_terms)
        if contains_lines:
            print(f"\nItems mentioning {focus}:")
            for i, line in enumerate(contains_lines[:10], start=1):
                print(f"{i}. {line}")
        if safe_lines:
            print(f"\nPossible options without {focus}:")
            for i, line in enumerate(safe_lines[:10], start=1):
                print(f"{i}. {line}")
        if not contains_lines and not safe_lines:
            print(f"\nI could not find allergen-specific results for {focus} on {target_date}.")
        print("\nWould you like me to narrow this by hall or meal?")
        return

    if is_diet_options_query(intent):
        options = list_diet_options(
            data=data,
            target_date=target_date,
            meal=explicit_meal,
            diets=intent.preferred_diets,
            avoid_allergens=intent.avoid_allergens,
            hall_filter=hall_filter,
            limit=25,
        )
        diet_label = ", ".join(intent.preferred_diets)
        meal_label = explicit_meal or "all meals"
        if options:
            print(f"\n{diet_label.title()} options for {target_date} ({meal_label}):")
            for i, o in enumerate(options, start=1):
                print(f"{i}. {o['item_name']} at {o['hall_name']} ({o['meal']})")
        else:
            print(f"\nNo {diet_label} options found for {target_date} ({meal_label}).")
        print("\nIs there something special you want to eat?")
        return

    results = recommend(
        data=data,
        intent=intent,
        target_date=target_date,
        now_t=now_t,
        explicit_meal=explicit_meal,
        top_k=args.top_k,
        hall_filter=hall_filter,
    )
    print_recommendations(results)

    if not results:
        next_option = find_next_available(
            data=data,
            intent=intent,
            start_date=datetime.strptime(target_date, "%Y-%m-%d"),
            days_ahead=args.days_ahead,
        )
        if next_option:
            print("\nNot available now.")
            print(
                f"Next likely match: {next_option['hall_name']} on {next_option['date']} "
                f"({next_option['meal']}) - {next_option['item_name']}."
            )
            print("You can use this output to create an alert workflow next.")
        else:
            if is_diet_options_query(intent):
                print("\nNo matching dietary options found in the configured lookahead window.")
                return
            similar = suggest_similar_by_embedding(
                data=data,
                intent=intent,
                start_date=datetime.strptime(target_date, "%Y-%m-%d"),
                days_ahead=args.days_ahead,
                client=client,
                embedding_model=embedding_model,
            )
            if similar:
                food = requested_food_label(intent)
                print(f"\nSorry, but {food} is not on the menu this week, but how about...")
                for i, s in enumerate(similar, start=1):
                    print(f"{i}. {s['item_name']} at {s['hall_name']} on {s['date']} ({s['meal']})")
            else:
                print("\nNo matching item found in the configured lookahead window.")


if __name__ == "__main__":
    main()
