import argparse
import io
import json
import math
import os
import re
import requests
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from .menu_scraper import scrape_menus, scrape_official_hours


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

DEFAULT_TOP_K = 3
DEFAULT_DAYS_AHEAD = 7
DEFAULT_MAX_CACHE_HOURS = 24
DEFAULT_DATA_FILE = Path(__file__).parent.parent / "data" / "menus_scraped.json"


def _candidate_models(primary: str, fallbacks: List[str]) -> List[str]:
    out: List[str] = []
    for m in [primary] + fallbacks:
        m = (m or "").strip()
        if m and m not in out:
            out.append(m)
    return out


def _is_model_not_found_error(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", None) or {}
    err = body.get("error", {}) if isinstance(body, dict) else {}
    code = err.get("code")
    msg = str(err.get("message", "")).lower()
    return code == "model_not_found" or "does not exist" in msg


def _responses_text_with_model_fallback(client: OpenAI, model: str, prompt: str) -> str:
    response = None
    models_to_try = _candidate_models(model, ["gpt-5-mini", "gpt-4.1-mini"])
    last_error: Optional[Exception] = None
    for m in models_to_try:
        try:
            response = client.responses.create(model=m, input=prompt)
            if m != model:
                print(f"Warning: model '{model}' unavailable; using '{m}' instead.")
            break
        except BadRequestError as exc:
            last_error = exc
            if _is_model_not_found_error(exc):
                continue
            raise
    if response is None:
        if last_error:
            raise last_error
        raise RuntimeError("Unable to call OpenAI responses API with the configured model.")
    return (response.output_text or "").strip()


def generate_user_reply(client: Optional[OpenAI], model: str, payload: Dict[str, Any]) -> Optional[str]:
    if client is None:
        return None
    prompt = (
        "You are UConn Eats, a friendly campus dining assistant.\n"
        "Write a short, conversational response from the provided JSON only.\n"
        "Rules: use plain language, max 5 listed items, and end with one follow-up question.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )
    try:
        text = _responses_text_with_model_fallback(client, model, prompt)
        return text if text else None
    except Exception:
        return None


def natural_join(items: List[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def format_menu_lookup(hall_name: str, target_date: str, meal_label: str, items: List[str]) -> str:
    if not items:
        return (
            f"I couldn't find menu items for {hall_name} on {target_date} during {meal_label}. "
            "Try another hall, meal, or date and I'll take another look."
        )
    highlights = natural_join(items[:5])
    extra = f" There are {len(items) - 5} more items as well." if len(items) > 5 else ""
    return (
        f"{hall_name} is serving {meal_label.lower()} on {target_date}. "
        f"Some options I found are {highlights}.{extra} "
        "If you want, I can narrow that down by diet, allergen, or craving."
    )


def format_hours_lookup(hours_rows: List[Dict[str, Any]], target_date: str) -> str:
    if not hours_rows:
        return (
            f"I couldn't find dining hall hours for {target_date}. "
            "Try another date or hall, and I can check again."
        )

    hall_summaries = []
    for row in hours_rows[:4]:
        windows = [
            f"{meal} from {window['start']} to {window['end']}"
            for meal, window in row["meals"].items()
        ]
        hall_summaries.append(f"{row['hall_name']} is open for {natural_join(windows)}")

    summary = " ".join(f"{entry}." for entry in hall_summaries)
    if len(hours_rows) > 4:
        summary += f" I found hours for {len(hours_rows) - 4} more halls too."
    return summary + " If you want, I can also show what one of those halls is serving."


def format_allergen_lookup(
    contains_lines: List[str],
    safe_lines: List[str],
    focus: str,
    target_date: str,
) -> str:
    if not contains_lines and not safe_lines:
        return (
            f"I couldn't find clear allergen-specific results for {focus} on {target_date}. "
            "Try adding a hall, meal, or item name and I can narrow it down."
        )

    parts: List[str] = []
    if contains_lines:
        parts.append(
            f"I found a few items that mention {focus}, including {natural_join(contains_lines[:3])}."
        )
    if safe_lines:
        parts.append(
            f"I also found possible options without {focus}, such as {natural_join(safe_lines[:3])}."
        )
    parts.append("If you want, I can narrow that down by hall or meal.")
    return " ".join(parts)


def format_diet_options(options: List[Dict[str, str]], diet_label: str, target_date: str, meal_label: str) -> str:
    if not options:
        return (
            f"I couldn't find {diet_label} options for {target_date} during {meal_label}. "
            "Try another hall, meal, or date and I can check again."
        )
    picks = natural_join(
        [f"{option['item_name']} at {option['hall_name']} for {option['meal']}" for option in options[:5]]
    )
    extra = f" There are {len(options) - 5} more matches too." if len(options) > 5 else ""
    return (
        f"I found some {diet_label} options for {target_date} during {meal_label}. "
        f"A few good ones are {picks}.{extra} "
        "If you want, I can also filter those by hall or allergen."
    )


def format_recommendations(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "I couldn't find a good match right now."
    picks = []
    for result in results[:3]:
        reason = natural_join(result.get("why", [])[:2])
        if reason:
            picks.append(f"{result['item_name']} at {result['hall_name']} for {result['meal']} because it {reason}")
        else:
            picks.append(f"{result['item_name']} at {result['hall_name']} for {result['meal']}")
    return (
        f"A few good options right now are {natural_join(picks)}. "
        "If you want, I can narrow that down by dining hall, meal, or dietary preference."
    )


def format_next_available(next_option: Dict[str, Any]) -> str:
    return (
        f"I couldn't find that exact match right now, but the next good option is "
        f"{next_option['item_name']} at {next_option['hall_name']} on {next_option['date']} during {next_option['meal']}. "
        "If you want, I can also suggest similar dishes that show up sooner."
    )


def format_similar_fallback(food: str, suggestions: List[Dict[str, str]]) -> str:
    picks = natural_join(
        [f"{item['item_name']} at {item['hall_name']} on {item['date']} during {item['meal']}" for item in suggestions[:4]]
    )
    return (
        f"I couldn't find {food} on the menu this week, but a few similar options are {picks}. "
        "If you want, I can keep narrowing by hall, meal, or cuisine."
    )


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
    raw = _responses_text_with_model_fallback(client, model, prompt)
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
    print(f"\n{format_hours_lookup(hours_rows, target_date)}")


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
    models_to_try = _candidate_models(model, ["text-embedding-3-small", "text-embedding-3-large"])
    res = None
    last_error: Optional[Exception] = None
    for m in models_to_try:
        try:
            res = client.embeddings.create(model=m, input=text)
            if m != model:
                print(f"Warning: embedding model '{model}' unavailable; using '{m}' instead.")
            break
        except BadRequestError as exc:
            last_error = exc
            if _is_model_not_found_error(exc):
                continue
            raise
    if res is None:
        if last_error:
            raise last_error
        raise RuntimeError("Unable to call OpenAI embeddings API with the configured model.")
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
    print(f"\n{format_recommendations(results)}")


def build_openai_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")
    if api_key.startswith("your_") or api_key.startswith("<") or "api_key_here" in api_key:
        raise RuntimeError(
            "OPENAI_API_KEY appears to be a placeholder. Put your real key in .env (not .env.example)."
        )
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def _execute_query(
    query: str,
    data_file: Path,
    max_cache_hours: int,
    explicit_allergens: Optional[List[str]] = None,
    explicit_diets: Optional[List[str]] = None,
) -> None:
    load_dotenv(override=True)
    explicit_allergens = explicit_allergens or []
    explicit_diets = explicit_diets or []
    now_et = datetime.now(ZoneInfo("America/New_York"))

    data = ensure_menu_cache(
        data_file=data_file,
        now_et=now_et,
        max_cache_hours=max_cache_hours,
        days_ahead=DEFAULT_DAYS_AHEAD,
    )
    client: Optional[OpenAI] = None
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    client = build_openai_client(api_key=api_key, base_url=base_url)
    intent = openai_parse_intent(
        client=client,
        model=model,
        query=query,
        explicit_allergens=explicit_allergens,
        explicit_diets=explicit_diets,
        now_et=now_et,
    )
    intent = normalize_intent(intent)

    target_date, now_t = resolve_target_dt(intent, now_et)
    explicit_meal = intent.requested_meal or None
    hall_filter = resolve_hall_id(intent.requested_hall, data)
    if not hall_filter:
        hall_filter = extract_hall_from_query_local(query, data)
    print(f"Looking at options for {target_date} around {now_t.strftime('%H:%M')} ET.")

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
            llm_reply = generate_user_reply(
                client,
                model,
                {
                    "mode": "menu_lookup",
                    "hall": hall_name,
                    "date": target_date,
                    "meal": meal_label,
                    "items": items[:25],
                },
            )
            if llm_reply:
                print(f"\n{llm_reply}")
                return
            print(f"\n{format_menu_lookup(hall_name, target_date, meal_label, items)}")
        else:
            print(f"\n{format_menu_lookup(hall_name, target_date, meal_label, items)}")
        return

    if intent.hours_lookup:
        rows = list_hours_for_lookup(
            data=data,
            target_date=target_date,
            hall_filter=hall_filter,
        )
        llm_reply = generate_user_reply(
            client,
            model,
            {
                "mode": "hours_lookup",
                "date": target_date,
                "rows": rows[:8],
            },
        )
        if llm_reply:
            print(f"\n{llm_reply}")
            return
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
        llm_reply = generate_user_reply(
            client,
            model,
            {
                "mode": "allergen_lookup",
                "date": target_date,
                "allergens": intent.allergen_terms,
                "contains_items": contains_lines[:10],
                "possible_options": safe_lines[:10],
            },
        )
        if llm_reply:
            print(f"\n{llm_reply}")
            return
        print(f"\n{format_allergen_lookup(contains_lines, safe_lines, focus, target_date)}")
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
        llm_reply = generate_user_reply(
            client,
            model,
            {
                "mode": "diet_options",
                "date": target_date,
                "meal": meal_label,
                "diets": intent.preferred_diets,
                "options": options[:25],
            },
        )
        if llm_reply:
            print(f"\n{llm_reply}")
            return
        print(f"\n{format_diet_options(options, diet_label, target_date, meal_label)}")
        return

    results = recommend(
        data=data,
        intent=intent,
        target_date=target_date,
        now_t=now_t,
        explicit_meal=explicit_meal,
        top_k=DEFAULT_TOP_K,
        hall_filter=hall_filter,
    )
    if results:
        llm_reply = generate_user_reply(
            client,
            model,
                {
                    "mode": "recommendations",
                    "date": target_date,
                    "time_et": now_t.strftime("%H:%M"),
                    "results": results[:5],
            },
        )
        if llm_reply:
            print(f"\n{llm_reply}")
        else:
            print_recommendations(results)
    else:
        print_recommendations(results)

    if not results:
        next_option = find_next_available(
            data=data,
            intent=intent,
            start_date=datetime.strptime(target_date, "%Y-%m-%d"),
            days_ahead=DEFAULT_DAYS_AHEAD,
        )
        if next_option:
            llm_reply = generate_user_reply(
                client,
                model,
                {
                    "mode": "next_available",
                    "query": query,
                    "next_option": next_option,
                },
            )
            if llm_reply:
                print(f"\n{llm_reply}")
            else:
                print(f"\n{format_next_available(next_option)}")
        else:
            if is_diet_options_query(intent):
                print("\nI couldn't find matching dietary options in the next few days.")
                return
            similar = suggest_similar_by_embedding(
                data=data,
                intent=intent,
                start_date=datetime.strptime(target_date, "%Y-%m-%d"),
                days_ahead=DEFAULT_DAYS_AHEAD,
                client=client,
                embedding_model=embedding_model,
            )
            if similar:
                llm_reply = generate_user_reply(
                    client,
                    model,
                    {
                        "mode": "similar_fallback",
                        "requested_food": requested_food_label(intent),
                        "suggestions": similar[:5],
                    },
                )
                if llm_reply:
                    print(f"\n{llm_reply}")
                else:
                    food = requested_food_label(intent)
                    print(f"\n{format_similar_fallback(food, similar)}")
            else:
                llm_reply = generate_user_reply(
                    client,
                    model,
                    {
                        "mode": "no_match",
                        "query": query,
                    },
                )
                if llm_reply:
                    print(f"\n{llm_reply}")
                else:
                    print("\nI couldn't find a close match in the next few days.")


def run_query(
    query: str,
    data_file: str | Path = DEFAULT_DATA_FILE,
    max_cache_hours: int = DEFAULT_MAX_CACHE_HOURS,
    explicit_allergens: Optional[List[str]] = None,
    explicit_diets: Optional[List[str]] = None,
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _execute_query(
            query=query,
            data_file=Path(data_file),
            max_cache_hours=max_cache_hours,
            explicit_allergens=explicit_allergens,
            explicit_diets=explicit_diets,
        )
    return buffer.getvalue().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="UConn Eats CLI recommender (MVP starter).")
    parser.add_argument("--query", required=True, help='Example: "I want pho, avoid peanuts"')
    parser.add_argument(
        "--max-cache-hours",
        type=int,
        default=DEFAULT_MAX_CACHE_HOURS,
        help="Auto-refresh scraped menu cache when older than this many hours",
    )
    parser.add_argument(
        "--data-file",
        default=str(DEFAULT_DATA_FILE),
        help="Path to normalized menu data JSON",
    )
    args = parser.parse_args()
    output = run_query(
        query=args.query,
        data_file=args.data_file,
        max_cache_hours=args.max_cache_hours,
    )
    if output:
        print(output)


if __name__ == "__main__":
    main()
