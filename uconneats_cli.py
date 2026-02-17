import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class ParsedIntent:
    craving_terms: List[str]
    cuisine_hints: List[str]
    avoid_allergens: List[str]
    preferred_diets: List[str]
    avoid_crowds: bool


def load_data(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_token(token: str) -> str:
    return token.strip().lower()


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


def meal_for_time(now_t: time) -> str:
    if now_t < time(15, 0):
        return "Lunch"
    return "Dinner"


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
    avoid_crowds: bool,
) -> ParsedIntent:
    prompt = f"""
Extract user dining intent into strict JSON.

User query: {query}
Explicit allergens to avoid: {explicit_allergens}
Explicit diet preferences: {explicit_diets}
Avoid crowds preference: {avoid_crowds}

Return JSON only with this schema:
{{
  "craving_terms": ["..."],
  "cuisine_hints": ["..."],
  "avoid_allergens": ["..."],
  "preferred_diets": ["..."],
  "avoid_crowds": true
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
        preferred_diets=[normalize_token(x) for x in parsed.get("preferred_diets", [])],
        avoid_crowds=bool(parsed.get("avoid_crowds", avoid_crowds)),
    )


def local_parse_intent(
    query: str,
    explicit_allergens: List[str],
    explicit_diets: List[str],
    avoid_crowds: bool,
) -> ParsedIntent:
    stopwords = {"i", "want", "need", "some", "food", "please", "me", "to", "eat"}
    tokens = [
        normalize_token(t)
        for t in query.replace("/", " ").split()
        if t.strip() and normalize_token(t) not in stopwords
    ]
    return ParsedIntent(
        craving_terms=tokens,
        cuisine_hints=[],
        avoid_allergens=explicit_allergens,
        preferred_diets=explicit_diets,
        avoid_crowds=avoid_crowds,
    )


class OccupancyForecaster:
    def __init__(self, db_url: Optional[str]) -> None:
        self.db_url = db_url or ""

    def forecast(self, hall_id: str, target_dt: datetime) -> Tuple[int, str, float]:
        # Placeholder until SQL integration is added.
        # Uses deterministic 45-minute block baseline so local testing is stable.
        block = (target_dt.hour * 60 + target_dt.minute) // 45
        seed = abs(hash(f"{hall_id}-{target_dt.weekday()}-{block}")) % 101
        occupancy = int(seed)
        if occupancy < 34:
            band = "Low"
        elif occupancy < 67:
            band = "Medium"
        else:
            band = "High"
        confidence = 0.55
        return occupancy, band, confidence


def safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


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


def distance_score(minutes: int) -> float:
    return max(0.0, 1.0 - (minutes / 20.0))


def crowd_score(occupancy: int, avoid_crowds: bool) -> float:
    if avoid_crowds:
        return max(0.0, 1.0 - occupancy / 100.0)
    return 0.5


def score_candidate(
    food_match: float,
    distance: float,
    open_now: float,
    crowd: float,
    pref_fit: float,
) -> float:
    w1, w2, w3, w4, w5 = 0.35, 0.2, 0.2, 0.15, 0.1
    return (w1 * food_match) + (w2 * distance) + (w3 * open_now) + (w4 * crowd) + (w5 * pref_fit)


def recommend(
    data: Dict[str, Any],
    intent: ParsedIntent,
    target_date: str,
    now_t: time,
    location_hint: str,
    explicit_meal: Optional[str],
    top_k: int,
    forecaster: OccupancyForecaster,
) -> List[Dict[str, Any]]:
    halls = {h["hall_id"]: h for h in data["halls"]}
    candidates: List[Dict[str, Any]] = []

    for entry in data["menus"]:
        if entry["menu_date"] != target_date:
            continue

        hall = halls[entry["hall_id"]]
        open_meals = get_open_meals(entry, now_t)
        if explicit_meal:
            active_meals = [explicit_meal] if explicit_meal in entry.get("meals", {}) else []
        else:
            active_meals = open_meals if open_meals else [meal_for_time(now_t)]

        for meal in active_meals:
            for item in entry.get("meals", {}).get(meal, []):
                if violates_hard_constraints(intent, item):
                    continue

                occupancy, crowd_band, confidence = forecaster.forecast(
                    entry["hall_id"],
                    datetime.combine(datetime.strptime(target_date, "%Y-%m-%d").date(), now_t),
                )
                walking_min = hall.get("walking_minutes_from", {}).get(location_hint, 12)
                fm = food_match_score(intent, item["item_name"], item.get("station", ""))
                pf = preference_fit_score(intent, item.get("diet_tags", []))
                if intent.craving_terms and fm <= 0:
                    continue
                if pf <= 0:
                    continue

                ds = distance_score(walking_min)
                os = 1.0 if meal in open_meals else 0.2
                cs = crowd_score(occupancy, intent.avoid_crowds)
                total = score_candidate(fm, ds, os, cs, pf)

                candidates.append(
                    {
                        "hall_id": entry["hall_id"],
                        "hall_name": hall["hall_name"],
                        "date": target_date,
                        "meal": meal,
                        "item_name": item["item_name"],
                        "station": item.get("station", ""),
                        "walking_minutes": walking_min,
                        "occupancy_index": occupancy,
                        "crowd_band": crowd_band,
                        "forecast_confidence": confidence,
                        "score": round(total, 4),
                        "why": [
                            f"matches item '{item['item_name']}'",
                            f"{walking_min} min walk",
                            f"predicted crowd {crowd_band} ({occupancy}/100)",
                        ],
                    }
                )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def find_next_available(
    data: Dict[str, Any],
    intent: ParsedIntent,
    start_date: datetime,
    location_hint: str,
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
            walking_min = hall.get("walking_minutes_from", {}).get(location_hint, 12)
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
                            "walking_minutes": walking_min,
                        }
    return None


def print_recommendations(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No eligible recommendation found for this time and constraints.")
        return
    print("\nTop recommendations:")
    for idx, r in enumerate(results, start=1):
        print(
            f"{idx}. {r['hall_name']} | {r['meal']} | {r['item_name']} | "
            f"score={r['score']} | crowd={r['crowd_band']} ({r['occupancy_index']}/100) | "
            f"walk={r['walking_minutes']} min"
        )
        print(f"   because: {', '.join(r['why'])}")


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="UConn Eats CLI recommender (MVP starter).")
    parser.add_argument("--query", required=True, help='Example: "I want pho, avoid peanuts, low crowd"')
    parser.add_argument("--location", default="student union", help="Starting point (e.g., student union)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target date YYYY-MM-DD")
    parser.add_argument("--time", default=datetime.now().strftime("%H:%M"), help="Target time HH:MM (24h)")
    parser.add_argument("--meal", default="", help="Optional meal override (Lunch/Dinner)")
    parser.add_argument("--avoid-allergens", default="", help="Comma-separated allergens")
    parser.add_argument("--diets", default="", help="Comma-separated diets (vegetarian, vegan, halal)")
    parser.add_argument("--avoid-crowds", action="store_true", help="Prefer lower occupancy halls")
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations")
    parser.add_argument("--days-ahead", type=int, default=7, help="Search window for next availability")
    parser.add_argument(
        "--data-file",
        default=str(Path(__file__).parent / "data" / "sample_menus.json"),
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
    location_hint = normalize_token(args.location)
    now_t = parse_time_hhmm(args.time)

    data = load_data(Path(args.data_file))
    db_url = os.getenv("UCONN_EATS_DB_URL", "")
    forecaster = OccupancyForecaster(db_url=db_url)

    if args.offline_intent:
        intent = local_parse_intent(args.query, explicit_allergens, explicit_diets, args.avoid_crowds)
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
            avoid_crowds=args.avoid_crowds,
        )

    results = recommend(
        data=data,
        intent=intent,
        target_date=args.date,
        now_t=now_t,
        location_hint=location_hint,
        explicit_meal=args.meal if args.meal else None,
        top_k=args.top_k,
        forecaster=forecaster,
    )
    print_recommendations(results)

    if not results:
        next_option = find_next_available(
            data=data,
            intent=intent,
            start_date=datetime.strptime(args.date, "%Y-%m-%d"),
            location_hint=location_hint,
            days_ahead=args.days_ahead,
        )
        if next_option:
            print("\nNot available now.")
            print(
                f"Next likely match: {next_option['hall_name']} on {next_option['date']} "
                f"({next_option['meal']}) - {next_option['item_name']} "
                f"({next_option['walking_minutes']} min walk)."
            )
            print("You can use this output to create an alert workflow next.")
        else:
            print("\nNo matching item found in the configured lookahead window.")


if __name__ == "__main__":
    main()
