from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from uconneats import cli


def test_parse_datetime_from_query_local_relative_time():
    now_et = datetime(2026, 2, 17, 13, 0, 0)
    d, t, meal = cli.parse_datetime_from_query_local("i want ramen tomorrow at 6:30 pm", now_et)
    assert d == "2026-02-18"
    assert t == "18:30"
    assert meal is None


def test_parse_datetime_from_query_local_meal_and_today():
    now_et = datetime(2026, 2, 17, 9, 0, 0)
    d, t, meal = cli.parse_datetime_from_query_local("lunch today", now_et)
    assert d == "2026-02-17"
    assert t is None
    assert meal == "Lunch"


def test_choose_meal_from_official_hours_picks_current_window():
    entry = {
        "hall_id": "south",
        "meals": {"Lunch": [{"item_name": "X"}], "Dinner": [{"item_name": "Y"}]},
    }
    hours = {
        "south": {
            "monday": {
                "Lunch": {"start": "11:00", "end": "15:00"},
                "Dinner": {"start": "16:30", "end": "19:15"},
            }
        }
    }
    meal = cli.choose_meal_from_official_hours(entry, "2026-02-16", cli.parse_time_hhmm("12:30"), hours)
    assert meal == "Lunch"


def test_choose_meal_from_official_hours_picks_next_future_meal():
    entry = {
        "hall_id": "south",
        "meals": {"Lunch": [{"item_name": "X"}], "Dinner": [{"item_name": "Y"}]},
    }
    hours = {
        "south": {
            "monday": {
                "Lunch": {"start": "11:00", "end": "15:00"},
                "Dinner": {"start": "16:30", "end": "19:15"},
            }
        }
    }
    meal = cli.choose_meal_from_official_hours(entry, "2026-02-16", cli.parse_time_hhmm("10:00"), hours)
    assert meal == "Lunch"


def test_cache_helpers():
    now_et = datetime(2026, 2, 17, 12, 0, 0, tzinfo=cli.ZoneInfo("America/New_York"))
    fresh = {"generated_at": "2026-02-17T08:30:00-05:00", "menus": [{"menu_date": "2026-02-17"}]}
    stale = {"generated_at": "2026-02-15T08:30:00-05:00", "menus": [{"menu_date": "2026-02-15"}]}
    assert cli.cache_is_stale(fresh, now_et, max_cache_hours=24) is False
    assert cli.cache_is_stale(stale, now_et, max_cache_hours=24) is True
    assert cli.cache_has_today(fresh, now_et) is True
    assert cli.cache_has_today(stale, now_et) is False


def test_ensure_menu_cache_refresh_when_missing(tmp_path, monkeypatch):
    target = tmp_path / "menus.json"

    fake_payload = {
        "generated_at": "2026-02-17T09:00:00-05:00",
        "menus": [{"menu_date": "2026-02-17"}],
    }

    called = {"count": 0}

    def fake_scrape(days_ahead, include_halls):
        called["count"] += 1
        return fake_payload

    monkeypatch.setattr(cli, "scrape_menus", fake_scrape)
    data = cli.ensure_menu_cache(
        data_file=target,
        now_et=datetime(2026, 2, 17, 12, 0, tzinfo=cli.ZoneInfo("America/New_York")),
        max_cache_hours=24,
        days_ahead=7,
    )
    assert called["count"] == 1
    assert target.exists()
    assert data["generated_at"] == fake_payload["generated_at"]


class _FakeEmbeddings:
    def create(self, model, input):
        text = input.lower()
        if "ramen" in text or "noodle" in text or "pho" in text:
            vec = [1.0, 0.0]
        elif "pizza" in text or "pasta" in text:
            vec = [0.0, 1.0]
        else:
            vec = [0.3, 0.3]
        return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


def test_suggest_similar_by_embedding_returns_semantic_matches():
    data = {
        "menus": [
            {
                "menu_date": "2026-02-17",
                "hall_name": "Northwest",
                "meals": {"Dinner": [{"item_name": "Chicken Ramen", "station": "Soup", "allergens": []}]},
            },
            {
                "menu_date": "2026-02-17",
                "hall_name": "South",
                "meals": {"Dinner": [{"item_name": "Pepperoni Pizza", "station": "Pizza", "allergens": []}]},
            },
        ]
    }
    intent = cli.ParsedIntent(
        craving_terms=["ramen"],
        cuisine_hints=[],
        avoid_allergens=[],
        allergen_terms=[],
        preferred_diets=[],
        requested_date=None,
        requested_time=None,
        requested_meal=None,
        requested_hall=None,
        menu_lookup=False,
        hours_lookup=False,
        allergen_lookup=False,
    )
    out = cli.suggest_similar_by_embedding(
        data=data,
        intent=intent,
        start_date=datetime(2026, 2, 17),
        days_ahead=0,
        client=_FakeOpenAIClient(),
        embedding_model="text-embedding-3-small",
        max_suggestions=2,
    )
    assert out
    assert out[0]["item_name"] == "Chicken Ramen"


def test_normalize_intent_moves_diet_terms_out_of_craving():
    intent = cli.ParsedIntent(
        craving_terms=["what", "vegetarian", "options"],
        cuisine_hints=[],
        avoid_allergens=[],
        allergen_terms=[],
        preferred_diets=[],
        requested_date=None,
        requested_time=None,
        requested_meal=None,
        requested_hall=None,
        menu_lookup=False,
        hours_lookup=False,
        allergen_lookup=False,
    )
    out = cli.normalize_intent(intent)
    assert out.preferred_diets == ["vegetarian"]
    assert out.craving_terms == []
    assert cli.is_diet_options_query(out) is True


def test_list_diet_options_returns_tagged_items():
    data = {
        "menus": [
            {
                "menu_date": "2026-02-18",
                "hall_name": "South",
                "hall_id": "south",
                "meals": {
                    "Lunch": [
                        {"item_name": "Chickpea Curry Bowl", "diet_tags": ["vegetarian", "vegan"], "allergens": []},
                        {"item_name": "Beef Burger", "diet_tags": [], "allergens": []},
                    ]
                },
            }
        ]
    }
    out = cli.list_diet_options(
        data=data,
        target_date="2026-02-18",
        meal="Lunch",
        diets=["vegetarian"],
        avoid_allergens=[],
        hall_filter=None,
        limit=10,
    )
    assert len(out) == 1
    assert out[0]["item_name"] == "Chickpea Curry Bowl"


def test_extract_allergen_terms():
    out = cli.extract_allergen_terms("does this contain peanuts or dairy")
    assert "peanuts" in out
    assert "milk" in out


def test_list_hours_for_lookup():
    data = {
        "halls": [{"hall_id": "south", "hall_name": "South"}],
        "official_hours": {
            "south": {
                "wednesday": {
                    "Lunch": {"start": "11:00", "end": "15:00"},
                    "Dinner": {"start": "16:30", "end": "19:15"},
                }
            }
        },
    }
    rows = cli.list_hours_for_lookup(data, target_date="2026-02-18", hall_filter="south")
    assert len(rows) == 1
    assert rows[0]["hall_name"] == "South"
    assert "Lunch" in rows[0]["meals"]


def test_run_query_returns_formatted_text(monkeypatch, tmp_path):
    fake_data = {
        "generated_at": "2026-02-17T08:30:00-05:00",
        "halls": [{"hall_id": "south", "hall_name": "South"}],
        "menus": [
            {
                "menu_date": "2026-02-17",
                "hall_id": "south",
                "hall_name": "South",
                "meals": {"Dinner": [{"item_name": "BBQ Tofu", "station": "Grill", "allergens": [], "diet_tags": ["vegan"]}]},
            }
        ],
        "official_hours": {},
    }

    monkeypatch.setattr(cli, "ensure_menu_cache", lambda data_file, now_et, max_cache_hours, days_ahead: fake_data)
    monkeypatch.setattr(
        cli,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: datetime(2026, 2, 17, 18, 0, tzinfo=tz),
            strptime=datetime.strptime,
        ),
    )
    monkeypatch.setattr(cli, "build_openai_client", lambda api_key, base_url: object())
    monkeypatch.setattr(
        cli,
        "openai_parse_intent",
        lambda client, model, query, explicit_allergens, explicit_diets, now_et: cli.ParsedIntent(
            craving_terms=["what's", "for", "dinner", "tonight", "at", "south?"],
            cuisine_hints=[],
            avoid_allergens=[],
            allergen_terms=[],
            preferred_diets=[],
            requested_date="2026-02-17",
            requested_time=None,
            requested_meal="Dinner",
            requested_hall="South",
            menu_lookup=True,
            hours_lookup=False,
            allergen_lookup=False,
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    output = cli.run_query(
        query="What's for dinner tonight at South?",
        data_file=tmp_path / "menus.json",
        max_cache_hours=24,
    )
    assert "South is serving dinner on 2026-02-17" in output
    assert "BBQ Tofu" in output


def test_format_hours_lookup_is_conversational():
    text = cli.format_hours_lookup(
        [
            {
                "hall_name": "South",
                "meals": {
                    "Breakfast": {"start": "07:00", "end": "10:45"},
                    "Lunch": {"start": "11:00", "end": "15:00"},
                },
            }
        ],
        "2026-02-18",
    )
    assert "South is open" in text
    assert "Breakfast from 07:00 to 10:45" in text
    assert "show what one of those halls is serving" in text


def test_format_recommendations_is_conversational():
    text = cli.format_recommendations(
        [
            {
                "item_name": "Chicken Ramen",
                "hall_name": "Northwest",
                "meal": "Dinner",
                "why": ["matches your craving", "is available now"],
            }
        ]
    )
    assert "A few good options right now are" in text
    assert "Chicken Ramen at Northwest for Dinner" in text
