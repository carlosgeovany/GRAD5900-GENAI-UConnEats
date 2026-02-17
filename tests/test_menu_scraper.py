from datetime import datetime
from types import SimpleNamespace

from bs4 import BeautifulSoup

from uconneats import menu_scraper as ms


def test_parse_time_token_with_meridiem_propagation():
    hhmm, mer = ms.parse_time_token("7:15pm")
    assert hhmm == "19:15"
    assert mer == "pm"

    hhmm2, mer2 = ms.parse_time_token("4:00", default_meridiem=mer)
    assert hhmm2 == "16:00"
    assert mer2 == "pm"


def test_parse_time_range():
    rng = ms.parse_time_range("4:00pm-7:15pm")
    assert rng == ("16:00", "19:15")

    rng2 = ms.parse_time_range("4-7:15pm")
    assert rng2 == ("16:00", "19:15")


def test_normalize_hall_label_grouped():
    halls = ms.normalize_hall_label("NORTH, SOUTH*, GELFENBIEN, WHITNEY")
    assert halls == ["north", "south", "gelfenbien", "whitney"]


def test_parse_hours_section_weekday_and_weekend():
    weekday_text = """
    CONNECTICUT HALL & PUTNAM
    Breakfast: 7am-10:45am
    Lunch: 11:00am-2:30pm
    Dinner: 4:00pm-7:15pm
    """
    weekend_text = """
    SOUTH
    Breakfast: (Saturday) 7:00am-9:30am
    Breakfast: (Sunday) 8:00am-9:30am
    Brunch: 9:30am-3pm
    Dinner: 4:30-7:15pm
    """
    wk = ms.parse_hours_section(weekday_text, weekday_mode=True)
    we = ms.parse_hours_section(weekend_text, weekday_mode=False)

    assert wk["connecticut"]["monday"]["Lunch"] == {"start": "11:00", "end": "14:30"}
    assert wk["putnam"]["friday"]["Dinner"] == {"start": "16:00", "end": "19:15"}
    assert we["south"]["saturday"]["Breakfast"] == {"start": "07:00", "end": "09:30"}
    assert we["south"]["sunday"]["Breakfast"] == {"start": "08:00", "end": "09:30"}


def test_scrape_official_hours_from_mocked_page(monkeypatch):
    text_blob = """
    MONDAY through FRIDAY HOURS
    McMAHON
    Breakfast: 7:00am-10:45am
    Lunch: 11:00am-2:00pm
    Dinner: 3:30pm-7:15pm
    WEEKEND HOURS
    NORTHWEST
    Brunch: 10:30am-2:15pm
    Dinner: 3:45pm-7:15pm
    KOSHER & HALAL
    """
    html = f'<html><body><div class="textwidget">{text_blob}</div></body></html>'
    soup = BeautifulSoup(html, "html.parser")

    monkeypatch.setattr(ms, "fetch_soup", lambda session, url: soup)
    out = ms.scrape_official_hours(session=SimpleNamespace())
    assert out["mcmahon"]["monday"]["Lunch"] == {"start": "11:00", "end": "14:00"}
    assert out["northwest"]["saturday"]["Brunch"] == {"start": "10:30", "end": "14:15"}


def test_build_date_url_sets_query_fields():
    base = "https://nutritionanalysis.dds.uconn.edu/shortmenu.aspx?sName=UCONN+Dining+Services&locationNum=16"
    out = ms.build_date_url(base, datetime(2026, 2, 17))
    assert "dtdate=2%2F17%2F2026" in out
    assert "myaction=read" in out
