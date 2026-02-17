# UConn Eats DevTest Guide

## Scope
This test plan validates:
- intent/date-time parsing behavior in the CLI
- cache refresh/staleness logic
- recommendation and fallback behavior
- official hours parsing from UConn hours page format
- menu scraper utility functions

## Test Layers
1. Unit tests (`tests/test_cli.py`)
- query datetime extraction
- meal-window inference from official hours
- cache policy checks
- embedding fallback ranking (with mocked embeddings)

2. Unit tests (`tests/test_menu_scraper.py`)
- time parsing (`7am-10:45am`, `4-7:15pm`)
- grouped hall label parsing
- weekday/weekend hours parsing
- date URL generation
- mocked full hours-page extraction

3. Manual smoke checks
- scrape data and run CLI end-to-end in OpenAI mode
- verify cache reuse on second run
- verify fallback suggestions when direct item is unavailable

## How To Run
```bash
pip install -r requirements.txt
pytest -q
```

Run specific test modules:
```bash
pytest -q tests/test_cli.py
pytest -q tests/test_menu_scraper.py
```

## Manual Smoke Test Commands
Scrape menus:
```bash
python menu_scraper.py --days-ahead 7 --out data/menus_scraped.json
```

Run CLI:
```bash
python uconneats_cli.py --query "I want ramen tomorrow at 6:30 pm" --data-file data/menus_scraped.json
```

## Expected Outcomes
- `pytest` completes with all tests passing.
- first CLI run may refresh cache.
- second CLI run should not refresh cache unless stale.
- when direct match is absent, fallback suggestions should be returned in OpenAI mode.
