# UConn Eats DevTest Guide

## Scope
This test plan validates:
- LLM-backed query handling through the shared runtime pipeline
- cache refresh and staleness behavior
- recommendation and fallback behavior
- official hours parsing from the live UConn hours page structure
- menu scraper utility functions
- Streamlit app import and shared query-path stability

## Test Layers
1. Unit tests in `tests/test_cli.py`
- meal-window inference from official hours
- cache policy checks
- embedding fallback ranking with mocked embeddings
- conversational formatter coverage for shared responses

2. Unit tests in `tests/test_menu_scraper.py`
- time parsing such as `7am-10:45am` and `4-7:15pm`
- grouped hall label parsing
- weekday and weekend hours parsing
- date URL generation
- mocked hours-page extraction
- parsing against the current `main/article`-style hours page structure

3. Manual smoke checks
- scrape data and run the Streamlit UI in OpenAI mode
- run the CLI against the same shared query engine
- verify cache reuse on a second run
- verify fallback suggestions when a direct match is unavailable

## How To Run
```bash
pip install -r requirements.txt
python -m pytest -q
```

Run specific modules:
```bash
python -m pytest -q tests/test_cli.py
python -m pytest -q tests/test_menu_scraper.py
```

## Manual Smoke Test Commands
Scrape menus:
```bash
python -m uconneats.menu_scraper --days-ahead 7 --out data/menus_scraped.json
```

Run the prompt UI:
```bash
streamlit run app.py
```

Then try prompts such as:
- `What's for dinner tonight at South?`
- `What are South hours tomorrow?`
- `Does chicken ramen contain soy?`

Run the CLI:
```bash
python -m uconneats.cli --query "I want ramen tomorrow at 6:30 pm" --data-file data/menus_scraped.json
```

## Expected Outcomes
- `python -m pytest -q` completes with all tests passing
- the first app or CLI run may refresh the cache
- the second run should reuse the cache unless it is stale
- hours queries should return conversational hall-hour summaries from the official UConn Dining hours page
- when a direct match is absent, fallback suggestions should still be returned in OpenAI mode
