# UConn Eats

UConn Eats is a public dining decision app for UConn Storrs that recommends where to eat based on menu match and dietary constraints.

## Project Goals
- Recommend the best open dining hall for the current meal window.
- Support direct hall menu lookup queries (e.g., "What's for dinner tonight at South?").
- Suggest the next best time/day if a desired item is not available now.
- Let users create alerts for future menu availability.

## Key Features (MVP)
- Public access (no SSO required)
- Web-scraped menu and dining hall data from official UConn Dining pages
- Hard-filtered allergen/dietary constraint handling
- Explainable recommendation results (why this hall)
- Hall-specific menu listing when user asks "what's on the menu at <hall>"
- "Not available now" fallback and alert subscriptions

## Data Sources
- Official UConn Dining public website (menus, hall details, hours)

## High-Level Architecture
- Menu Ingestion Service: scrape -> parse -> normalize -> store
- Recommendation API: filter + rank halls by user context
- Notification Service: alerts when requested items appear

## Safety and Constraints
- Allergen/dietary restrictions are treated as hard constraints.
- If data cannot verify a requested hard constraint, options are excluded by default.
- The app surfaces official substitution/allergen disclaimers and advises on-site verification for severe allergies.

## Repository Files
- `product-spec.md`: full Product Spec v2
- `README.md`: repository overview and usage context
- `DEVTEST.md`: developer test plan and test execution guide

## Planned Phases
1. Phase 0: scraper + normalized storage + initial UI
2. Phase 1 (MVP): full halls, open-now logic, alerts
3. Phase 2: improved NLP matching, stronger forecasting, personalization

## Status
Specification phase complete. Implementation scaffolding next.

## CLI Starter (Current Version)
This repository now includes a CLI prototype:
- `uconneats/cli.py`: recommend dining halls from query + constraints and support hall menu lookup
- `data/sample_menus.json`: normalized sample menu data for local testing
- `.env.example`: environment variable template
- `requirements.txt`: Python dependencies

### How To Run
1. Open a terminal in this repository root and install dependencies:
```bash
pip install -r requirements.txt
```
2. Create an environment file:
```bash
copy .env.example .env
```
3. Edit `.env` and set:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.3-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=
```
If your key requires a custom endpoint, set:
`OPENAI_BASE_URL=https://us.api.openai.com/v1`
4. Scrape live menus from the official nutrition site:
```bash
python -m uconneats.menu_scraper --days-ahead 7 --out data/menus_scraped.json
```
5. Run recommendations with OpenAI intent parsing:
```bash
python -m uconneats.cli --query "I want pho for lunch today" --data-file data/menus_scraped.json
```
6. Optional: run without OpenAI using local parsing only:
```bash
python -m uconneats.cli --query "I want ramen tomorrow at 6:30 pm" --offline-intent --data-file data/menus_scraped.json
```

### Additional Commands
Scrape selected halls only:
```bash
python -m uconneats.menu_scraper --days-ahead 2 --halls "south,northwest" --out data/menus_scraped.json
```

Run with sample local data (no scraper):
```bash
python -m uconneats.cli --query "I want pho" --offline-intent --data-file data/sample_menus.json
```

Menu lookup example:
```bash
python -m uconneats.cli --query "Whats for dinner tonight at South?" --data-file data/menus_scraped.json
```
When the query is a menu lookup and a hall is identified, the app returns the menu items for that hall/date/meal and asks:
`Is there something special you want to eat?`

Automatic cache refresh behavior:
- Default `--data-file` is `data/menus_scraped.json`.
- On each query, the app refreshes only if cache is missing, unreadable, older than `--max-cache-hours` (default 24), or missing today's ET menu.
- Otherwise it uses cached menus (no web scrape on that query).
- Cached payload includes official hours scraped from `https://dining.uconn.edu/hours/`, used for hall/day meal-window inference (no hardcoded meal cutoff).

### Useful Options
- `--avoid-allergens "peanuts,sesame"`
- `--diets "vegetarian"`
- `--meal "Lunch"`
- `--top-k 3`
- `--days-ahead 7`
- `--max-cache-hours 24`
- `--offline-intent` (skip OpenAI and use local parsing)

Date/time behavior:
- The app extracts target date/time from `--query` when present.
- If not present, it defaults to current `America/New_York` system time.

Fallback behavior:
- If no direct match is found in lookahead, the app can suggest semantically similar dishes using OpenAI embeddings.
- This similarity fallback requires OpenAI mode (not `--offline-intent`).
