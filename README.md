# UConn Eats

UConn Eats is a public dining decision app for UConn Storrs that recommends where to eat based on menu match, dietary constraints, and predicted crowd level.

## Project Goals
- Recommend the best open dining hall for the current meal window.
- Suggest the next best time/day if a desired item is not available now.
- Let users create alerts for future menu availability.
- Show occupancy forecasts using 45-minute time blocks.

## Key Features (MVP)
- Public access (no SSO required)
- Web-scraped menu and dining hall data from official UConn Dining pages
- Hard-filtered allergen/dietary constraint handling
- Explainable recommendation results (why this hall)
- Occupancy forecast labels (`Low`, `Medium`, `High`) plus score
- "Not available now" fallback and alert subscriptions

## Data Sources
- Official UConn Dining public website (menus, hall details, hours)
- Internal SQL occupancy data source (45-minute granularity)

## High-Level Architecture
- Menu Ingestion Service: scrape -> parse -> normalize -> store
- Recommendation API: filter + rank halls by user context
- Forecast Service: occupancy baseline and later ML enhancements
- Notification Service: alerts when requested items appear

## Safety and Constraints
- Allergen/dietary restrictions are treated as hard constraints.
- If data cannot verify a requested hard constraint, options are excluded by default.
- The app surfaces official substitution/allergen disclaimers and advises on-site verification for severe allergies.

## Repository Files
- `product-spec.md`: full Product Spec v2
- `README.md`: repository overview and usage context

## Planned Phases
1. Phase 0: scraper + normalized storage + initial UI
2. Phase 1 (MVP): full halls, open-now logic, baseline occupancy, alerts
3. Phase 2: improved NLP matching, stronger forecasting, personalization

## Status
Specification phase complete. Implementation scaffolding next.

## CLI Starter (Current Version)
This repository now includes a CLI prototype:
- `uconneats_cli.py`: recommend dining halls from query + constraints
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
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=
UCONN_EATS_DB_URL=
```
If your key requires a custom endpoint, set:
`OPENAI_BASE_URL=https://us.api.openai.com/v1`
4. Scrape live menus from the official nutrition site:
```bash
python menu_scraper.py --days-ahead 7 --out data/menus_scraped.json
```
5. Run recommendations with OpenAI intent parsing:
```bash
python uconneats_cli.py --query "I want pho and low crowd for lunch today" --avoid-crowds --data-file data/menus_scraped.json
```
6. Optional: run without OpenAI using local parsing only:
```bash
python uconneats_cli.py --query "I want ramen tomorrow at 6:30 pm" --offline-intent --data-file data/menus_scraped.json
```

### Additional Commands
Scrape selected halls only:
```bash
python menu_scraper.py --days-ahead 2 --halls "south,northwest" --out data/menus_scraped.json
```

Run with sample local data (no scraper):
```bash
python uconneats_cli.py --query "I want pho" --offline-intent --data-file data/sample_menus.json
```

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

### SQL Occupancy Hook
Set `UCONN_EATS_DB_URL` later to connect the occupancy forecaster to your SQL source.  
Current implementation uses a deterministic placeholder baseline at 45-minute granularity.
