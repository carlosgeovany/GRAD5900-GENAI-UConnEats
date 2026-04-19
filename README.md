# UConn Eats

UConn Eats is a prompt-based dining assistant for UConn Storrs dining halls. It answers natural-language questions about menus, hours, allergens, and food options through a Streamlit web UI and uses the OpenAI API for both intent understanding and reply generation.

## Current State
- Primary interface: Streamlit chat app
- Secondary interface: CLI entrypoint for debugging and smoke tests
- Runtime mode: LLM-only
- Data sources:
  - Menus and nutrition pages from `https://dining.uconn.edu/nutrition/`
  - Hours from `https://dining.uconn.edu/hours/`
- Styling: dedicated stylesheet in `assets/app.css`

## What the App Can Do
- Menu lookup from plain-English prompts
- Dining hall hours lookup using the official UConn Dining hours page
- Allergen-aware answers
- Diet-based option listing such as vegetarian or vegan options
- General recommendation and next-best fallback when an item is unavailable
- Conversational responses generated from structured dining results
- Automatic cache refresh when data is stale or missing current ET menu data

## Query Examples
- `What's for dinner tonight at South?`
- `What are South hours tomorrow?`
- `Does chicken ramen contain soy?`
- `What vegetarian options are there for tomorrow at lunch?`
- `I want Mexican but I'm allergic to peanuts.`

## Project Structure
- `app.py`: Streamlit prompt UI
- `assets/app.css`: dedicated app styling
- `uconneats/cli.py`: shared query pipeline and CLI entrypoint
- `uconneats/menu_scraper.py`: menu and hours scraper
- `data/`: cached and sample data
- `tests/`: test suite
- `product-spec.md`: project specification
- `DEVTEST.md`: developer testing guide

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create the env file:
```bash
copy .env.example .env
```
3. Set environment values:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=
```

`OPENAI_API_KEY` is required in the current version. The project no longer includes an offline intent mode.

If your endpoint does not support the configured model, the runtime falls back to another supported OpenAI model when possible.

## Run
Prompt UI:
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal.

Refresh scraped data manually:
```bash
python -m uconneats.menu_scraper --out data/menus_scraped.json
```

Run the CLI directly:
```bash
python -m uconneats.cli --query "What's for dinner tonight at South?" --data-file data/menus_scraped.json
```

The Streamlit app is the intended day-to-day interface. The CLI is mainly useful for debugging the shared query engine.

## Output Style
Menu lookup:
```text
South is serving dinner on 2026-02-17. Some options I found are ... If you want, I can narrow that down by diet, allergen, or craving.
```

Hours lookup:
```text
South is open for Breakfast from 07:00 to 10:45, Lunch from 11:00 to 15:00, and Dinner from 16:30 to 19:15. If you want, I can also show what one of those halls is serving.
```

Allergen lookup:
```text
I found a few items that mention soy, and I also found possible options without soy. If you want, I can narrow that down by hall or meal.
```

Diet options:
```text
I found some vegetarian options for 2026-02-18 during lunch. A few good ones are ... If you want, I can also filter those by hall or allergen.
```

Unavailable item fallback:
```text
I couldn't find that exact match right now, but the next good option is ... If you want, I can also suggest similar dishes that show up sooner.
```

## Data and Hours
- Default cache file: `data/menus_scraped.json`
- Menu source: `https://dining.uconn.edu/nutrition/`
- Hours source: `https://dining.uconn.edu/hours/`
- The hours parser supports the current content structure on the official UConn Dining hours page, including weekday and weekend hall windows.

## Cache Behavior
- Default data file: `data/menus_scraped.json`
- Auto-refresh triggers when cache is missing, unreadable, stale, or missing current ET menu data
- Otherwise cache is reused for fast responses

## Testing
```bash
python -m pytest -q
```
