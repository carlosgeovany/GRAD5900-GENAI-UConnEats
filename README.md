# UConn Eats

UConn Eats is a CLI assistant for UConn Storrs dining halls.  
It answers natural-language questions about menus, hours, allergens, and food options.

## Current Functionality
- Hall menu lookup from plain-English queries
- Dining hall hours lookup (from official UConn hours page)
- Allergen-aware answers (contains/without style questions)
- Diet-based option listing (for example vegetarian/vegan)
- General food recommendation and next-best fallback when an item is unavailable
- GPT-generated user-facing replies based on structured internal results
- Automatic menu cache refresh (no need to scrape every query)

## Query Types Supported
- `Menu`: "What's for dinner tonight at South?"
- `Hours`: "What are South hours tomorrow?"
- `Allergens`: "Does chicken ramen contain soy?"
- `Diet options`: "What vegetarian options are there for tomorrow at lunch?"
- `Recommendation`: "I want Mexican but I'm allergic to peanuts."

## Project Structure
- `uconneats/cli.py`: main CLI app
- `uconneats/menu_scraper.py`: menu/hours scraper and normalization
- `data/`: cached and sample data
- `tests/`: test suite
- `product-spec.md`: product specification
- `DEVTEST.md`: developer testing guide

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create env file:
```bash
copy .env.example .env
```
3. Set env values:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.3-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=
```

If your account/endpoint does not support the configured model, the app automatically falls back to available models.

## Run
Scrape data:
```bash
python -m uconneats.menu_scraper --out data/menus_scraped.json
```

Run CLI:
```bash
python -m uconneats.cli --query "What's for dinner tonight at South?" --data-file data/menus_scraped.json
```

Offline mode (no OpenAI call):
```bash
python -m uconneats.cli --query "What are South hours tomorrow?" --offline-intent --data-file data/menus_scraped.json
```

## Examples (Input -> Output Style)
1. Menu lookup  
Input:
```bash
python -m uconneats.cli --query "What's for dinner tonight at South?"
```
Output style:
```text
South | 2026-02-17 | Dinner
1. ...
2. ...
Is there something special you want to eat?
```

2. Hours lookup  
Input:
```bash
python -m uconneats.cli --query "What are South hours tomorrow?"
```
Output style:
```text
Dining hall hours for 2026-02-18:
South:
  - Breakfast: 07:00 - 10:45
  - Lunch: 11:00 - 15:00
  - Dinner: 16:30 - 19:15
Do you want menu options for one of these halls?
```

3. Allergen question  
Input:
```bash
python -m uconneats.cli --query "Does chicken ramen contain soy?"
```
Output style:
```text
Items mentioning soy:
1. ...
Possible options without soy:
1. ...
Would you like me to narrow this by hall or meal?
```

4. Diet options  
Input:
```bash
python -m uconneats.cli --query "What vegetarian options there are for tomorrow at lunch?"
```
Output style:
```text
Vegetarian options for 2026-02-18 (Lunch):
1. ...
2. ...
Is there something special you want to eat?
```

5. Unavailable item fallback  
Input:
```bash
python -m uconneats.cli --query "I want ramen tonight"
```
Output style:
```text
I couldn't find that right now.
The next good match is ... on ... (...).
Want me to suggest something similar that is available sooner?
```

## Cache Behavior
- Default data file: `data/menus_scraped.json`
- Auto-refresh triggers when cache is missing, unreadable, stale, or missing current ET date
- Otherwise cache is reused for fast responses

## Testing
```bash
pytest -q
```
