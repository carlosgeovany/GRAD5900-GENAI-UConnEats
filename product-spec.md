## Product Spec v3: UConn Eats (Storrs Dining Prompt App)

### 1) Product Summary
UConn Eats is a public, mobile-first prompt app that helps anyone on or near UConn Storrs decide where to eat now by combining:
1. Official dining menus and hall information scraped from public UConn Dining webpages.
2. User cravings, dietary preferences, and allergen constraints.
3. OpenAI-powered intent parsing and conversational response generation.

No SSO is required.

### 2) Core Definitions
1. `Hard constraints`: restrictions that must never be violated, such as allergen excludes and strict dietary excludes.
2. `Soft preferences`: ranking preferences such as craving match or cuisine preference.

### 3) Scope (Current MVP)
1. Public app with no login required.
2. Prompt-based Streamlit interface.
3. Hall and menu ingestion via web scraping from official UConn Dining pages.
4. Official-hours scraping from the UConn Dining hours page.
5. Recommendation engine for "eat now" and "next best time/day".
6. Hall-specific menu lookup flow, such as "What's for dinner tonight at South?"
7. Explainable recommendations in conversational language.

### 4) Safety and Constraint Rules
1. All allergen and dietary tags are sourced from the official dining website.
2. Recommendation pipeline must apply hard constraints before scoring.
3. If data is missing or uncertain for a requested hard constraint, the item is treated as unsafe by default.
4. If no hall passes hard constraints, the app returns safe fallback messaging rather than an unsafe recommendation.

### 5) Data Sources and Contracts
1. Menus and hall info:
- Source: official publicly available UConn Dining pages
- Method: scheduled or on-demand web scraping
- Entry page: `https://dining.uconn.edu/nutrition/`
- Hall discovery selector: `div#pg-60-2 a[href]`
- Hall page parser targets: `.shortmenumeals`, `.shortmenucats`, `.shortmenurecipes`, `.shortmenuproddesc`

2. Hours and open status:
- Source: `https://dining.uconn.edu/hours/`
- Method: scrape the official hours page and persist hall/day meal windows
- Runtime use: meal-window inference comes from scraped official hours, not fixed hardcoded time cutoffs
- Current parser supports the live content structure exposed under the main/article content area of the page

### 6) Recommendation Logic
1. Step 1: eligibility filter
- hall is open in the target window
- hard dietary and allergen constraints pass

2. Step 2: score eligible options
- `TotalScore = w1*FoodMatch + w2*OpenNow + w3*PreferenceFit`

3. Step 3: explanation output
- return top reason codes in conversational form

4. Step 4: not-available-now fallback
- search the next N days and suggest the soonest viable hall/meal

### 6.1) Menu Lookup Logic
1. If query intent is menu lookup and a hall is identified:
- resolve target date/time from the query
- resolve meal from the query or inferred meal window
- return the menu for that hall/date/meal directly

2. After listing items:
- the response should invite a narrower follow-up by diet, allergen, hall, or craving

### 7) Similarity Fallback
1. If no direct lookahead match exists, rank alternatives using embedding similarity between the requested food and candidate menu items.
2. Source model: OpenAI embedding model configured via environment variable.
3. Always enforce hard allergen and dietary constraints before suggesting alternatives.

### 8) Public Access and Security
1. App is publicly accessible without SSO.
2. API protections required:
- rate limiting
- input validation and sanitization
3. Privacy:
- location is optional and ephemeral by default
- if analytics are retained, apply retention and deletion policies

### 9) Non-Functional Requirements
1. Recommendation response should feel fast on a warm cache.
2. If the source site is unavailable, serve the latest cached data and make staleness visible in logs or operator tooling.
3. Accessibility:
- readable text contrast
- clear labels
- mobile-friendly layout
4. Observability:
- scrape health
- parser break detection
- stale-data monitoring

### 10) Current Implementation Notes
1. Primary interface:
- `app.py`: Streamlit prompt UI
- `assets/app.css`: dedicated app styling

2. Shared runtime:
- `uconneats/cli.py`: shared LLM-backed query pipeline and CLI entrypoint

3. Scraping:
- `uconneats/menu_scraper.py`: scrapes official nutrition pages and official hours pages and writes normalized JSON

4. Current storage format:
- `halls[]`: `hall_id`, `hall_name`, `source_url`
- `menus[]`: `hall_id`, `hall_name`, `source_url`, `menu_date`, `meals`
- `official_hours{}`: hall/day/meal windows from the official hours page

5. Runtime mode:
- LLM-only
- requires `OPENAI_API_KEY`
- uses OpenAI for intent parsing and conversational response generation
