## Product Spec v2: UConn Eats (Storrs Dining Decision App)

### 1) Product Summary
UConn Eats is a public, mobile-first app that helps anyone on/near UConn Storrs decide where to eat now by combining:
1. Official dining menus/hall information scraped from public UConn Dining webpages.
2. User cravings, dietary preferences, and allergen constraints.

No SSO is required.

### 2) Core Definitions
1. `Hard constraints`: restrictions that must never be violated (allergen excludes, strict dietary excludes).
2. `Soft preferences`: ranking preferences (craving match, cuisine likes).

### 3) Scope (MVP)
1. Public app (no login required).
2. Hall/menu ingestion via web scraping from official UConn Dining pages.
3. Recommendation engine for "eat now" and "next best time/day".
4. Hall-specific menu lookup flow (e.g., "What's for dinner tonight at South?").
5. Explainable recommendations ("because" reasons).

### 4) Safety and Constraint Rules
1. All allergen and dietary tags are sourced from the official dining website.
2. Recommendation pipeline must apply hard constraints before scoring.
3. If data is missing/uncertain for a requested hard constraint, item/hall is marked `cannot verify` and excluded by default.
4. If no hall passes hard constraints, returns safe fallback messaging, not a ranked unsafe option.

### 5) Data Sources and Contracts
1. Menus and hall info:
- Source: official publicly available UConn Dining pages.
- Method: scheduled web scraping.
- Entry page: `https://dining.uconn.edu/nutrition/`
- Hall discovery selector: `div#pg-60-2 a[href]`
- Hall page parser targets: `.shortmenumeals`, `.shortmenucats`, `.shortmenurecipes`, `.shortmenuproddesc`
2. Hours/open status:
- Source: official hours pages.
- Method: scrape (`https://dining.uconn.edu/hours/`) and persist hall/day meal windows.
- Runtime use: meal-window inference comes from scraped official hours (not fixed time cutoffs).

### 6) Recommendation Logic (v2)
1. Step 1: Eligibility filter.
- Hall is open in target window.
- Hard dietary/allergen constraints pass.
2. Step 2: Score eligible options.
- `TotalScore = w1*FoodMatch + w2*OpenNow + w3*PreferenceFit`
3. Step 3: Explanation output.
- Return top reason codes: item match, open now.
4. Step 4: "Not available now" fallback.
- Search next N days and suggest soonest viable hall/meal.

### 6.1) Menu Lookup Logic
1. If query intent is menu lookup and a hall is identified:
- Resolve target date/time from query (default current ET if omitted).
- Resolve meal from query or inferred meal window.
- Return menu list for that hall/date/meal directly (not ranked recommendations).
2. After listing items, prompt follow-up:
- `Is there something special you want to eat?`

### 7) Fallback Similarity Spec
1. If no exact lookahead match exists, rank alternatives using embedding similarity between requested food and candidate menu items.
2. Source model: OpenAI embedding model configured via environment variable.
3. Always enforce hard allergen/dietary constraints before suggesting alternatives.

### 8) Public Access and Security
1. App is publicly accessible without SSO.
2. API protections required:
- Rate limiting.
- Input validation and query sanitization.
3. Privacy:
- Location is optional and ephemeral by default.
- If analytics retained, apply retention window and deletion policy.

### 9) Non-Functional Requirements
1. Recommendation API p95 latency: < 500 ms (cache warm).
2. Graceful degradation if source website unavailable: serve latest cached data with staleness banner.
3. Accessibility: screen-reader labels and clear text labels.
4. Observability: scrape health, parser break detection, stale-data monitoring.

### 10) MVP Success Metrics
1. Session recommendation action rate: >= 80%.
2. Median time-to-decision: < 20 seconds.
3. Constraint violation rate for hard restrictions: 0%.

### 11) Acceptance Criteria (Launch Gates)
1. Hard constraints never violated in test suite.
2. Scraper successfully ingests all target halls for 14 consecutive days.
3. Hours exceptions can be overridden via admin mechanism within minutes.
4. End-to-end recommendation response includes explanation fields.

### 12) Risks and Mitigations
1. Scraping fragility:
- Mitigation: parser tests, DOM change monitors, fast patch playbook.
2. Official menu substitutions:
- Mitigation: explicit disclaimer, freshness labels, safe-default filtering.
3. Public endpoint abuse:
- Mitigation: rate limits, bot checks, request throttling.
4. Hours disruptions:
- Mitigation: admin overrides + incident runbook.

### 13) Current CLI v1 Implementation Notes
1. Current interface is CLI only (no UI in v1).
2. Implemented scripts:
- `uconneats/menu_scraper.py`: scrapes official nutrition pages and writes normalized JSON.
- `uconneats/cli.py`: loads normalized menu JSON, returns ranked hall recommendations, and handles hall menu lookup queries.
3. Current menu storage format:
- `halls[]`: `hall_id`, `hall_name`, `source_url`
- `menus[]`: `hall_id`, `hall_name`, `source_url`, `menu_date`, `meals`
4. Intent parsing modes:
- OpenAI mode (default): uses `OPENAI_API_KEY`.
- Offline mode: `--offline-intent` for local parsing without API calls.
