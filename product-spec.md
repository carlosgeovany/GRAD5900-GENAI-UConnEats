## Product Spec v2: UConn Eats (Storrs Dining Decision App)

### 1) Product Summary
UConn Eats is a public, mobile-first app that helps anyone on/near UConn Storrs decide where to eat now by combining:
1. Official dining menus/hall information scraped from public UConn Dining webpages.
2. Predicted hall occupancy (crowd level) at 45-minute intervals.
3. User cravings, dietary preferences, and allergen constraints.

No SSO is required.

### 2) Core Definitions
1. `Occupancy forecast`: predicted crowd level for a dining hall/time block using 45-minute granularity.
2. `Hard constraints`: restrictions that must never be violated (allergen excludes, strict dietary excludes).
3. `Soft preferences`: ranking preferences (craving match, distance, low crowd, cuisine likes).

### 3) Scope (MVP)
1. Public app (no login required).
2. Hall/menu ingestion via web scraping from official UConn Dining pages.
3. Recommendation engine for "eat now" and "next best time/day".
4. Alerting for item availability.
5. Crowd forecasts using occupancy baseline at 45-minute blocks.
6. Explainable recommendations ("because" reasons).

### 4) Safety and Constraint Rules
1. All allergen and dietary tags are sourced from the official dining website.
2. Recommendation pipeline must apply hard constraints before scoring.
3. If data is missing/uncertain for a requested hard constraint, item/hall is marked `cannot verify` and excluded by default.
4. UI must show official substitution/disclaimer notice and "verify on-site for severe allergies".
5. If no hall passes hard constraints, app returns safe fallback messaging, not a ranked unsafe option.

### 5) Data Sources and Contracts
1. Menus and hall info:
- Source: official publicly available UConn Dining pages.
- Method: scheduled web scraping.
- Update cadence: every 1-3 hours.
- Persist `last_updated_at` and source URL for traceability.
2. Hours/open status:
- Source: official hours pages.
- Method: scrape + admin override table for emergency changes.
3. Occupancy:
- Source: your SQL DB.
- Grain: 45-minute blocks in America/New_York.
- Target variable: occupancy index (0-100) derived from count/occupancy input.

### 6) Recommendation Logic (v2)
1. Step 1: Eligibility filter.
- Hall is open in target window.
- Hard dietary/allergen constraints pass.
- Data freshness within configured SLA.
2. Step 2: Score eligible options.
- `TotalScore = w1*FoodMatch + w2*Distance + w3*Crowd + w4*PreferenceFit`
3. Step 3: Explanation output.
- Return top reason codes: item match, open now, walk time, predicted crowd.
4. Step 4: "Not available now" fallback.
- Search next N days and suggest soonest viable hall/meal.
- Offer alert subscription.

### 7) Occupancy Forecasting Spec
1. Time granularity: 45-minute blocks.
2. MVP baseline:
- Rolling 4-week average by hall + day-of-week + 45-min block.
3. Output:
- Occupancy index (0-100), band (`Low/Medium/High`), confidence level.
4. Phase 2:
- Add semester-week, weather, events, exam periods, anomaly handling.

### 8) Public Access and Security
1. App is publicly accessible without SSO.
2. API protections required:
- Rate limiting.
- Basic abuse prevention on alerts.
- Input validation and query sanitization.
3. Privacy:
- Location is optional and ephemeral by default.
- If analytics retained, apply retention window and deletion policy.

### 9) Non-Functional Requirements
1. Recommendation API p95 latency: < 500 ms (cache warm).
2. Graceful degradation if source website unavailable: serve latest cached data with staleness banner.
3. Accessibility: screen-reader labels, non-color-only crowd cues.
4. Observability: scrape health, parser break detection, stale-data alerts.

### 10) MVP Success Metrics
1. Session recommendation action rate (`Navigate` or `Set Alert`): >= 80%.
2. Median time-to-decision: < 20 seconds.
3. Alert conversion in suggested window: >= 30%.
4. Constraint violation rate for hard restrictions: 0%.
5. Occupancy forecast error improves vs naive global baseline.

### 11) Acceptance Criteria (Launch Gates)
1. Hard constraints never violated in test suite.
2. Scraper successfully ingests all target halls for 14 consecutive days.
3. Hours exceptions can be overridden via admin mechanism within minutes.
4. Occupancy forecast pipeline produces complete 45-min blocks for all halls.
5. End-to-end recommendation response includes explanation fields.

### 12) Risks and Mitigations
1. Scraping fragility:
- Mitigation: parser tests, DOM change monitors, fast patch playbook.
2. Official menu substitutions:
- Mitigation: explicit disclaimer, freshness labels, safe-default filtering.
3. Public endpoint abuse:
- Mitigation: rate limits, bot checks, alert throttling.
4. Hours disruptions:
- Mitigation: admin overrides + incident runbook.
